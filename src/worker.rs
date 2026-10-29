//! Bounded owner-thread transport. Backend construction and settlement stay on the owner.
use crate::{
    admission::{ActiveWork, Admission, DisconnectGuard},
    request::ChatRequest,
};
use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;

pub enum Event {
    Text(String),
    ToolStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolArguments {
        index: usize,
        fragment: String,
    },
    Finished(&'static str),
    Error(String),
}

pub trait Backend: 'static {
    fn generate(
        &mut self,
        request: ChatRequest,
        work: &ActiveWork,
        output: &mpsc::Sender<Event>,
    ) -> Result<&'static str>;
    /// Must establish device completion before the request lease can be released.
    fn settle(&mut self) -> Result<()>;
}

struct Job {
    request: ChatRequest,
    work: ActiveWork,
    output: mpsc::Sender<Event>,
}

#[derive(Clone)]
pub struct Worker {
    jobs: mpsc::Sender<Job>,
    admission: Admission,
}

pub struct Generation {
    pub events: mpsc::Receiver<Event>,
    _disconnect: DisconnectGuard,
}

impl Worker {
    pub async fn start<B: Backend>(
        factory: impl FnOnce() -> Result<B> + Send + 'static,
    ) -> Result<Self> {
        let (jobs, mut input) = mpsc::channel::<Job>(1);
        let (ready, loaded) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("eredu-owner".into())
            .spawn(move || {
                let mut backend = match factory() {
                    Ok(backend) => backend,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                while let Some(job) = input.blocking_recv() {
                    let result = backend.generate(job.request, &job.work, &job.output);
                    // Hold the final event until settlement releases admission. A client may
                    // immediately submit its tool result after observing completion.
                    let settled = backend.settle();
                    drop(job.work);
                    match (result, settled) {
                        (_, Err(error)) => {
                            let _ = job.output.try_send(Event::Error(error.to_string()));
                            break;
                        }
                        (Err(error), Ok(())) => {
                            let _ = job.output.try_send(Event::Error(error.to_string()));
                        }
                        (Ok(reason), Ok(())) => {
                            let _ = job.output.try_send(Event::Finished(reason));
                        }
                    }
                }
            })?;
        loaded
            .await
            .map_err(|_| anyhow!("owner exited during loading"))??;
        Ok(Self {
            jobs,
            admission: Admission::default(),
        })
    }

    pub fn submit(&self, request: ChatRequest) -> Result<Generation> {
        let work = self
            .admission
            .try_admit()
            .ok_or_else(|| anyhow!("worker busy"))?;
        let disconnect = work.disconnect_guard();
        let (output, events) = mpsc::channel(32);
        if self
            .jobs
            .try_send(Job {
                request,
                work,
                output,
            })
            .is_err()
        {
            bail!("worker unavailable");
        }
        Ok(Generation {
            events,
            _disconnect: disconnect,
        })
    }
}

/// Never block a native callback on a stalled HTTP consumer.
pub fn publish(output: &mpsc::Sender<Event>, event: Event) -> Result<()> {
    output
        .try_send(event)
        .map_err(|_| anyhow!("consumer disconnected or output buffer full"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct Mock(Arc<AtomicBool>);
    impl Backend for Mock {
        fn generate(
            &mut self,
            _: ChatRequest,
            work: &ActiveWork,
            output: &mpsc::Sender<Event>,
        ) -> Result<&'static str> {
            while !work.is_cancelled() {
                if publish(output, Event::Text("hello".into())).is_err() {
                    break;
                }
            }
            Ok("stop")
        }
        fn settle(&mut self) -> Result<()> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }
    fn request() -> ChatRequest {
        serde_json::from_value(json!({"model":"local","messages":[{"role":"user","content":"hi"}]}))
            .unwrap()
    }
    #[tokio::test]
    async fn bounded_output_settles_and_allows_next_request() {
        let settled = Arc::new(AtomicBool::new(false));
        let state = settled.clone();
        let worker = Worker::start(move || Ok(Mock(state))).await.unwrap();
        let mut generation = worker.submit(request()).unwrap();
        while generation.events.recv().await.is_some() {}
        assert!(settled.load(Ordering::Acquire));
        drop(generation);
        let next = worker.submit(request()).unwrap();
        drop(next);
    }
    #[tokio::test]
    async fn loading_failure_is_not_ready() {
        assert!(
            Worker::start(|| -> Result<Mock> { bail!("load failed") })
                .await
                .is_err()
        );
    }
    struct Immediate;
    impl Backend for Immediate {
        fn generate(
            &mut self,
            _: ChatRequest,
            _: &ActiveWork,
            _: &mpsc::Sender<Event>,
        ) -> Result<&'static str> {
            Ok("stop")
        }
        fn settle(&mut self) -> Result<()> {
            Ok(())
        }
    }
    #[tokio::test]
    async fn completion_releases_admission_before_immediate_followup() {
        let worker = Worker::start(|| Ok(Immediate)).await.unwrap();
        for _ in 0..100 {
            let mut generation = worker.submit(request()).unwrap();
            assert!(matches!(
                generation.events.recv().await,
                Some(Event::Finished("stop"))
            ));
            // Do not wait for channel closure before starting the next request.
        }
    }
}
