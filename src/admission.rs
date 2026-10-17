use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Capacity stays owned by native work until settlement, not by the HTTP task.
#[derive(Clone)]
pub struct Admission(Arc<Semaphore>);

impl Default for Admission {
    fn default() -> Self {
        Self(Arc::new(Semaphore::new(1)))
    }
}

impl Admission {
    pub fn try_admit(&self) -> Option<ActiveWork> {
        self.0
            .clone()
            .try_acquire_owned()
            .ok()
            .map(|permit| ActiveWork {
                _permit: permit,
                cancelled: Arc::new(AtomicBool::new(false)),
            })
    }
}

pub struct ActiveWork {
    _permit: OwnedSemaphorePermit,
    cancelled: Arc<AtomicBool>,
}

impl ActiveWork {
    pub fn disconnect_guard(&self) -> DisconnectGuard {
        DisconnectGuard(self.cancelled.clone())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub struct DisconnectGuard(Arc<AtomicBool>);

impl Drop for DisconnectGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnect_does_not_release_native_capacity() {
        let admission = Admission::default();
        let work = admission.try_admit().unwrap();
        let consumer = work.disconnect_guard();
        assert!(admission.try_admit().is_none());
        drop(consumer);
        assert!(work.is_cancelled());
        assert!(admission.try_admit().is_none());
        drop(work);
        assert!(admission.try_admit().is_some());
    }
}
