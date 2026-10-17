use crate::{
    request::ChatRequest,
    worker::{Event, Worker},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone)]
pub struct Endpoint {
    pub worker: Worker,
    pub model: Arc<str>,
}
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub fn router(endpoint: Endpoint) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(endpoint)
}
async fn models(State(state): State<Endpoint>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":state.model.as_ref(),"object":"model","owned_by":"eredu"}]}),
    )
}
fn error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":"eredu_error"}})),
    )
        .into_response()
}
async fn chat(State(state): State<Endpoint>, Json(request): Json<ChatRequest>) -> Response {
    let request = match request.prepare(&state.model) {
        Ok(request) => request,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let streaming = request.stream;
    let mut generation = match state.worker.submit(request) {
        Ok(generation) => generation,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
    };
    let id = format!(
        "chatcmpl-eredu-{}",
        REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    );
    let model = state.model.to_string();
    if streaming {
        let stream = async_stream::stream! {
            // Move the whole Generation: edition-2021 disjoint capture would otherwise take only
            // `events`, dropping the DisconnectGuard on handler return and cancelling the stream.
            let mut generation = generation;
            yield Ok::<_, Infallible>(SseEvent::default().data(chunk(&id, &model, json!({"role":"assistant"}), None).to_string()));
            let mut terminal = false;
            while let Some(event) = generation.events.recv().await {
                let payload = match event {
                    Event::Text(text) => chunk(&id, &model, json!({"content":text}), None),
                    Event::Finished(reason) => { terminal = true; chunk(&id, &model, json!({}), Some(reason)) },
                    Event::Error(message) => { terminal = true; json!({"error":{"message":message}}) },
                };
                yield Ok(SseEvent::default().data(payload.to_string()));
                if terminal { break; }
            }
            if terminal { yield Ok(SseEvent::default().data("[DONE]")); }
            else { yield Ok(SseEvent::default().data(json!({"error":{"message":"worker exited without terminal event"}}).to_string())); }
        };
        return Sse::new(stream).into_response();
    }
    let mut text = String::new();
    while let Some(event) = generation.events.recv().await {
        match event {
            Event::Text(delta) => {
                if text.len() + delta.len() > 1024 * 1024 { return error(StatusCode::BAD_GATEWAY, "output exceeds limit".into()); }
                text.push_str(&delta);
            }
            Event::Finished(reason) => return Json(json!({"id":id,"object":"chat.completion","model":model,"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason}]})).into_response(),
            Event::Error(message) => return error(StatusCode::BAD_GATEWAY, message),
        }
    }
    error(
        StatusCode::BAD_GATEWAY,
        "worker exited without terminal event".into(),
    )
}
fn chunk(id: &str, model: &str, delta: Value, reason: Option<&str>) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","model":model,"choices":[{"index":0,"delta":delta,"finish_reason":reason}]})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        admission::ActiveWork,
        worker::{Backend, publish},
    };
    use anyhow::Result;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;
    struct Mock;
    impl Backend for Mock {
        fn generate(
            &mut self,
            _: ChatRequest,
            _: &ActiveWork,
            output: &tokio::sync::mpsc::Sender<Event>,
        ) -> Result<()> {
            publish(output, Event::Text("héllo\nworld".into()))?;
            publish(output, Event::Finished("stop"))
        }
        fn settle(&mut self) -> Result<()> {
            Ok(())
        }
    }
    async fn call(stream: bool) -> (StatusCode, String) {
        let worker = Worker::start(|| Ok(Mock)).await.unwrap();
        let app = router(Endpoint {
            worker,
            model: "local".into(),
        });
        let response = app.oneshot(Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json!({"model":"local","messages":[{"role":"user","content":"hi"}],"stream":stream}).to_string())).unwrap()).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 65536).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }
    #[tokio::test]
    async fn nonstream_returns_text_and_terminal_reason() {
        let (status, body) = call(false).await;
        assert_eq!(status, StatusCode::OK);
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "héllo\nworld");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
    }
    #[tokio::test]
    async fn sse_has_json_frames_and_done_after_terminal() {
        let (status, body) = call(true).await;
        assert_eq!(status, StatusCode::OK);
        let frames: Vec<_> = body.split("\n\n").filter(|s| !s.is_empty()).collect();
        assert_eq!(frames.last().unwrap(), &"data: [DONE]");
        for frame in &frames[..frames.len() - 1] {
            let _: Value = serde_json::from_str(frame.strip_prefix("data: ").unwrap()).unwrap();
        }
        assert!(frames[frames.len() - 2].contains("\"finish_reason\":\"stop\""));
    }
}
