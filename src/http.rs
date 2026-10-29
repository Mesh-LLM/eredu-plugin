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
                    Event::ToolStart { index, id: call_id, name } => chunk(&id, &model, json!({"tool_calls":[{"index":index,"id":call_id,"type":"function","function":{"name":name,"arguments":""}}]}), None),
                    Event::ToolArguments { index, fragment } => chunk(&id, &model, json!({"tool_calls":[{"index":index,"function":{"arguments":fragment}}]}), None),
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
    let mut calls: Vec<Value> = Vec::new();
    let mut output_bytes = 0usize;
    while let Some(event) = generation.events.recv().await {
        match event {
            Event::ToolStart { index, id, name } => {
                if index != calls.len() || calls.len() >= 128 {
                    return error(StatusCode::BAD_GATEWAY, "invalid tool call index".into());
                }
                output_bytes += id.len() + name.len();
                calls.push(
                    json!({"id":id,"type":"function","function":{"name":name,"arguments":""}}),
                );
            }
            Event::ToolArguments { index, fragment } => {
                let Some(call) = calls.get_mut(index) else {
                    return error(
                        StatusCode::BAD_GATEWAY,
                        "arguments without tool call".into(),
                    );
                };
                output_bytes += fragment.len();
                let args = call["function"]["arguments"].as_str().unwrap().to_owned() + &fragment;
                call["function"]["arguments"] = Value::String(args);
            }
            Event::Text(delta) => {
                output_bytes += delta.len();
                text.push_str(&delta);
            }
            Event::Finished(reason) => {
                let mut message = json!({"role":"assistant","content":text});
                if !calls.is_empty() {
                    if text.is_empty() {
                        message["content"] = Value::Null;
                    }
                    message["tool_calls"] = json!(calls);
                }
                return Json(json!({"id":id,"object":"chat.completion","model":model,"choices":[{"index":0,"message":message,"finish_reason":reason}]})).into_response();
            }
            Event::Error(message) => return error(StatusCode::BAD_GATEWAY, message),
        }
        if output_bytes > 1024 * 1024 {
            return error(StatusCode::BAD_GATEWAY, "output exceeds limit".into());
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
        ) -> Result<&'static str> {
            publish(output, Event::Text("héllo\nworld".into()))?;
            Ok("stop")
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
    struct ToolMock;
    impl Backend for ToolMock {
        fn generate(
            &mut self,
            request: ChatRequest,
            _: &ActiveWork,
            output: &tokio::sync::mpsc::Sender<Event>,
        ) -> Result<&'static str> {
            if request.messages.iter().any(|m| m["role"] == "tool") {
                assert_eq!(request.messages[1]["tool_calls"][0]["id"], "call_7");
                assert_eq!(
                    request.messages[1]["tool_calls"][0]["function"]["arguments"],
                    json!({"city":"Bogotá"})
                );
                assert_eq!(request.messages[2]["tool_call_id"], "call_7");
                publish(output, Event::Text("Sunny in Bogotá".into()))?;
                return Ok("stop");
            }
            assert_eq!(request.tool_choice.as_deref(), Some("required"));
            assert_eq!(request.tools[0]["function"]["name"], "weather");
            publish(
                output,
                Event::ToolStart {
                    index: 0,
                    id: "call_7".into(),
                    name: "weather".into(),
                },
            )?;
            publish(
                output,
                Event::ToolArguments {
                    index: 0,
                    fragment: "{\"city\":".into(),
                },
            )?;
            publish(
                output,
                Event::ToolArguments {
                    index: 0,
                    fragment: "\"Bogotá\"}".into(),
                },
            )?;
            Ok("tool_calls")
        }
        fn settle(&mut self) -> Result<()> {
            Ok(())
        }
    }
    async fn tool_call(body: Value) -> (StatusCode, String) {
        let worker = Worker::start(|| Ok(ToolMock)).await.unwrap();
        let app = router(Endpoint {
            worker,
            model: "local".into(),
        });
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }
    fn tool_request(stream: bool) -> Value {
        json!({"model":"local","stream":stream,"messages":[{"role":"user","content":"Weather?"}],
            "tool_choice":"required","tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]})
    }
    #[tokio::test]
    async fn tool_json_and_result_replay_round_trip() {
        let (status, body) = tool_call(tool_request(false)).await;
        assert_eq!(status, StatusCode::OK);
        let value: Value = serde_json::from_str(&body).unwrap();
        let message = value["choices"][0]["message"].clone();
        assert!(message["content"].is_null());
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Bogotá\"}"
        );
        let (status, body) = tool_call(json!({"model":"local","messages":[
            {"role":"user","content":"Weather?"}, message,
            {"role":"tool","tool_call_id":"call_7","content":"sunny"}]}))
        .await;
        assert_eq!(status, StatusCode::OK);
        let answer: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            answer["choices"][0]["message"]["content"],
            "Sunny in Bogotá"
        );
    }
    #[tokio::test]
    async fn tool_sse_fragments_reconstruct_openai_call() {
        let (status, body) = tool_call(tool_request(true)).await;
        assert_eq!(status, StatusCode::OK);
        let mut arguments = String::new();
        let mut id = String::new();
        let mut name = String::new();
        let mut finish = String::new();
        for frame in body.split("\n\n").filter_map(|f| f.strip_prefix("data: ")) {
            if frame == "[DONE]" {
                continue;
            }
            let v: Value = serde_json::from_str(frame).unwrap();
            let call = &v["choices"][0]["delta"]["tool_calls"][0];
            if !call.is_null() {
                assert_eq!(call["index"], 0);
                if let Some(s) = call["id"].as_str() {
                    id.push_str(s);
                }
                if let Some(s) = call["function"]["name"].as_str() {
                    name.push_str(s);
                }
                if let Some(s) = call["function"]["arguments"].as_str() {
                    arguments.push_str(s);
                }
            }
            if let Some(s) = v["choices"][0]["finish_reason"].as_str() {
                finish = s.into();
            }
        }
        assert_eq!(id, "call_7");
        assert_eq!(name, "weather");
        assert_eq!(arguments, "{\"city\":\"Bogotá\"}");
        assert_eq!(finish, "tool_calls");
        assert!(body.ends_with("data: [DONE]\n\n"));
    }
}
