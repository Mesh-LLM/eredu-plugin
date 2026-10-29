use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    pub enable_thinking: Option<bool>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

impl ChatRequest {
    pub fn prepare(mut self, model_id: &str) -> Result<Self> {
        if self.model != model_id {
            bail!("unknown model");
        }
        if self.messages.is_empty() || self.messages.len() > 256 {
            bail!("messages must contain 1 to 256 entries");
        }
        if self.max_tokens.is_some_and(|n| n == 0 || n > 4096) {
            bail!("max_tokens must be between 1 and 4096");
        }
        if self.temperature.is_some_and(|t| !t.is_finite() || t < 0.0) {
            bail!("temperature must be finite and nonnegative");
        }
        if self.tools.len() > 128 {
            bail!("at most 128 tools are supported");
        }
        if let Some(choice) = self.tool_choice.as_deref() {
            if !matches!(choice, "auto" | "none" | "required") {
                bail!("tool_choice must be auto, none or required; named selection is unsupported");
            }
            if choice == "required" && self.tools.is_empty() {
                bail!("tool_choice required needs tools");
            }
        }
        for tool in &self.tools {
            if tool["type"] != "function"
                || tool
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                || !tool
                    .pointer("/function/parameters")
                    .is_some_and(Value::is_object)
            {
                bail!("tools require function name and JSON schema parameters");
            }
        }
        for message in &mut self.messages {
            normalize_message(message)?;
        }
        Ok(self)
    }
}

fn normalize_message(message: &mut Value) -> Result<()> {
    let Some(object) = message.as_object_mut() else {
        bail!("message must be an object");
    };
    let role = object.get("role").and_then(Value::as_str);
    if !matches!(role, Some("system" | "user" | "assistant" | "tool")) {
        bail!("unsupported message role");
    }
    if role == Some("tool") && object.get("tool_call_id").and_then(Value::as_str).is_none() {
        bail!("tool results require tool_call_id");
    }
    if let Some(calls) = object.get_mut("tool_calls") {
        let Some(calls) = calls.as_array_mut() else {
            bail!("tool_calls must be an array");
        };
        for call in calls {
            let Some(arguments) = call.pointer_mut("/function/arguments") else {
                bail!("tool call requires function arguments");
            };
            if let Some(encoded) = arguments.as_str() {
                *arguments = serde_json::from_str(encoded)?;
            }
            if !arguments.is_object() {
                bail!("tool call arguments must be a JSON object");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replay_normalizes_arguments_and_preserves_identity() {
        let input = json!({"model":"local", "messages":[
            {"role":"assistant","tool_calls":[{"id":"call_7","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Bogotá\"}"}}]},
            {"role":"tool","tool_call_id":"call_7","content":"sunny"}
        ]});
        let request: ChatRequest = serde_json::from_value(input).unwrap();
        let prepared = request.prepare("local").unwrap();
        assert_eq!(
            prepared.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"city":"Bogotá"})
        );
        assert_eq!(prepared.messages[0]["tool_calls"][0]["id"], "call_7");
        assert_eq!(prepared.messages[1]["tool_call_id"], "call_7");
    }

    #[test]
    fn rejects_nonobject_replayed_arguments() {
        let mut message =
            json!({"role":"assistant", "tool_calls":[{"function":{"arguments":"[]"}}]});
        assert!(normalize_message(&mut message).is_err());
    }

    #[test]
    fn rejects_missing_tool_result_identity() {
        let mut message = json!({"role":"tool", "content":"sunny"});
        assert!(normalize_message(&mut message).is_err());
    }
    #[test]
    fn validates_tool_policy_and_schemas() {
        for policy in [
            json!("bogus"),
            json!({"type":"function","function":{"name":"weather"}}),
            json!("required"),
        ] {
            let request: Result<ChatRequest, _> = serde_json::from_value(
                json!({"model":"local","messages":[{"role":"user","content":"hi"}],"tool_choice":policy}),
            );
            assert!(
                request
                    .map_err(anyhow::Error::from)
                    .and_then(|r| r.prepare("local"))
                    .is_err()
            );
        }
        let request: ChatRequest = serde_json::from_value(json!({"model":"local","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"x"}}]})).unwrap();
        assert!(request.prepare("local").is_err());
    }
}
