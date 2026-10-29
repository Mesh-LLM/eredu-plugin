use crate::{
    admission::ActiveWork,
    request::ChatRequest,
    worker::{Backend, Event, publish},
};
use anyhow::{Result, bail};
use eredu::{
    api::{
        LoadedModel, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
        PreparedChatInput, default_local_device, local_device_plan,
    },
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, ToolChoice},
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::ExecutionPlanBackendFactory;
use eredu_core::{
    ExecutionPlan, FinishReason, GenerationCancellationToken, GenerationConfigOverrides,
    SemanticEvent,
};
use std::path::PathBuf;
use tokio::sync::mpsc;

pub struct Native(LoadedModel<<MlxBackendFactory as ExecutionPlanBackendFactory>::Backend>);

impl Native {
    pub fn load(path: PathBuf) -> Result<Self> {
        let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
        let planned = LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &plan)?;
        let (model, _) = planned.into_parts();
        Ok(Self(model))
    }
}

impl Backend for Native {
    fn generate(
        &mut self,
        request: ChatRequest,
        work: &ActiveWork,
        output: &mpsc::Sender<Event>,
    ) -> Result<&'static str> {
        let needs_tools = !request.tools.is_empty()
            || request
                .messages
                .iter()
                .any(|m| m.get("tool_calls").is_some() || m["role"] == "tool");
        self.0.reset()?;
        let tool_choice = match request.tool_choice.as_deref() {
            Some("none") => ToolChoice::None,
            Some("required") => ToolChoice::Required,
            _ => ToolChoice::Auto,
        };
        let prepared = self.0.prepare_chat(ChatTemplateRequest {
            messages: request.messages,
            tools: request.tools,
            tool_choice,
            parallel_tool_calls: if request.parallel_tool_calls {
                ParallelToolCallPolicy::Enabled { max_calls: None }
            } else {
                ParallelToolCallPolicy::Disabled
            },
            enable_thinking: request.enable_thinking,
            add_generation_prompt: true,
            ..Default::default()
        })?;
        if needs_tools
            && let NativeToolSupport::Unsupported { reason } = prepared.native_tool_support()
        {
            bail!("native tools unavailable for this template: {reason}");
        }
        let mut started_calls = 0usize;
        let mut ended_calls = 0usize;
        let cancellation = GenerationCancellationToken::new();
        let mut delivery_failed = false;
        let result = self
            .0
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&prepared),
                settings: PreparedChatGenerationSettings {
                    overrides: GenerationConfigOverrides {
                        max_new_tokens: Some(request.max_tokens.unwrap_or(256)),
                        temperature: request.temperature,
                        do_sample: request.temperature.map(|t| t > 0.0),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                caller_stop_sequences: &[],
                cancellation: cancellation.clone(),
                on_event: |event| {
                    if work.is_cancelled() {
                        cancellation.cancel();
                        return;
                    }
                    let event = match event {
                        SemanticEvent::TextDelta(text) => Some(Event::Text(text)),
                        SemanticEvent::ToolCallStart { index, id, name } => {
                            started_calls += 1;
                            Some(Event::ToolStart { index, id, name })
                        }
                        SemanticEvent::ToolArgumentsDelta {
                            index,
                            json_fragment,
                        } => Some(Event::ToolArguments {
                            index,
                            fragment: json_fragment,
                        }),
                        SemanticEvent::ToolCallEnd => {
                            ended_calls += 1;
                            None
                        }
                        _ => None,
                    };
                    if let Some(event) = event
                        && publish(output, event).is_err()
                    {
                        delivery_failed = true;
                        cancellation.cancel();
                    }
                },
            })?;
        if delivery_failed || work.is_cancelled() {
            bail!("generation cancelled by consumer");
        }
        if started_calls != ended_calls {
            bail!("generation ended with incomplete tool calls");
        }
        let reason = match result.finish_reason {
            FinishReason::MaxTokens => "length",
            FinishReason::Cancelled => bail!("generation cancelled"),
            _ if started_calls > 0 => "tool_calls",
            _ => "stop",
        };
        Ok(reason)
    }

    fn settle(&mut self) -> Result<()> {
        self.0.synchronize()?;
        self.0.reset()?;
        Ok(())
    }
}
