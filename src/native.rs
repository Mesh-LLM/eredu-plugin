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
    runtime::chat::ChatTemplateRequest,
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
    ) -> Result<()> {
        if !request.tools.is_empty()
            || request
                .messages
                .iter()
                .any(|m| m.get("tool_calls").is_some() || m["role"] == "tool")
        {
            bail!("native tool calls are not yet validated by this adapter");
        }
        self.0.reset()?;
        let prepared = self.0.prepare_chat(ChatTemplateRequest {
            messages: request.messages,
            add_generation_prompt: true,
            ..Default::default()
        })?;
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
                    if let SemanticEvent::TextDelta(text) = event
                        && publish(output, Event::Text(text)).is_err()
                    {
                        delivery_failed = true;
                        cancellation.cancel();
                    }
                },
            })?;
        if delivery_failed || work.is_cancelled() {
            bail!("generation cancelled by consumer");
        }
        let reason = match result.finish_reason {
            FinishReason::MaxTokens => "length",
            FinishReason::Cancelled => bail!("generation cancelled"),
            _ => "stop",
        };
        publish(output, Event::Finished(reason))
    }

    fn settle(&mut self) -> Result<()> {
        self.0.synchronize()?;
        self.0.reset()?;
        Ok(())
    }
}
