//! Mesh plugin registration. Inference stays on the loopback HTTP endpoint; IPC is control only.
use mesh_llm_plugin::{
    PROTOCOL_VERSION, PluginError, PluginMetadata, PluginStartupPolicy, SimplePlugin,
    plugin_server_info,
};

pub fn plugin(url: String) -> SimplePlugin {
    let name = std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| "eredu".into());
    let version = env!("CARGO_PKG_VERSION");
    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(name.clone(), version, plugin_server_info(
            name, version, "Eredu", "Local Eredu (MLX) inference", None::<String>)),
        startup_policy: PluginStartupPolicy::Any,
        inference: [mesh_llm_plugin::inference::provider("eredu", url)],
    }
    .on_initialize(|request, _| {
        Box::pin(async move {
            if request.host_protocol_version != PROTOCOL_VERSION {
                return Err(PluginError::internal(format!(
                    "incompatible Mesh plugin protocol: host={}, plugin={PROTOCOL_VERSION}",
                    request.host_protocol_version
                )));
            }
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use mesh_llm_plugin::Plugin;

    #[test]
    fn manifest_registers_one_managed_openai_provider() {
        let manifest = super::plugin("http://127.0.0.1:1/v1".into())
            .manifest()
            .unwrap();
        assert_eq!(manifest.endpoints.len(), 1);
        assert!(manifest.endpoints[0].managed_by_plugin);
        assert_eq!(
            manifest.endpoints[0].protocol.as_deref(),
            Some("openai_compatible")
        );
    }
}
