use anyhow::{Result, bail};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Local Eredu inference for Mesh")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Check the pinned native loading API without starting a server.
    #[arg(long)]
    check_model: bool,
    #[arg(long, default_value = "0")]
    port: u16,
    #[arg(long, default_value = "eredu-local")]
    model_id: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.model.exists() {
        bail!("model path does not exist: {}", args.model.display());
    }
    if args.check_model {
        check_model(args.model)
    } else {
        serve(args)
    }
}

#[cfg(not(feature = "mlx"))]
fn check_model(_model: PathBuf) -> Result<()> {
    bail!("native model validation requires a build with --features mlx")
}

#[cfg(feature = "mlx")]
fn check_model(path: PathBuf) -> Result<()> {
    use eredu::api::{LoadedModel, default_local_device, local_device_plan};
    use eredu_backend_mlx::MlxBackendFactory;
    use eredu_core::ExecutionPlan;

    let owner = std::thread::spawn(move || -> Result<()> {
        let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
        let planned =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &plan)?;
        let (mut model, _) = planned.into_parts();
        model.reset()?;
        model.synchronize()?;
        println!("native load succeeded: {}", path.display());
        Ok(())
    });
    owner
        .join()
        .map_err(|_| anyhow::anyhow!("native owner thread panicked"))?
}

#[cfg(not(feature = "mlx"))]
fn serve(_: Args) -> Result<()> {
    bail!("serving requires a build with --features mlx")
}

#[cfg(feature = "mlx")]
fn serve(args: Args) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(async move {
        let listener =
            tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, args.port)).await?;
        let address = listener.local_addr()?;
        let worker = eredu_plugin::worker::Worker::start(move || {
            eredu_plugin::native::Native::load(args.model)
        })
        .await?;
        let app = eredu_plugin::http::router(eredu_plugin::http::Endpoint {
            worker,
            model: args.model_id.into(),
        });
        let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
        // Readiness (and Mesh registration) only after the model has loaded.
        let url = format!("http://{address}/v1");
        eprintln!(
            "{}",
            serde_json::json!({"status":"ready","address":address.to_string(),"tools":false})
        );
        if std::env::var_os("MESH_LLM_PLUGIN_ENDPOINT").is_none() {
            return Ok(server.await?);
        }
        let plugin = eredu_plugin::mesh::plugin(url);
        tokio::select! {
            result = server => Ok(result?),
            result = mesh_llm_plugin::PluginRuntime::run(plugin) => result,
        }
    })
}

#[cfg(feature = "mlx")]
async fn shutdown_signal() {
    let mut terminate =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = async { match terminate.as_mut() { Some(t) => { t.recv().await; } None => std::future::pending().await } } => {},
    }
}
