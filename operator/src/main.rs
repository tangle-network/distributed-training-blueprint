use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use distributed_training::config::OperatorConfig;
use distributed_training::health;
use distributed_training::TrainingServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for DistributedTrainingBSM.onRegister.
/// Format: abi.encode(uint32 gpuCount, uint32 totalVramMib, uint64 networkBandwidthMbps, string gpuModel, string endpoint)
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.gpu_count;
    let total_vram = config.gpu.total_vram_mib;
    let bandwidth = config.gpu.network_bandwidth_mbps;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (gpu_count, total_vram, bandwidth, gpu_model, endpoint).abi_encode()
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit payload and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            blueprint_sdk::std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        blueprint_sdk::std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            gpu_count = config.gpu.gpu_count,
            vram_mib = config.gpu.total_vram_mib,
            bandwidth_mbps = config.gpu.network_bandwidth_mbps,
            "Registration payload saved"
        );
        return Ok(());
    }

    // Detect GPUs
    match health::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed — running in CPU mode");
        }
    }

    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        match distributed_training::qos::start_heartbeat(config.clone()).await {
            Ok(_handle) => {
                let interval = config.qos.as_ref().unwrap().heartbeat_interval_secs;
                tracing::info!(interval_secs = interval, "QoS heartbeat started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS heartbeat failed to start");
            }
        }
    } else {
        tracing::info!("QoS heartbeat disabled");
    }

    let training_server = TrainingServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(distributed_training::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(training_server)
        .run()
        .await?;

    Ok(())
}
