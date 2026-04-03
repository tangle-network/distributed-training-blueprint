//! QoS heartbeat with training-specific metrics.
//!
//! Submits operator liveness proofs and training progress to the Tangle chain.
//! Metrics include epoch progress, loss, GPU utilization, and peer connectivity.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use alloy::{
    network::EthereumWallet,
    primitives::Address,
    providers::{Provider, ProviderBuilder},
    signers::local::PrivateKeySigner,
    sol,
};
use serde::{Deserialize, Serialize};

use crate::config::OperatorConfig;

sol! {
    #[sol(rpc)]
    interface IOperatorStatusRegistry {
        struct MetricPair {
            string key;
            uint64 value;
        }

        function submitHeartbeat(
            uint64 serviceId,
            uint64 blueprintId,
            uint64 blockNumber,
            MetricPair[] calldata metrics
        ) external;
    }
}

/// QoS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSConfig {
    /// Heartbeat interval in seconds. 0 = disabled.
    #[serde(default)]
    pub heartbeat_interval_secs: u64,

    /// On-chain address of the IOperatorStatusRegistry contract.
    #[serde(default)]
    pub status_registry_address: Option<String>,
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 0,
            status_registry_address: None,
        }
    }
}

/// Training-specific metrics for heartbeat submission.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub current_epoch: u64,
    pub steps_completed: u64,
    /// Loss scaled to integer (loss * 10000, e.g. 2.5 -> 25000).
    pub loss_scaled: u64,
    /// GPU utilization percentage (0-100).
    pub gpu_utilization: u64,
    /// Number of connected training peers.
    pub peers_connected: u64,
    /// Total GPU-hours contributed (scaled: hours * 1000).
    pub contribution_gpu_hours_scaled: u64,
}

/// Global training metrics (updated by coordinator).
static TRAINING_METRICS: blueprint_sdk::std::sync::OnceLock<Arc<blueprint_sdk::std::sync::RwLock<TrainingMetrics>>> =
    blueprint_sdk::std::sync::OnceLock::new();

/// Get or initialize the global training metrics.
pub fn training_metrics() -> Arc<blueprint_sdk::std::sync::RwLock<TrainingMetrics>> {
    TRAINING_METRICS
        .get_or_init(|| Arc::new(blueprint_sdk::std::sync::RwLock::new(TrainingMetrics::default())))
        .clone()
}

/// Update training metrics (called by coordinator after each step/epoch).
pub fn update_metrics(
    epoch: u64,
    steps: u64,
    loss: f32,
    gpu_util: u64,
    peers: u64,
    gpu_hours: f64,
) {
    if let Ok(mut m) = training_metrics().write() {
        m.current_epoch = epoch;
        m.steps_completed = steps;
        m.loss_scaled = (loss * 10000.0) as u64;
        m.gpu_utilization = gpu_util;
        m.peers_connected = peers;
        m.contribution_gpu_hours_scaled = (gpu_hours * 1000.0) as u64;
    }
}

/// Build metric pairs for on-chain submission.
fn on_chain_metrics() -> Vec<(String, u64)> {
    let m = training_metrics();
    let metrics = m.read().unwrap();

    vec![
        ("current_epoch".to_string(), metrics.current_epoch),
        ("steps_completed".to_string(), metrics.steps_completed),
        ("loss_scaled".to_string(), metrics.loss_scaled),
        ("gpu_utilization".to_string(), metrics.gpu_utilization),
        ("peers_connected".to_string(), metrics.peers_connected),
        (
            "contribution_gpu_hours".to_string(),
            metrics.contribution_gpu_hours_scaled,
        ),
    ]
}

/// Start the QoS heartbeat loop as a background task.
pub async fn start_heartbeat(
    config: Arc<OperatorConfig>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let qos = config
        .qos
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("qos config missing"))?;

    let interval_secs = qos.heartbeat_interval_secs;
    if interval_secs == 0 {
        anyhow::bail!("heartbeat disabled (interval = 0)");
    }

    let registry_addr: Address = qos
        .status_registry_address
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("status_registry_address not configured"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid status_registry_address: {e}"))?;

    let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
    let wallet = EthereumWallet::from(signer);
    let rpc_url: reqwest::Url = config.tangle.rpc_url.parse()?;
    let service_id = config
        .tangle
        .service_id
        .ok_or_else(|| anyhow::anyhow!("service_id required for QoS heartbeat"))?;
    let blueprint_id = config.tangle.blueprint_id;

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.tick().await; // skip first tick

        loop {
            interval.tick().await;

            match send_heartbeat(
                &wallet,
                &rpc_url,
                registry_addr,
                service_id,
                blueprint_id,
            )
            .await
            {
                Ok(()) => {
                    tracing::debug!(service_id, blueprint_id, "heartbeat submitted");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "heartbeat submission failed");
                }
            }
        }
    });

    Ok(handle)
}

async fn send_heartbeat(
    wallet: &EthereumWallet,
    rpc_url: &reqwest::Url,
    registry_addr: Address,
    service_id: u64,
    blueprint_id: u64,
) -> anyhow::Result<()> {
    let provider = ProviderBuilder::new()
        .wallet(wallet.clone())
        .connect_http(rpc_url.clone());

    let block_number = provider.get_block_number().await?;

    let chain_metrics = on_chain_metrics();
    let metric_pairs: Vec<IOperatorStatusRegistry::MetricPair> = chain_metrics
        .into_iter()
        .map(|(key, value)| IOperatorStatusRegistry::MetricPair { key, value })
        .collect();

    let registry = IOperatorStatusRegistry::new(registry_addr, &provider);
    let call = registry.submitHeartbeat(service_id, blueprint_id, block_number, metric_pairs);

    let tx_hash = call.send().await?.watch().await?;
    tracing::trace!(?tx_hash, "heartbeat tx confirmed");

    Ok(())
}
