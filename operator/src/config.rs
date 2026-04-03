//! Operator configuration — training, networking, billing, GPU, and Tangle settings.

use blueprint_sdk::std::fmt;
use blueprint_sdk::std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration.
    pub tangle: TangleConfig,

    /// Training backend configuration.
    pub training: TrainingConfig,

    /// HTTP server configuration.
    pub server: ServerConfig,

    /// Networking (libp2p) configuration.
    pub network: NetworkConfig,

    /// Billing configuration.
    pub billing: BillingConfig,

    /// GPU configuration.
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("training", &self.training)
            .field("server", &self.server)
            .field("network", &self.network)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain.
    pub rpc_url: String,

    /// Chain ID.
    pub chain_id: u64,

    /// Operator's private key (hex). Use KMS in production.
    pub operator_key: String,

    /// Tangle core contract address.
    pub tangle_core: String,

    /// ShieldedCredits contract address.
    pub shielded_credits: String,

    /// Blueprint ID.
    pub blueprint_id: u64,

    /// Service ID (set after service activation).
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

/// Training backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Training backend HTTP endpoint (e.g. "http://localhost:5000").
    /// Can be overridden by TRAINING_ENDPOINT env var.
    #[serde(default = "default_training_endpoint")]
    pub endpoint: String,

    /// DeMo sync interval in local training steps.
    #[serde(default = "default_sync_interval")]
    pub sync_interval_steps: u64,

    /// Maximum operators per training job.
    #[serde(default = "default_max_operators")]
    pub max_operators: u32,

    /// Supported training methods.
    #[serde(default = "default_supported_methods")]
    pub supported_methods: Vec<String>,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind host.
    #[serde(default = "default_host")]
    pub host: String,

    /// Bind port.
    #[serde(default = "default_port")]
    pub port: u16,
}

/// Networking (libp2p) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address for libp2p (multiaddr format).
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Bootstrap peer addresses for initial discovery.
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
}

/// Billing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Price per GPU-hour in tsUSD base units (6 decimals).
    pub price_per_gpu_hour: u64,

    /// Whether billing is required.
    #[serde(default = "default_billing_required")]
    pub required: bool,
}

/// GPU hardware configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Number of GPUs available.
    pub gpu_count: u32,

    /// Total VRAM across all GPUs in MiB.
    pub total_vram_mib: u32,

    /// GPU model name for on-chain registration.
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// Network bandwidth in Mbps (for DeMo efficiency estimation).
    #[serde(default = "default_bandwidth")]
    pub network_bandwidth_mbps: u64,
}

// --- Defaults ---

fn default_training_endpoint() -> String {
    blueprint_sdk::std::env::var("TRAINING_ENDPOINT").unwrap_or_else(|_| "http://localhost:5000".to_string())
}

fn default_sync_interval() -> u64 {
    500
}

fn default_max_operators() -> u32 {
    256
}

fn default_supported_methods() -> Vec<String> {
    vec![
        "sft".to_string(),
        "dpo".to_string(),
        "grpo".to_string(),
        "pretrain".to_string(),
    ]
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_listen_addr() -> String {
    "/ip4/0.0.0.0/tcp/9000".to_string()
}

fn default_billing_required() -> bool {
    true
}

fn default_bandwidth() -> u64 {
    1000 // 1 Gbps default
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: TRAIN_OP_ (e.g. TRAIN_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("TRAIN_OP")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "training": {
                "endpoint": "http://localhost:5000",
                "sync_interval_steps": 500,
                "max_operators": 64
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "network": {
                "listen_addr": "/ip4/0.0.0.0/tcp/9000",
                "bootstrap_peers": []
            },
            "billing": {
                "price_per_gpu_hour": 1000000
            },
            "gpu": {
                "gpu_count": 4,
                "total_vram_mib": 327680,
                "gpu_model": "NVIDIA H100",
                "network_bandwidth_mbps": 10000
            }
        }"#
    }

    #[test]
    fn test_deserialize_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.gpu.gpu_count, 4);
        assert_eq!(cfg.gpu.total_vram_mib, 327680);
        assert_eq!(cfg.training.sync_interval_steps, 500);
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn test_defaults_applied() {
        let json = r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xdead",
                "tangle_core": "0x01",
                "shielded_credits": "0x02",
                "blueprint_id": 1
            },
            "training": {},
            "server": {},
            "network": {},
            "billing": { "price_per_gpu_hour": 1000000 },
            "gpu": { "gpu_count": 1, "total_vram_mib": 24000 }
        }"#;
        let cfg: OperatorConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.training.sync_interval_steps, 500);
        assert_eq!(cfg.training.max_operators, 256);
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.network.listen_addr, "/ip4/0.0.0.0/tcp/9000");
        assert_eq!(cfg.gpu.network_bandwidth_mbps, 1000);
    }
}
