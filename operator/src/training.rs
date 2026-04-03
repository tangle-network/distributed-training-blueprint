//! Training backend interface — abstracts the local training engine.
//!
//! Operators can use any training framework (axolotl, unsloth, torchtune) as long
//! as it exposes an HTTP API matching the TrainingBackend trait. The backend runs
//! as a separate process (typically a Python server) and communicates via HTTP.

use blueprint_sdk::std::time::Duration;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;

/// Result returned when a training job completes.
pub struct TrainingResult {
    pub checkpoint_hash: [u8; 32],
    pub total_steps: u64,
    pub final_epoch: u32,
}

/// Training backend interface.
///
/// Any local training engine must support these operations:
/// - init_model(base_model) — initialize or load a base model
/// - train_step(batch_index) -> gradients — execute one training step
/// - get_momentum() -> tensors — get current momentum buffers for DeMo sync
/// - apply_momentum_update(update) — apply a DeMo aggregated momentum update
/// - save_state() -> bytes — serialize model + optimizer state
/// - load_state(checkpoint) — restore from checkpoint

/// Local training backend that calls a Python training server over HTTP.
///
/// The Python server (axolotl, unsloth, torchtune) exposes endpoints:
/// - POST /init   { "model": "..." }
/// - POST /step   { "batch_index": N } -> { "gradients": [...] }
/// - GET  /momentum -> { "momentum": [...] }
/// - POST /apply_momentum { "update": [...] }
/// - POST /save_state -> bytes
/// - POST /load_state { "checkpoint": base64 }
pub struct LocalTrainingBackend {
    endpoint: String,
    client: reqwest::Client,
}

impl LocalTrainingBackend {
    pub fn new(endpoint: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
        }
    }

    pub async fn init_model(&self, base_model: &str) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!("{}/init", self.endpoint))
            .json(&serde_json::json!({ "model": base_model }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("init_model failed: {body}");
        }

        Ok(())
    }

    pub async fn train_step(&self, batch_index: u64) -> anyhow::Result<Vec<Array2<f32>>> {
        let resp = self
            .client
            .post(format!("{}/step", self.endpoint))
            .json(&serde_json::json!({ "batch_index": batch_index }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("train_step failed: {body}");
        }

        let body: StepResponse = resp.json().await?;
        // Convert flat gradient arrays to Array2
        let gradients = body
            .gradients
            .into_iter()
            .map(|g| {
                let rows = g.shape.0;
                let cols = g.shape.1;
                Array2::from_shape_vec((rows, cols), g.data)
                    .unwrap_or_else(|_| Array2::zeros((rows, cols)))
            })
            .collect();

        Ok(gradients)
    }

    pub async fn get_momentum(&self) -> anyhow::Result<Vec<Array2<f32>>> {
        let resp = self
            .client
            .get(format!("{}/momentum", self.endpoint))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("get_momentum failed: {body}");
        }

        let body: MomentumResponse = resp.json().await?;
        let momentum = body
            .momentum
            .into_iter()
            .map(|m| {
                Array2::from_shape_vec((m.shape.0, m.shape.1), m.data)
                    .unwrap_or_else(|_| Array2::zeros((m.shape.0, m.shape.1)))
            })
            .collect();

        Ok(momentum)
    }

    pub async fn apply_momentum_update(&self, update: &Array2<f32>) -> anyhow::Result<()> {
        let flat: Vec<f32> = update.iter().copied().collect();
        let shape = (update.shape()[0], update.shape()[1]);

        let resp = self
            .client
            .post(format!("{}/apply_momentum", self.endpoint))
            .json(&serde_json::json!({
                "update": { "data": flat, "shape": [shape.0, shape.1] }
            }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("apply_momentum_update failed: {body}");
        }

        Ok(())
    }

    pub async fn save_state(&self) -> anyhow::Result<Vec<u8>> {
        let resp = self
            .client
            .post(format!("{}/save_state", self.endpoint))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("save_state failed: {body}");
        }

        Ok(resp.bytes().await?.to_vec())
    }

    pub async fn load_state(&self, checkpoint: &[u8]) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!("{}/load_state", self.endpoint))
            .body(checkpoint.to_vec())
            .header("content-type", "application/octet-stream")
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("load_state failed: {body}");
        }

        Ok(())
    }
}

/// Remote API training backend — calls a cloud training service.
pub struct ApiTrainingBackend {
    endpoint: String,
    api_key: String,
    client: reqwest::Client,
}

impl ApiTrainingBackend {
    pub fn new(endpoint: &str, api_key: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client,
        }
    }

    pub async fn init_model(&self, base_model: &str) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!("{}/init", self.endpoint))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({ "model": base_model }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("API init_model failed: {body}");
        }

        Ok(())
    }

    pub async fn train_step(&self, batch_index: u64) -> anyhow::Result<Vec<Array2<f32>>> {
        let resp = self
            .client
            .post(format!("{}/step", self.endpoint))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({ "batch_index": batch_index }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("API train_step failed: {body}");
        }

        let body: StepResponse = resp.json().await?;
        let gradients = body
            .gradients
            .into_iter()
            .map(|g| {
                Array2::from_shape_vec((g.shape.0, g.shape.1), g.data)
                    .unwrap_or_else(|_| Array2::zeros((g.shape.0, g.shape.1)))
            })
            .collect();

        Ok(gradients)
    }

    pub async fn apply_momentum_update(&self, update: &Array2<f32>) -> anyhow::Result<()> {
        let flat: Vec<f32> = update.iter().copied().collect();
        let shape = (update.shape()[0], update.shape()[1]);

        let resp = self
            .client
            .post(format!("{}/apply_momentum", self.endpoint))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "update": { "data": flat, "shape": [shape.0, shape.1] }
            }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("API apply_momentum failed: {body}");
        }

        Ok(())
    }

    pub async fn save_state(&self) -> anyhow::Result<Vec<u8>> {
        let resp = self
            .client
            .post(format!("{}/save_state", self.endpoint))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        Ok(resp.bytes().await?.to_vec())
    }

    pub async fn load_state(&self, checkpoint: &[u8]) -> anyhow::Result<()> {
        self.client
            .post(format!("{}/load_state", self.endpoint))
            .bearer_auth(&self.api_key)
            .body(checkpoint.to_vec())
            .header("content-type", "application/octet-stream")
            .send()
            .await?;

        Ok(())
    }
}

/// Create a training backend from config.
pub fn create_backend(config: &TrainingConfig) -> anyhow::Result<LocalTrainingBackend> {
    Ok(LocalTrainingBackend::new(&config.endpoint))
}

// --- Wire types for training backend HTTP API ---

#[derive(Debug, Serialize, Deserialize)]
struct TensorPayload {
    data: Vec<f32>,
    shape: (usize, usize),
}

#[derive(Debug, Serialize, Deserialize)]
struct StepResponse {
    gradients: Vec<TensorPayload>,
    loss: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct MomentumResponse {
    momentum: Vec<TensorPayload>,
}
