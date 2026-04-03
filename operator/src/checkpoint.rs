//! Checkpoint management — save, load, hash model state for on-chain submission.
//!
//! Checkpoints are stored locally on disk. Each checkpoint contains the full model
//! state, optimizer state, and training metadata. The SHA-256 hash is submitted
//! on-chain for verification and resumability.

use blueprint_sdk::std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub job_id: u64,
    pub step: u64,
    pub epoch: u32,
    pub loss: f32,
    pub operators: Vec<String>,
    pub hash: [u8; 32],
}

/// Full checkpoint data (serializable).
#[derive(Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    /// Serialized model state (opaque bytes from training backend).
    pub model_state: Vec<u8>,
    /// Serialized optimizer state (DeMo momentum buffers).
    pub optimizer_state: Vec<u8>,
}

/// Base directory for checkpoints.
fn checkpoint_dir() -> PathBuf {
    PathBuf::from("data/checkpoints")
}

/// Path for a specific checkpoint file.
pub fn checkpoint_path(job_id: u64, step: u64) -> PathBuf {
    checkpoint_dir().join(format!("job_{job_id}_step_{step}.ckpt"))
}

/// Save a checkpoint to disk.
pub async fn save_checkpoint(
    model_state: &[u8],
    optimizer_state: &[u8],
    job_id: u64,
    step: u64,
    epoch: u32,
    loss: f32,
    operators: &[String],
) -> anyhow::Result<PathBuf> {
    let path = checkpoint_path(job_id, step);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut hasher = Sha256::new();
    hasher.update(model_state);
    hasher.update(optimizer_state);
    let hash: [u8; 32] = hasher.finalize().into();

    let checkpoint = Checkpoint {
        metadata: CheckpointMetadata {
            job_id,
            step,
            epoch,
            loss,
            operators: operators.to_vec(),
            hash,
        },
        model_state: model_state.to_vec(),
        optimizer_state: optimizer_state.to_vec(),
    };

    let data = serde_json::to_vec(&checkpoint)?;
    tokio::fs::write(&path, &data).await?;

    tracing::info!(
        job_id,
        step,
        epoch,
        hash = hex::encode(hash),
        size_bytes = data.len(),
        "checkpoint saved"
    );

    Ok(path)
}

/// Save raw checkpoint data to a path (used by coordinator).
pub async fn save_checkpoint_file(path: &Path, data: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, data).await?;
    Ok(())
}

/// Load a checkpoint from disk.
pub async fn load_checkpoint(path: &Path) -> anyhow::Result<Checkpoint> {
    let data = tokio::fs::read(path).await?;
    let checkpoint: Checkpoint = serde_json::from_slice(&data)?;

    tracing::info!(
        job_id = checkpoint.metadata.job_id,
        step = checkpoint.metadata.step,
        "checkpoint loaded"
    );

    Ok(checkpoint)
}

/// Compute SHA-256 hash of a checkpoint file.
pub fn hash_checkpoint(path: &Path) -> anyhow::Result<[u8; 32]> {
    let data = blueprint_sdk::std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hasher.finalize().into())
}

/// List available checkpoints for a job, sorted by step descending.
pub async fn list_checkpoints(job_id: u64) -> anyhow::Result<Vec<PathBuf>> {
    let dir = checkpoint_dir();
    let prefix = format!("job_{job_id}_step_");

    let mut entries = Vec::new();
    if let Ok(mut read_dir) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = read_dir.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(".ckpt") {
                entries.push(entry.path());
            }
        }
    }

    // Sort by step number descending
    entries.sort_by(|a, b| {
        let step_a = extract_step(a);
        let step_b = extract_step(b);
        step_b.cmp(&step_a)
    });

    Ok(entries)
}

/// Extract step number from a checkpoint filename.
fn extract_step(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit('_').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Get the latest checkpoint for a job.
pub async fn latest_checkpoint(job_id: u64) -> anyhow::Result<Option<PathBuf>> {
    let checkpoints = list_checkpoints(job_id).await?;
    Ok(checkpoints.into_iter().next())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_step() {
        let path = PathBuf::from("data/checkpoints/job_1_step_500.ckpt");
        assert_eq!(extract_step(&path), 500);
    }

    #[test]
    fn test_hash_deterministic() {
        let data = b"test checkpoint data";
        let mut h1 = Sha256::new();
        h1.update(data);
        let hash1: [u8; 32] = h1.finalize().into();

        let mut h2 = Sha256::new();
        h2.update(data);
        let hash2: [u8; 32] = h2.finalize().into();

        assert_eq!(hash1, hash2);
    }
}
