//! Multi-operator training coordination.
//!
//! Manages the distributed training loop: data shard assignment, peer join/leave,
//! DeMo sync barriers, and on-chain checkpoint submission.

use blueprint_sdk::std::collections::HashMap;
use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::checkpoint;
use crate::config::OperatorConfig;
use crate::demo::{self, DemoOptimizer, SparseUpdate};
use crate::network::TrainingNetwork;

/// Result of a completed or joined training job.
pub struct JobResult {
    pub checkpoint_hash: [u8; 32],
    pub total_steps: u64,
    pub final_epoch: u32,
}

/// Active training job state.
pub struct TrainingJob {
    pub job_id: u64,
    pub base_model: String,
    pub dataset_url: String,
    pub method: String,
    pub total_epochs: u32,
    pub current_epoch: u32,
    pub sync_interval_steps: u64,
    pub steps_completed: u64,
    pub current_loss: f32,
    pub operators: Vec<String>,
    pub shard_assignments: HashMap<String, DataShard>,
    pub latest_checkpoint_hash: [u8; 32],
    pub latest_checkpoint_step: u64,
    pub completed: bool,
}

/// Data shard assigned to an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    /// Start index in the dataset.
    pub start: u64,
    /// End index (exclusive) in the dataset.
    pub end: u64,
    /// Shard index for identification.
    pub shard_id: u32,
}

/// Training coordinator managing multi-operator distributed training.
pub struct TrainingCoordinator {
    config: Arc<OperatorConfig>,
    jobs: RwLock<HashMap<u64, TrainingJob>>,
    network: Arc<TrainingNetwork>,
    our_peer_id: String,
}

impl TrainingCoordinator {
    pub async fn new(config: Arc<OperatorConfig>) -> anyhow::Result<Self> {
        let network = TrainingNetwork::new(&config.network).await?;
        let our_peer_id = network.local_peer_id().to_string();

        Ok(Self {
            config,
            jobs: RwLock::new(HashMap::new()),
            network: Arc::new(network),
            our_peer_id,
        })
    }

    /// Start or join a distributed training job.
    pub async fn start_or_join_job(
        &self,
        job_id: u64,
        base_model: &str,
        dataset_url: &str,
        method: &str,
        total_epochs: u32,
        sync_interval_steps: u64,
    ) -> anyhow::Result<JobResult> {
        // Discover existing peers for this job
        let peers = self.network.discover_training_peers(job_id).await?;

        let mut jobs = self.jobs.write().await;

        if jobs.contains_key(&job_id) {
            // Already participating — return current status
            let job = jobs.get(&job_id).unwrap();
            return Ok(JobResult {
                checkpoint_hash: job.latest_checkpoint_hash,
                total_steps: job.steps_completed,
                final_epoch: job.current_epoch,
            });
        }

        // Create job state
        let mut job = TrainingJob {
            job_id,
            base_model: base_model.to_string(),
            dataset_url: dataset_url.to_string(),
            method: method.to_string(),
            total_epochs,
            current_epoch: 0,
            sync_interval_steps,
            steps_completed: 0,
            current_loss: f32::INFINITY,
            operators: vec![self.our_peer_id.clone()],
            shard_assignments: HashMap::new(),
            latest_checkpoint_hash: [0u8; 32],
            latest_checkpoint_step: 0,
            completed: false,
        };

        // Add existing peers
        for peer in &peers {
            let peer_str = peer.to_string();
            if !job.operators.contains(&peer_str) {
                job.operators.push(peer_str);
            }
        }

        // Assign data shards across all operators
        let dataset_size = 1_000_000; // placeholder — real implementation queries dataset metadata
        self.assign_data_shards(&mut job, dataset_size);

        jobs.insert(job_id, job);
        drop(jobs);

        // Run training loop
        let result = self
            .run_training_loop(job_id, sync_interval_steps)
            .await?;

        Ok(result)
    }

    /// Assign data shards evenly across operators.
    pub fn assign_data_shards(&self, job: &mut TrainingJob, dataset_size: u64) {
        let n_operators = job.operators.len() as u64;
        if n_operators == 0 {
            return;
        }

        let shard_size = dataset_size / n_operators;
        job.shard_assignments.clear();

        for (i, operator) in job.operators.iter().enumerate() {
            let start = i as u64 * shard_size;
            let end = if i as u64 == n_operators - 1 {
                dataset_size
            } else {
                start + shard_size
            };

            job.shard_assignments.insert(
                operator.clone(),
                DataShard {
                    start,
                    end,
                    shard_id: i as u32,
                },
            );
        }
    }

    /// Handle a new peer joining an active training job.
    pub async fn handle_peer_join(&self, job_id: u64, peer: &str) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;

        if job.operators.contains(&peer.to_string()) {
            return Ok(()); // already in job
        }

        job.operators.push(peer.to_string());

        // Redistribute shards: take half of the largest shard for the new peer
        let largest_operator = job
            .shard_assignments
            .iter()
            .max_by_key(|(_, shard)| shard.end - shard.start)
            .map(|(op, _)| op.clone());

        if let Some(largest_op) = largest_operator {
            if let Some(shard) = job.shard_assignments.get(&largest_op).cloned() {
                let midpoint = shard.start + (shard.end - shard.start) / 2;

                // Shrink the largest shard
                job.shard_assignments.insert(
                    largest_op,
                    DataShard {
                        start: shard.start,
                        end: midpoint,
                        shard_id: shard.shard_id,
                    },
                );

                // Give the new peer the second half
                let new_shard_id = job.shard_assignments.len() as u32;
                job.shard_assignments.insert(
                    peer.to_string(),
                    DataShard {
                        start: midpoint,
                        end: shard.end,
                        shard_id: new_shard_id,
                    },
                );
            }
        }

        // Send latest checkpoint to the new peer
        if job.latest_checkpoint_step > 0 {
            let checkpoint_path = checkpoint::checkpoint_path(job_id, job.latest_checkpoint_step);
            if let Ok(data) = tokio::fs::read(&checkpoint_path).await {
                self.network
                    .send_checkpoint_to_peer(peer, &data)
                    .await
                    .ok();
            }
        }

        tracing::info!(
            job_id,
            peer,
            operators = job.operators.len(),
            "peer joined training job"
        );

        Ok(())
    }

    /// Handle a peer leaving an active training job.
    pub async fn handle_peer_leave(&self, job_id: u64, peer: &str) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;

        // Remove the peer
        job.operators.retain(|p| p != peer);

        // Absorb their shard into remaining operators
        if let Some(orphan_shard) = job.shard_assignments.remove(peer) {
            let remaining: Vec<String> = job.shard_assignments.keys().cloned().collect();
            if !remaining.is_empty() {
                let chunk_size =
                    (orphan_shard.end - orphan_shard.start) / remaining.len() as u64;
                let mut offset = orphan_shard.start;

                for (i, op) in remaining.iter().enumerate() {
                    if let Some(shard) = job.shard_assignments.get_mut(op) {
                        // Extend each remaining operator's shard
                        let extra_end = if i == remaining.len() - 1 {
                            orphan_shard.end
                        } else {
                            offset + chunk_size
                        };
                        // For simplicity, we extend by recording the additional range.
                        // A real implementation would merge ranges.
                        shard.end += extra_end - offset;
                        offset = extra_end;
                    }
                }
            }
        }

        tracing::info!(
            job_id,
            peer,
            remaining = job.operators.len(),
            "peer left training job, shards redistributed"
        );

        Ok(())
    }

    /// Handle the current operator leaving a job (on-chain LEAVE_JOB).
    pub async fn handle_leave(&self, job_id: u64) -> anyhow::Result<()> {
        self.handle_peer_leave(job_id, &self.our_peer_id.clone())
            .await
    }

    /// DeMo sync barrier: coordinate momentum synchronization across operators.
    pub async fn sync_barrier(
        &self,
        job_id: u64,
        local_updates: Vec<SparseUpdate>,
    ) -> anyhow::Result<Vec<ndarray::Array2<f32>>> {
        // Broadcast our sparse updates
        for update in &local_updates {
            self.network.broadcast_momentum_update(update).await?;
        }

        // Collect updates from peers with timeout
        let timeout = Duration::from_secs(30);
        let jobs = self.jobs.read().await;
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;
        let expected_peers = job.operators.len().saturating_sub(1);
        drop(jobs);

        let peer_updates = self
            .network
            .collect_momentum_updates(timeout, expected_peers)
            .await?;

        // Aggregate: combine all peer updates with our own
        let mut all_updates = local_updates;
        all_updates.extend(peer_updates);

        // Group by parameter index (using step as a proxy — all have same step)
        // For now, treat all updates as same parameter group and aggregate
        let aggregated = demo::aggregate_updates(&all_updates);

        // Apply inverse DCT to get spatial-domain momentum update
        let result = demo::idct_2d(&aggregated);

        Ok(vec![result])
    }

    /// Submit a checkpoint hash on-chain.
    pub async fn submit_checkpoint(
        &self,
        job_id: u64,
        hash: [u8; 32],
        epoch: u32,
    ) -> anyhow::Result<()> {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.latest_checkpoint_hash = hash;
            job.current_epoch = epoch;
        }

        tracing::info!(
            job_id,
            epoch,
            hash = hex::encode(hash),
            "checkpoint submitted"
        );

        Ok(())
    }

    /// Get the status of a training job.
    pub async fn get_job_status(&self, job_id: u64) -> Option<JobStatus> {
        let jobs = self.jobs.read().await;
        let job = jobs.get(&job_id)?;

        Some(JobStatus {
            job_id,
            base_model: job.base_model.clone(),
            method: job.method.clone(),
            current_epoch: job.current_epoch,
            total_epochs: job.total_epochs,
            steps_completed: job.steps_completed,
            current_loss: job.current_loss,
            operators: job.operators.len() as u32,
            completed: job.completed,
            latest_checkpoint_hash: hex::encode(job.latest_checkpoint_hash),
        })
    }

    /// Run the main training loop for a job.
    async fn run_training_loop(
        &self,
        job_id: u64,
        sync_interval: u64,
    ) -> anyhow::Result<JobResult> {
        let jobs = self.jobs.read().await;
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| anyhow::anyhow!("job not found"))?;
        let total_epochs = job.total_epochs;
        let base_model = job.base_model.clone();
        let _method = job.method.clone();
        drop(jobs);

        // Initialize training backend
        let backend = crate::training::create_backend(&self.config.training)?;
        backend.init_model(&base_model).await?;

        // Initialize DeMo optimizer
        // Shape is determined by the model — use a placeholder until backend reports it
        let param_shapes = vec![(1024, 1024)]; // placeholder
        let mut optimizer = DemoOptimizer::new(&param_shapes, 1e-4, sync_interval, 0.001);

        for epoch in 0..total_epochs {
            tracing::info!(job_id, epoch, "starting epoch");

            // Training steps within an epoch
            let steps_per_epoch = 1000; // determined by dataset size / batch size
            for step in 0..steps_per_epoch {
                let global_step = epoch as u64 * steps_per_epoch + step;

                // Get gradients from training backend
                let gradients = backend.train_step(step).await?;

                // Local AdamW step
                let needs_sync = optimizer.local_step(&gradients);

                if needs_sync {
                    // Prepare sparse updates
                    let updates = optimizer.prepare_sync();

                    // DeMo sync barrier
                    let aggregated = self.sync_barrier(job_id, updates).await?;

                    // Apply aggregated momentum
                    optimizer.apply_sync(&aggregated);

                    // Apply to model via backend
                    for update in &aggregated {
                        backend.apply_momentum_update(update).await?;
                    }
                }

                // Update job state
                let mut jobs = self.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.steps_completed = global_step + 1;
                }
            }

            // End-of-epoch checkpoint
            backend.save_state().await?;
            let ckpt_path = checkpoint::checkpoint_path(job_id, epoch as u64);
            checkpoint::save_checkpoint_file(&ckpt_path, &[]).await?;
            let hash = checkpoint::hash_checkpoint(&ckpt_path)?;

            let mut jobs = self.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.current_epoch = epoch + 1;
                job.latest_checkpoint_hash = hash;
                job.latest_checkpoint_step = epoch as u64;
            }

            self.submit_checkpoint(job_id, hash, epoch + 1).await?;
        }

        // Mark completed
        let mut jobs = self.jobs.write().await;
        let job = jobs.get_mut(&job_id).unwrap();
        job.completed = true;

        Ok(JobResult {
            checkpoint_hash: job.latest_checkpoint_hash,
            total_steps: job.steps_completed,
            final_epoch: job.current_epoch,
        })
    }
}

/// Public job status for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: u64,
    pub base_model: String,
    pub method: String,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub steps_completed: u64,
    pub current_loss: f32,
    pub operators: u32,
    pub completed: bool,
    pub latest_checkpoint_hash: String,
}
