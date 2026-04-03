pub mod checkpoint;
pub mod config;
pub mod coordinator;
pub mod demo;
pub mod health;
pub mod network;
pub mod qos;
pub mod server;
pub mod training;
pub mod verification;

use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::coordinator::TrainingCoordinator;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct TrainingJobRequest {
        uint64 jobId;
        string baseModel;
        string datasetUrl;
        string method;
        uint32 totalEpochs;
        uint64 syncIntervalSteps;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct TrainingJobResult {
        uint64 jobId;
        bytes32 finalCheckpointHash;
        uint64 totalSteps;
        uint32 finalEpoch;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct CheckpointRequest {
        uint64 jobId;
        bytes32 checkpointHash;
        uint32 epoch;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct CheckpointResult {
        bool accepted;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeaveRequest {
        uint64 jobId;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeaveResult {
        bool acknowledged;
    }
}

// --- Job IDs ---

pub const TRAINING_JOB: u8 = 0;
pub const CHECKPOINT_JOB: u8 = 1;
pub const LEAVE_JOB: u8 = 2;

// --- Shared coordinator ---

static COORDINATOR: blueprint_sdk::std::sync::OnceLock<Arc<TrainingCoordinator>> =
    blueprint_sdk::std::sync::OnceLock::new();

fn get_coordinator() -> Result<&'static Arc<TrainingCoordinator>, RunnerError> {
    COORDINATOR
        .get()
        .ok_or_else(|| RunnerError::Other("TrainingCoordinator not initialized".into()))
}

pub fn register_coordinator(coord: Arc<TrainingCoordinator>) {
    let _ = COORDINATOR.set(coord);
}

// --- Router ---

pub fn router() -> Router {
    Router::new()
        .route(
            TRAINING_JOB,
            handle_training_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            CHECKPOINT_JOB,
            handle_checkpoint_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            LEAVE_JOB,
            handle_leave_job
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
}

// --- Job handlers ---

#[debug_job]
pub async fn handle_training_job(
    TangleArg(request): TangleArg<TrainingJobRequest>,
) -> Result<TangleResult<TrainingJobResult>, RunnerError> {
    let coord = get_coordinator()?;

    let result = coord
        .start_or_join_job(
            request.jobId,
            &request.baseModel,
            &request.datasetUrl,
            &request.method,
            request.totalEpochs,
            request.syncIntervalSteps,
        )
        .await
        .map_err(|e| RunnerError::Other(format!("training job failed: {e}").into()))?;

    Ok(TangleResult(TrainingJobResult {
        jobId: request.jobId,
        finalCheckpointHash: alloy::primitives::FixedBytes(result.checkpoint_hash),
        totalSteps: result.total_steps,
        finalEpoch: result.final_epoch,
    }))
}

#[debug_job]
pub async fn handle_checkpoint_job(
    TangleArg(request): TangleArg<CheckpointRequest>,
) -> Result<TangleResult<CheckpointResult>, RunnerError> {
    let coord = get_coordinator()?;

    coord
        .submit_checkpoint(request.jobId, request.checkpointHash.into(), request.epoch)
        .await
        .map_err(|e| RunnerError::Other(format!("checkpoint submit failed: {e}").into()))?;

    Ok(TangleResult(CheckpointResult { accepted: true }))
}

#[debug_job]
pub async fn handle_leave_job(
    TangleArg(request): TangleArg<LeaveRequest>,
) -> Result<TangleResult<LeaveResult>, RunnerError> {
    let coord = get_coordinator()?;

    coord
        .handle_leave(request.jobId)
        .await
        .map_err(|e| RunnerError::Other(format!("leave failed: {e}").into()))?;

    Ok(TangleResult(LeaveResult { acknowledged: true }))
}

// --- Background service: training coordinator + HTTP server ---

#[derive(Clone)]
pub struct TrainingServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for TrainingServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // Initialize training coordinator
            let coord = match TrainingCoordinator::new(config.clone()).await {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create TrainingCoordinator");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            register_coordinator(coord.clone());

            // Start the HTTP API server
            let state = server::AppState {
                config: config.clone(),
                coordinator: coord.clone(),
            };

            match server::start(state).await {
                Ok(_handle) => {
                    tracing::info!("Training HTTP server started");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // Keep alive until shutdown
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("received shutdown signal");
        });

        Ok(rx)
    }
}
