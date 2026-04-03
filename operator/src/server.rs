//! HTTP API server for the distributed training operator.
//!
//! Endpoints:
//! - POST /v1/training/jobs          — create or join a training job
//! - GET  /v1/training/jobs/:id      — get job status
//! - POST /v1/training/jobs/:id/leave — gracefully leave a job
//! - GET  /v1/training/jobs/:id/checkpoint — download latest checkpoint
//! - GET  /health                    — health check

use blueprint_sdk::std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::coordinator::TrainingCoordinator;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub coordinator: Arc<TrainingCoordinator>,
}

/// Start the HTTP server. Returns a JoinHandle.
pub async fn start(state: AppState) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/training/jobs", post(create_job))
        .route("/v1/training/jobs/{id}", get(get_job))
        .route("/v1/training/jobs/{id}/leave", post(leave_job))
        .route("/v1/training/jobs/{id}/checkpoint", get(get_checkpoint))
        .route("/health", get(health_check))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind_addr = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    tracing::info!(addr = %bind_addr, "HTTP server listening");

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request/Response types ---

#[derive(Debug, Deserialize)]
struct CreateJobRequest {
    job_id: u64,
    base_model: String,
    dataset_url: String,
    method: String,
    total_epochs: u32,
    #[serde(default = "default_sync_interval")]
    sync_interval_steps: u64,
}

fn default_sync_interval() -> u64 {
    500
}

#[derive(Debug, Serialize)]
struct CreateJobResponse {
    job_id: u64,
    status: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// --- Handlers ---

async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    // SpendAuth validation placeholder — in production, validate X-Payment-Signature
    // header against ShieldedCredits contract.

    match state
        .coordinator
        .start_or_join_job(
            req.job_id,
            &req.base_model,
            &req.dataset_url,
            &req.method,
            req.total_epochs,
            req.sync_interval_steps,
        )
        .await
    {
        Ok(result) => (
            StatusCode::OK,
            Json(CreateJobResponse {
                job_id: req.job_id,
                status: "running".to_string(),
                message: format!(
                    "Training job started. Checkpoint: {}",
                    hex::encode(result.checkpoint_hash)
                ),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    match state.coordinator.get_job_status(id).await {
        Some(status) => (StatusCode::OK, Json(status)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("job {id} not found"),
            }),
        )
            .into_response(),
    }
}

async fn leave_job(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    match state.coordinator.handle_leave(id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({ "acknowledged": true })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn get_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> impl IntoResponse {
    match crate::checkpoint::latest_checkpoint(id).await {
        Ok(Some(path)) => match tokio::fs::read(&path).await {
            Ok(data) => (
                StatusCode::OK,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "application/octet-stream",
                )],
                data,
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to read checkpoint: {e}"),
                }),
            )
                .into_response(),
        },
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("no checkpoint found for job {id}"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn health_check() -> impl IntoResponse {
    let gpu_status = match crate::health::detect_gpus().await {
        Ok(gpus) => serde_json::json!({
            "available": true,
            "count": gpus.len(),
            "gpus": gpus,
        }),
        Err(_) => serde_json::json!({
            "available": false,
            "count": 0,
        }),
    };

    Json(serde_json::json!({
        "status": "ok",
        "service": "distributed-training-operator",
        "gpu": gpu_status,
    }))
}
