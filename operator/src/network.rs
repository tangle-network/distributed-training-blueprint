//! Networking layer for distributed training using Blueprint SDK's libp2p stack.
//!
//! Provides gossip-based momentum update broadcasting and direct peer-to-peer
//! checkpoint transfers. Built on blueprint-networking's gossipsub + request-response.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};

use crate::config::NetworkConfig;
use crate::demo::SparseUpdate;

/// Gossip topic for DeMo momentum synchronization.
const MOMENTUM_TOPIC: &str = "/tangle/training/momentum/1.0.0";

/// Gossip topic for training coordination messages.
const COORDINATION_TOPIC: &str = "/tangle/training/coordination/1.0.0";

/// Opaque peer identifier (libp2p PeerId string).
pub type PeerId = String;

/// Messages exchanged over the coordination gossip channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    /// Peer announces it is joining a training job.
    JoinJob {
        job_id: u64,
        peer_id: String,
        gpu_count: u32,
        vram_mib: u32,
    },
    /// Peer announces it is leaving a training job.
    LeaveJob { job_id: u64, peer_id: String },
    /// Sync barrier acknowledgement.
    SyncReady {
        job_id: u64,
        peer_id: String,
        step: u64,
    },
    /// Checkpoint available notification.
    CheckpointReady {
        job_id: u64,
        peer_id: String,
        step: u64,
        hash: [u8; 32],
    },
}

/// Training network node wrapping Blueprint SDK networking.
pub struct TrainingNetwork {
    /// Our local peer ID.
    peer_id: String,
    /// Collected momentum updates from peers.
    momentum_inbox: Arc<RwLock<Vec<SparseUpdate>>>,
    /// Channel for outgoing momentum broadcasts.
    momentum_tx: mpsc::Sender<Vec<u8>>,
    /// Channel for outgoing coordination messages.
    coordination_tx: mpsc::Sender<Vec<u8>>,
    /// Known peers per job.
    job_peers: Arc<RwLock<blueprint_sdk::std::collections::HashMap<u64, Vec<PeerId>>>>,
    /// Configuration.
    config: NetworkConfig,
}

impl TrainingNetwork {
    /// Create a new training network node.
    pub async fn new(config: &NetworkConfig) -> anyhow::Result<Self> {
        let (momentum_tx, mut momentum_rx) = mpsc::channel::<Vec<u8>>(256);
        let (coordination_tx, mut coordination_rx) = mpsc::channel::<Vec<u8>>(64);
        let momentum_inbox = Arc::new(RwLock::new(Vec::new()));
        let job_peers = Arc::new(RwLock::new(blueprint_sdk::std::collections::HashMap::new()));

        // Generate a peer ID from config or random
        let peer_id = uuid::Uuid::new_v4().to_string();

        // In production, this initializes blueprint-networking's NetworkService:
        //
        // let net_config = blueprint_networking::NetworkConfig::new(config.listen_addr.clone())
        //     .with_gossip()
        //     .with_request_response();
        // let (service, handle) = blueprint_networking::NetworkService::new(net_config)?;
        // tokio::spawn(service.run());
        //
        // For now, we use channels as a local abstraction that the real
        // NetworkService would drive.

        let inbox = momentum_inbox.clone();
        tokio::spawn(async move {
            // Process outgoing momentum broadcasts
            while let Some(_data) = momentum_rx.recv().await {
                // In production: handle.gossip_publish(MOMENTUM_TOPIC, data).await
                tracing::trace!("momentum update broadcast (stub)");
            }
        });

        tokio::spawn(async move {
            // Process outgoing coordination messages
            while let Some(_data) = coordination_rx.recv().await {
                // In production: handle.gossip_publish(COORDINATION_TOPIC, data).await
                tracing::trace!("coordination message broadcast (stub)");
            }
        });

        Ok(Self {
            peer_id,
            momentum_inbox,
            momentum_tx,
            coordination_tx,
            job_peers,
            config: config.clone(),
        })
    }

    /// Get our local peer ID.
    pub fn local_peer_id(&self) -> &str {
        &self.peer_id
    }

    /// Broadcast a sparse momentum update to all peers via gossip.
    pub async fn broadcast_momentum_update(&self, update: &SparseUpdate) -> anyhow::Result<()> {
        let mut update = update.clone();
        update.peer_id = self.peer_id.clone();

        let data = serde_json::to_vec(&update)?;
        tracing::debug!(
            peer_id = %self.peer_id,
            size_bytes = data.len(),
            indices = update.indices.len(),
            "broadcasting momentum update"
        );

        self.momentum_tx.send(data).await.map_err(|e| {
            anyhow::anyhow!("failed to send momentum update: {e}")
        })?;

        Ok(())
    }

    /// Collect momentum updates from peers with a timeout.
    /// Blocks until `expected_count` updates are received or timeout expires.
    pub async fn collect_momentum_updates(
        &self,
        timeout: Duration,
        expected_count: usize,
    ) -> anyhow::Result<Vec<SparseUpdate>> {
        if expected_count == 0 {
            return Ok(Vec::new());
        }

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let inbox = self.momentum_inbox.read().await;
            if inbox.len() >= expected_count {
                let updates = inbox.clone();
                drop(inbox);

                // Clear inbox after collection
                let mut inbox = self.momentum_inbox.write().await;
                inbox.clear();

                return Ok(updates);
            }
            drop(inbox);

            if tokio::time::Instant::now() >= deadline {
                // Return whatever we have
                let mut inbox = self.momentum_inbox.write().await;
                let updates = inbox.drain(..).collect();
                return Ok(updates);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Send checkpoint data directly to a specific peer.
    pub async fn send_checkpoint_to_peer(
        &self,
        peer: &str,
        checkpoint_data: &[u8],
    ) -> anyhow::Result<()> {
        tracing::info!(
            peer,
            size_bytes = checkpoint_data.len(),
            "sending checkpoint to peer"
        );

        // In production: handle.send_request(peer_id, checkpoint_data).await
        // The blueprint-networking request-response protocol handles large payloads
        // with automatic chunking.

        Ok(())
    }

    /// Discover peers participating in a specific training job.
    pub async fn discover_training_peers(&self, job_id: u64) -> anyhow::Result<Vec<PeerId>> {
        let peers = self.job_peers.read().await;
        Ok(peers.get(&job_id).cloned().unwrap_or_default())
    }

    /// Announce participation in a training job.
    pub async fn announce_join(
        &self,
        job_id: u64,
        gpu_count: u32,
        vram_mib: u32,
    ) -> anyhow::Result<()> {
        let msg = CoordinationMessage::JoinJob {
            job_id,
            peer_id: self.peer_id.clone(),
            gpu_count,
            vram_mib,
        };

        let data = serde_json::to_vec(&msg)?;
        self.coordination_tx.send(data).await.map_err(|e| {
            anyhow::anyhow!("failed to send join announcement: {e}")
        })?;

        // Add ourselves to the job peer list
        let mut peers = self.job_peers.write().await;
        peers
            .entry(job_id)
            .or_insert_with(Vec::new)
            .push(self.peer_id.clone());

        Ok(())
    }

    /// Announce departure from a training job.
    pub async fn announce_leave(&self, job_id: u64) -> anyhow::Result<()> {
        let msg = CoordinationMessage::LeaveJob {
            job_id,
            peer_id: self.peer_id.clone(),
        };

        let data = serde_json::to_vec(&msg)?;
        self.coordination_tx.send(data).await.map_err(|e| {
            anyhow::anyhow!("failed to send leave announcement: {e}")
        })?;

        // Remove ourselves from the job peer list
        let mut peers = self.job_peers.write().await;
        if let Some(job_peers) = peers.get_mut(&job_id) {
            job_peers.retain(|p| p != &self.peer_id);
        }

        Ok(())
    }

    /// Deliver a momentum update received from the network (called by network event loop).
    pub async fn on_momentum_received(&self, data: &[u8]) -> anyhow::Result<()> {
        let update: SparseUpdate = serde_json::from_slice(data)?;
        let mut inbox = self.momentum_inbox.write().await;
        inbox.push(update);
        Ok(())
    }

    /// Deliver a coordination message received from the network.
    pub async fn on_coordination_received(&self, data: &[u8]) -> anyhow::Result<()> {
        let msg: CoordinationMessage = serde_json::from_slice(data)?;

        match msg {
            CoordinationMessage::JoinJob {
                job_id, peer_id, ..
            } => {
                let mut peers = self.job_peers.write().await;
                let job_peers = peers.entry(job_id).or_insert_with(Vec::new);
                if !job_peers.contains(&peer_id) {
                    job_peers.push(peer_id);
                }
            }
            CoordinationMessage::LeaveJob { job_id, peer_id } => {
                let mut peers = self.job_peers.write().await;
                if let Some(job_peers) = peers.get_mut(&job_id) {
                    job_peers.retain(|p| p != &peer_id);
                }
            }
            CoordinationMessage::SyncReady { .. } => {
                // Handled by sync barrier logic
            }
            CoordinationMessage::CheckpointReady { .. } => {
                // Handled by checkpoint logic
            }
        }

        Ok(())
    }
}
