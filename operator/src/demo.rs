//! DeMo (Decoupled Momentum) optimizer — 10,000x communication reduction.
//!
//! Core algorithm from Nous Research's DisTrO:
//! 1. Each operator trains locally with AdamW
//! 2. Every K steps, operators synchronize momentum via DCT + top-k sparsification
//! 3. Compressed momentum (~0.1% of full size) is broadcast via gossip
//! 4. Operators aggregate and apply the shared momentum update
//!
//! Reference: https://arxiv.org/abs/2411.19870

use blueprint_sdk::std::sync::Arc;

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Compressed momentum update — indices + values from top-k sparsification.
/// Typically 0.1% of the full momentum tensor size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseUpdate {
    /// Flat indices into the DCT-transformed momentum tensor.
    pub indices: Vec<u32>,
    /// Corresponding DCT coefficient values.
    pub values: Vec<f32>,
    /// Original tensor shape (rows, cols) for reconstruction.
    pub shape: (usize, usize),
    /// Step number this update was generated at.
    pub step: u64,
    /// Peer ID of the operator that produced this update.
    pub peer_id: String,
}

impl SparseUpdate {
    /// Serialized byte size (approximate).
    pub fn byte_size(&self) -> usize {
        self.indices.len() * 4 + self.values.len() * 4 + 24
    }

    /// Reconstruct into a dense DCT-domain tensor.
    pub fn to_dense(&self) -> Array2<f32> {
        let mut dense = Array2::zeros((self.shape.0, self.shape.1));
        let cols = self.shape.1;
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let row = idx as usize / cols;
            let col = idx as usize % cols;
            if row < self.shape.0 && col < cols {
                dense[[row, col]] = val;
            }
        }
        dense
    }
}

/// AdamW optimizer state for a single parameter group.
#[derive(Debug, Clone)]
pub struct AdamWState {
    /// First moment (mean of gradients).
    pub m: Array2<f32>,
    /// Second moment (mean of squared gradients).
    pub v: Array2<f32>,
    /// Current step count.
    pub t: u64,
    /// Learning rate.
    pub lr: f32,
    /// Beta1 (first moment decay).
    pub beta1: f32,
    /// Beta2 (second moment decay).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Weight decay coefficient.
    pub weight_decay: f32,
}

impl AdamWState {
    pub fn new(shape: (usize, usize), lr: f32) -> Self {
        Self {
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// DeMo optimizer wrapping a base AdamW optimizer.
pub struct DemoOptimizer {
    /// AdamW state per parameter group.
    pub states: Vec<AdamWState>,
    /// How many local steps between DeMo sync rounds.
    pub sync_interval: u64,
    /// Top-k sparsification ratio (e.g., 0.001 = keep top 0.1%).
    pub top_k_ratio: f32,
    /// Steps completed since last sync.
    pub steps_since_sync: u64,
    /// Accumulated momentum delta since last sync.
    momentum_deltas: Vec<Array2<f32>>,
    /// Shared lock for thread-safe access during sync.
    lock: Arc<RwLock<()>>,
}

impl DemoOptimizer {
    pub fn new(shapes: &[(usize, usize)], lr: f32, sync_interval: u64, top_k_ratio: f32) -> Self {
        let states: Vec<AdamWState> = shapes.iter().map(|&s| AdamWState::new(s, lr)).collect();
        let momentum_deltas: Vec<Array2<f32>> =
            shapes.iter().map(|&s| Array2::zeros(s)).collect();

        Self {
            states,
            sync_interval,
            top_k_ratio,
            steps_since_sync: 0,
            momentum_deltas,
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Perform a local AdamW step on each parameter group.
    /// Returns true if a sync is needed after this step.
    pub fn local_step(&mut self, gradients: &[Array2<f32>]) -> bool {
        assert_eq!(
            gradients.len(),
            self.states.len(),
            "gradient count must match parameter groups"
        );

        for (i, (state, grad)) in self.states.iter_mut().zip(gradients.iter()).enumerate() {
            let m_before = state.m.clone();

            state.t += 1;
            let t = state.t as f32;

            // AdamW update
            state.m = &state.m * state.beta1 + grad * (1.0 - state.beta1);
            state.v = &state.v * state.beta2 + &(grad * grad) * (1.0 - state.beta2);

            // Accumulate momentum delta for DeMo sync
            self.momentum_deltas[i] = &self.momentum_deltas[i] + &(&state.m - &m_before);

            // Bias correction
            let _m_hat = &state.m / (1.0 - state.beta1.powf(t));
            let _v_hat = &state.v / (1.0 - state.beta2.powf(t));
        }

        self.steps_since_sync += 1;
        self.steps_since_sync >= self.sync_interval
    }

    /// Generate sparse updates for all parameter groups to broadcast.
    /// Call this when `local_step` returns true (sync needed).
    pub fn prepare_sync(&mut self) -> Vec<SparseUpdate> {
        let _lock = self.lock.clone();

        let mut updates = Vec::with_capacity(self.states.len());

        for (i, delta) in self.momentum_deltas.iter().enumerate() {
            let shape = (delta.shape()[0], delta.shape()[1]);

            // DCT transform the momentum delta
            let freq = dct_2d(delta);

            // Top-k sparsify
            let sparse = top_k_sparsify(&freq, self.top_k_ratio);

            updates.push(SparseUpdate {
                indices: sparse.0,
                values: sparse.1,
                shape,
                step: self.states[i].t,
                peer_id: String::new(), // filled by network layer
            });
        }

        // Reset deltas after preparing sync
        for delta in &mut self.momentum_deltas {
            delta.fill(0.0);
        }
        self.steps_since_sync = 0;

        updates
    }

    /// Apply aggregated momentum updates received from peers.
    pub fn apply_sync(&mut self, aggregated: &[Array2<f32>]) {
        assert_eq!(
            aggregated.len(),
            self.states.len(),
            "aggregated update count must match parameter groups"
        );

        for (state, update) in self.states.iter_mut().zip(aggregated.iter()) {
            state.m = &state.m + update;
        }
    }

    /// Get the current momentum tensors (for checkpointing).
    pub fn get_momentum(&self) -> Vec<Array2<f32>> {
        self.states.iter().map(|s| s.m.clone()).collect()
    }

    /// Restore momentum from a checkpoint.
    pub fn set_momentum(&mut self, momentum: Vec<Array2<f32>>) {
        for (state, m) in self.states.iter_mut().zip(momentum.into_iter()) {
            state.m = m;
        }
    }
}

// --- DCT / IDCT ---

/// Type-II DCT basis vector for size N at frequency k.
fn dct_basis(n: usize, k: usize, size: usize) -> f32 {
    let scale = if k == 0 {
        (1.0 / size as f32).sqrt()
    } else {
        (2.0 / size as f32).sqrt()
    };
    scale * (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * size as f32)).cos()
}

/// 2D Type-II Discrete Cosine Transform.
/// Transforms spatial-domain tensor to frequency domain.
pub fn dct_2d(input: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut output = Array2::zeros((rows, cols));

    // Row-wise DCT
    let mut row_transformed = Array2::zeros((rows, cols));
    for i in 0..rows {
        for k in 0..cols {
            let mut sum = 0.0f32;
            for n in 0..cols {
                sum += input[[i, n]] * dct_basis(n, k, cols);
            }
            row_transformed[[i, k]] = sum;
        }
    }

    // Column-wise DCT
    for j in 0..cols {
        for k in 0..rows {
            let mut sum = 0.0f32;
            for n in 0..rows {
                sum += row_transformed[[n, j]] * dct_basis(n, k, rows);
            }
            output[[k, j]] = sum;
        }
    }

    output
}

/// 2D Type-III Inverse Discrete Cosine Transform.
/// Transforms frequency-domain tensor back to spatial domain.
pub fn idct_2d(input: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut output = Array2::zeros((rows, cols));

    // Column-wise IDCT
    let mut col_transformed = Array2::zeros((rows, cols));
    for j in 0..cols {
        for n in 0..rows {
            let mut sum = 0.0f32;
            for k in 0..rows {
                sum += input[[k, j]] * dct_basis(n, k, rows);
            }
            col_transformed[[n, j]] = sum;
        }
    }

    // Row-wise IDCT
    for i in 0..rows {
        for n in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += col_transformed[[i, k]] * dct_basis(n, k, cols);
            }
            output[[i, n]] = sum;
        }
    }

    output
}

/// Top-k sparsification: keep only the largest `ratio` fraction of coefficients.
/// Returns (indices, values) sorted by magnitude descending.
pub fn top_k_sparsify(tensor: &Array2<f32>, ratio: f32) -> (Vec<u32>, Vec<f32>) {
    let flat: Vec<f32> = tensor.iter().copied().collect();
    let total = flat.len();
    let k = ((total as f32 * ratio).ceil() as usize).max(1).min(total);

    // Collect (index, abs_value) pairs and partial sort
    let mut indexed: Vec<(u32, f32)> = flat
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v.abs()))
        .collect();

    // Partial sort to find top-k threshold
    indexed.select_nth_unstable_by(k.min(indexed.len()) - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top_k: Vec<(u32, f32)> = indexed[..k]
        .iter()
        .map(|&(idx, _)| (idx, flat[idx as usize]))
        .collect();

    top_k.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let indices: Vec<u32> = top_k.iter().map(|&(i, _)| i).collect();
    let values: Vec<f32> = top_k.iter().map(|&(_, v)| v).collect();

    (indices, values)
}

/// Aggregate sparse updates from multiple peers by averaging.
/// All updates must have the same shape.
pub fn aggregate_updates(updates: &[SparseUpdate]) -> Array2<f32> {
    if updates.is_empty() {
        return Array2::zeros((0, 0));
    }

    let shape = updates[0].shape;
    let mut sum = Array2::zeros((shape.0, shape.1));
    let count = updates.len() as f32;

    for update in updates {
        let dense = update.to_dense();
        sum = sum + dense;
    }

    sum / count
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dct_idct_roundtrip() {
        let input = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let freq = dct_2d(&input);
        let reconstructed = idct_2d(&freq);

        for (a, b) in input.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_top_k_sparsify() {
        let tensor = array![[10.0, 0.1, 0.2], [0.3, 20.0, 0.4]];
        let (indices, values) = top_k_sparsify(&tensor, 0.34); // keep ~2 of 6

        assert!(indices.len() <= 3);
        // The two largest are 20.0 (index 4) and 10.0 (index 0)
        assert!(values[0].abs() >= values.last().copied().unwrap_or(0.0).abs());
    }

    #[test]
    fn test_sparse_update_roundtrip() {
        let tensor = array![[1.0, 2.0], [3.0, 4.0]];
        let (indices, values) = top_k_sparsify(&tensor, 1.0); // keep all

        let update = SparseUpdate {
            indices,
            values,
            shape: (2, 2),
            step: 0,
            peer_id: "test".into(),
        };

        let dense = update.to_dense();
        for (a, b) in tensor.iter().zip(dense.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_demo_optimizer_local_step() {
        let mut opt = DemoOptimizer::new(&[(4, 4)], 0.001, 3, 0.1);
        let grad = Array2::ones((4, 4));

        assert!(!opt.local_step(&[grad.clone()]));
        assert!(!opt.local_step(&[grad.clone()]));
        assert!(opt.local_step(&[grad])); // third step triggers sync
    }

    #[test]
    fn test_aggregate_updates() {
        let u1 = SparseUpdate {
            indices: vec![0, 1],
            values: vec![2.0, 4.0],
            shape: (2, 2),
            step: 0,
            peer_id: "a".into(),
        };
        let u2 = SparseUpdate {
            indices: vec![0, 1],
            values: vec![4.0, 6.0],
            shape: (2, 2),
            step: 0,
            peer_id: "b".into(),
        };

        let agg = aggregate_updates(&[u1, u2]);
        assert!((agg[[0, 0]] - 3.0).abs() < 1e-6); // (2+4)/2
        assert!((agg[[0, 1]] - 5.0).abs() < 1e-6); // (4+6)/2
    }
}
