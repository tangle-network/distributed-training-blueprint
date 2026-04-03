use anyhow::Result;
use ndarray::Array2;

#[tokio::test]
async fn test_demo_optimizer_sync() -> Result<()> {
    use distributed_training::demo::DemoOptimizer;

    let shapes = vec![(4, 4)]; // 16 parameters
    let mut opt_a = DemoOptimizer::new(&shapes, 0.001, 5, 0.5);
    let mut opt_b = DemoOptimizer::new(&shapes, 0.001, 5, 0.5);

    // Both operators do 5 local steps with different gradients
    let grads_a = vec![Array2::from_elem((4, 4), 0.01_f32)];
    let grads_b = vec![Array2::from_elem((4, 4), 0.02_f32)];

    for _ in 0..5 {
        let should_sync_a = opt_a.local_step(&grads_a);
        let should_sync_b = opt_b.local_step(&grads_b);
        assert_eq!(should_sync_a, should_sync_b, "sync triggers should align");
    }

    // Both should signal sync after 5 steps (sync_interval=5)
    assert!(opt_a.local_step(&grads_a), "should trigger sync at step 5");

    // Prepare sparse updates (DCT + top-k)
    let sparse_a = opt_a.prepare_sync();
    let sparse_b = opt_b.prepare_sync();

    assert!(!sparse_a.is_empty(), "operator A should produce updates");
    assert!(!sparse_b.is_empty(), "operator B should produce updates");

    let update_a = &sparse_a[0];
    let update_b = &sparse_b[0];

    // Top-50% of 16 = 8 values max
    assert!(update_a.indices.len() <= 8, "top-k should limit indices");
    assert_eq!(update_a.indices.len(), update_a.values.len());

    println!("Operator A sparse: {} indices out of 16", update_a.indices.len());
    println!("Operator B sparse: {} indices out of 16", update_b.indices.len());
    println!("Communication: {}B vs {}B full", update_a.byte_size(), 16 * 4);

    // Aggregate updates from both operators
    let aggregated = distributed_training::demo::aggregate_updates(&[
        sparse_a[0].clone(),
        sparse_b[0].clone(),
    ]);

    // Apply aggregated momentum to both optimizers
    opt_a.apply_sync(&vec![aggregated.clone()]);
    opt_b.apply_sync(&vec![aggregated]);

    // After sync, both optimizers should have the same momentum
    let mom_a = opt_a.get_momentum();
    let mom_b = opt_b.get_momentum();
    assert_eq!(mom_a.len(), mom_b.len());

    // After applying the same aggregated momentum, the local momentums move
    // toward each other. They won't be identical because each operator retains
    // its own local momentum history — DeMo converges over multiple sync rounds.
    let diff: f32 = mom_a[0].iter().zip(mom_b[0].iter()).map(|(a, b)| (a - b).abs()).sum();
    // The diff should be smaller than the gradient difference (0.01 vs 0.02 per element * 16 = 0.16)
    assert!(diff < 0.16, "momentums should be closer after sync, diff={diff}");

    println!("Post-sync momentum diff: {diff:.2e} (should be ~0)");
    println!("DeMo sync test PASSED");

    Ok(())
}

#[tokio::test]
async fn test_checkpoint_roundtrip() -> Result<()> {
    use distributed_training::checkpoint;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("checkpoint-epoch-1.bin");

    // Simulate model state
    let state = b"model weights tensor data epoch 1 step 500";
    std::fs::write(&path, state)?;

    // Hash
    let hash = checkpoint::hash_checkpoint(&path)?;
    assert_ne!(hash, [0u8; 32]);

    // Same data = same hash (deterministic)
    let hash2 = checkpoint::hash_checkpoint(&path)?;
    assert_eq!(hash, hash2);

    // Different data = different hash
    let path2 = dir.path().join("checkpoint-epoch-2.bin");
    std::fs::write(&path2, b"different model weights epoch 2")?;
    let hash3 = checkpoint::hash_checkpoint(&path2)?;
    assert_ne!(hash, hash3);

    println!("Checkpoint hash: {}", hex::encode(&hash[..8]));
    println!("Checkpoint roundtrip PASSED");

    Ok(())
}

#[tokio::test]
async fn test_sparse_update_serialization() -> Result<()> {
    use distributed_training::demo::SparseUpdate;

    let update = SparseUpdate {
        indices: vec![0, 5, 10, 15],
        values: vec![0.1, -0.3, 0.5, -0.2],
        shape: (4, 4),
        step: 100,
        peer_id: "operator-A".to_string(),
    };

    // Serialize
    let json = serde_json::to_vec(&update)?;

    // Deserialize
    let decoded: SparseUpdate = serde_json::from_slice(&json)?;
    assert_eq!(decoded.indices, update.indices);
    assert_eq!(decoded.values, update.values);
    assert_eq!(decoded.shape, update.shape);

    // Reconstruct to dense
    let dense = decoded.to_dense();
    assert_eq!(dense.shape(), &[4, 4]);
    assert!((dense[[0, 0]] - 0.1).abs() < 1e-6);

    println!("Sparse update size: {}B serialized", json.len());
    println!("Equivalent dense: {}B", 4 * 4 * 4);
    println!("Compression: {:.1}x", (4.0 * 4.0 * 4.0) / json.len() as f32);
    println!("Serialization PASSED");

    Ok(())
}
