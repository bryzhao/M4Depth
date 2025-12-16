"""
UseGeo Fine-tuning Script for M4Depth
=====================================

This script fine-tunes an M4Depth model (pre-trained on MidAir synthetic data)
on the UseGeo real-world drone dataset.

KEY CONCEPTS:
-------------
1. TRANSFER LEARNING: We start from MidAir weights rather than random initialization.
   The encoder has learned useful visual features from synthetic data that transfer
   to real images. We fine-tune with a lower learning rate to adapt these features
   without destroying the learned representations.

2. SIM-TO-REAL GAP: Synthetic images (MidAir) are "too perfect" - no noise, blur,
   or lighting variations. Real images (UseGeo) have these imperfections. Fine-tuning
   helps the model adapt to real-world image characteristics.

3. THE POSE PROBLEM: M4Depth estimates depth using MOTION PARALLAX - it measures
   how pixels shift between frames due to camera movement, then computes depth
   geometrically: depth = baseline * focal_length / parallax

   UseGeo provides NO camera poses! Without knowing camera motion, parallax-based
   depth estimation is impossible. Our workaround:
   - Use synthetic translation (0.5m forward) so the model doesn't crash
   - The geometric depth won't be accurate, but the model learns to recognize
     real-world depth patterns from the visual features
   - This is effectively using M4Depth as a learned visual prior

4. MULTI-SCALE LOSS: M4Depth outputs 6 pyramid levels (highest=192px, lowest=6px).
   We compute loss at each scale, weighted by 0.5^scale_idx so higher resolutions
   contribute more. This helps the model learn both global structure and fine details.

USAGE:
------
    # Fine-tune from MidAir checkpoint (recommended)
    python train_usegeo.py --from_midair_ckpt --epochs=30

    # Train from scratch (for comparison)
    python train_usegeo.py --from_scratch --epochs=50

OUTPUT:
-------
    - Checkpoints saved to: ./checkpoints_usegeo/
    - TensorBoard logs: ./checkpoints_usegeo/summaries/
    - Training history: ./checkpoints_usegeo/training_history.json
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

import tensorflow as tf
import argparse
import json
import numpy as np
from datetime import datetime

# M4Depth imports
from m4depth_network import M4Depth
from dataloaders import get_loader, DataloaderParameters


def create_model(depth_type, nbre_levels=6):
    """
    Create M4Depth model.

    Args:
        depth_type: 'map' for direct depth values (UseGeo), 'disp' for disparity (MidAir)
        nbre_levels: Number of pyramid levels (default 6)

    Returns:
        M4Depth model instance
    """
    model = M4Depth(
        depth_type=depth_type,
        nbre_levels=nbre_levels,
        is_training=True  # Enables multi-scale output for training
    )
    return model


def load_midair_checkpoint(model, ckpt_dir):
    """
    Load pre-trained weights from MidAir training.

    This is the key to transfer learning - we don't start from scratch,
    but from weights that already understand depth estimation.
    """
    checkpoint = tf.train.Checkpoint(model)

    # Try 'best' checkpoint first, fall back to 'train'
    weights_dir = os.path.join(ckpt_dir, "best")
    latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is None:
        weights_dir = os.path.join(ckpt_dir, "train")
        latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is not None:
        print(f"Loading MidAir checkpoint: {latest_ckpt}")
        # expect_partial() suppresses warnings about optimizer state not being restored
        checkpoint.restore(latest_ckpt).expect_partial()
        return True
    else:
        print("WARNING: No MidAir checkpoint found!")
        return False


def compute_loss(predictions, gt_depth):
    """
    Compute multi-scale L1 loss.

    Why L1 (not L2)?
    - L1 is more robust to outliers
    - Depth maps often have outliers at edges and occlusions

    Why multi-scale?
    - Coarse scales (6px) capture global structure
    - Fine scales (192px) capture details and edges
    - Weighting by 0.5^scale gives more importance to fine scales

    Args:
        predictions: List of 6 scale outputs, each is {'depth': tensor}
        gt_depth: Ground truth depth [batch, H, W, 1]

    Returns:
        total_loss: Weighted sum of per-scale losses
    """
    total_loss = tf.constant(0.0)

    for scale_idx, scale_pred in enumerate(predictions):
        pred_depth = scale_pred["depth"]

        # Resize GT to match this scale
        # Using nearest-neighbor to avoid interpolation artifacts in depth
        gt_scaled = gt_depth
        if pred_depth.shape[1:3] != gt_depth.shape[1:3]:
            gt_scaled = tf.image.resize(
                gt_depth,
                pred_depth.shape[1:3],
                method='nearest'
            )

        # Mask invalid depths
        # UseGeo depths are in meters, valid range ~10-200m
        valid_mask = (gt_scaled > 0.1) & (gt_scaled < 200.0)

        # L1 loss on valid pixels only
        l1_error = tf.abs(pred_depth - gt_scaled)
        l1_error = tf.where(valid_mask, l1_error, tf.zeros_like(l1_error))

        # Mean over valid pixels
        num_valid = tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-8
        scale_loss = tf.reduce_sum(l1_error) / num_valid

        # Weight by scale (0.5^idx): scale 0 (192px) has weight 1.0, scale 5 (6px) has weight 0.03
        scale_weight = 0.5 ** scale_idx
        total_loss += scale_loss * scale_weight

    return total_loss


def train_step(model, optimizer, batch):
    """
    Execute one training step with gradient descent.

    The training loop:
    1. Format batch data for model input
    2. Forward pass with gradient tape recording
    3. Compute multi-scale loss
    4. Backpropagate gradients
    5. Update weights
    """
    # Format data: model expects list of per-frame dicts
    seq_len = batch["depth"].shape[1]
    traj_samples = [{} for _ in range(seq_len)]

    for key in ["depth", "RGB_im", "new_traj", "rot", "trans"]:
        # Unstack along sequence dimension
        value_list = tf.unstack(batch[key], axis=1)
        for i, item in enumerate(value_list):
            traj_samples[i][key] = item

    camera = batch["camera"]

    with tf.GradientTape() as tape:
        # Forward pass
        # Output: predictions[seq_idx][scale_idx] = {'depth': tensor}
        predictions = model([traj_samples, camera], training=True)

        # Use last frame's predictions (has seen the full sequence)
        last_frame_preds = predictions[-1]

        # Ground truth is last frame's depth
        gt_depth = traj_samples[-1]["depth"]

        # Multi-scale loss
        loss = compute_loss(last_frame_preds, gt_depth)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Clip gradients to prevent explosions
    gradients = [
        tf.clip_by_norm(g, 1.0) if g is not None else g
        for g in gradients
    ]

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def evaluate(model, dataset, max_batches=50):
    """
    Evaluate model on validation/test set.

    Returns mean loss over the dataset (or first max_batches).
    """
    losses = []

    for i, batch in enumerate(dataset):
        if max_batches and i >= max_batches:
            break

        # Format data
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        for key in ["depth", "RGB_im", "new_traj", "rot", "trans"]:
            value_list = tf.unstack(batch[key], axis=1)
            for j, item in enumerate(value_list):
                traj_samples[j][key] = item

        # Forward pass (no gradients)
        # When training=False, model returns {"depth": tensor} directly at full resolution
        predictions = model([traj_samples, batch["camera"]], training=False)

        # Get predicted depth - model with training=False returns dict directly
        pred_depth = predictions["depth"]
        gt_depth = traj_samples[-1]["depth"]

        # Resize pred to match gt if needed
        if pred_depth.shape[1:3] != gt_depth.shape[1:3]:
            pred_depth = tf.image.resize(pred_depth, gt_depth.shape[1:3])

        # Compute error on valid pixels
        valid_mask = (gt_depth > 0.1) & (gt_depth < 200.0)
        l1_error = tf.abs(pred_depth - gt_depth)
        l1_error = tf.where(valid_mask, l1_error, tf.zeros_like(l1_error))

        loss = tf.reduce_sum(l1_error) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-8)
        losses.append(loss.numpy())

    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune M4Depth on UseGeo dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2, limited by GPU memory)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, lower than training from scratch)')
    parser.add_argument('--from_midair_ckpt', action='store_true',
                        help='Start from MidAir pre-trained checkpoint')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from random initialization')
    parser.add_argument('--midair_ckpt_dir', type=str, default='./checkpoints',
                        help='Path to MidAir checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_usegeo',
                        help='Output directory for checkpoints')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip validation during training (avoids batch size mismatch issues)')
    parser.add_argument('--data_dir', type=str, default='data/usegeo',
                        help='Data directory (default: data/usegeo, use data/usegeo_poses for real poses)')
    args = parser.parse_args()

    # Validate args
    if not args.from_midair_ckpt and not args.from_scratch:
        print("ERROR: Specify --from_midair_ckpt or --from_scratch")
        return

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'best'), exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'summaries')
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset config
    print("="*60)
    print("UseGeo Fine-tuning")
    print("="*60)

    with open('datasets_location.json') as f:
        db_config = json.load(f)

    # Create dataloaders
    print("\nLoading UseGeo dataset...")

    train_loader = get_loader('usegeo')
    train_settings = DataloaderParameters(
        db_path_config=db_config,
        records_path=f'{args.data_dir}/train_data',
        db_seq_len=4,   # Load 4 frames from disk
        seq_len=4,      # Use 4 frames per training sample (need multiple for parallax)
        augment=True    # Enable color augmentation
    )
    train_loader.get_dataset('train', train_settings, batch_size=args.batch_size)
    train_dataset = train_loader.dataset
    print(f"  Train samples: {train_loader.length}")
    print(f"  Data dir: {args.data_dir}")

    val_loader = get_loader('usegeo')
    val_settings = DataloaderParameters(
        db_path_config=db_config,
        records_path=f'{args.data_dir}/test_data',
        db_seq_len=4,
        seq_len=4,
        augment=False   # No augmentation for validation
    )
    val_loader.get_dataset('eval', val_settings, batch_size=1)
    val_dataset = val_loader.dataset
    print(f"  Val samples: {val_loader.length}")

    # Create model
    print("\nCreating model...")
    model = create_model(depth_type=train_loader.depth_type)

    # Build model by running one forward pass
    print("  Building model graph...")
    for batch in train_dataset.take(1):
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        for key in ["depth", "RGB_im", "new_traj", "rot", "trans"]:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item
        _ = model([traj_samples, batch["camera"]], training=False)
    print("  Model built!")

    # Load checkpoint if requested
    if args.from_midair_ckpt:
        loaded = load_midair_checkpoint(model, args.midair_ckpt_dir)
        if not loaded:
            print("Falling back to random initialization")
    else:
        print("Training from scratch (random initialization)")

    # Create optimizer
    # Using legacy Adam for TF2.15 compatibility
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)

    # TensorBoard writer
    train_writer = tf.summary.create_file_writer(log_dir)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print("-"*60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_losses = []
        batch_count = 0

        for batch in train_dataset:
            loss = train_step(model, optimizer, batch)
            epoch_losses.append(loss.numpy())
            batch_count += 1

            # Progress update every 10 batches
            if batch_count % 10 == 0:
                print(f"  Epoch {epoch+1}, batch {batch_count}, loss: {loss.numpy():.4f}")

        # Epoch stats
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation (skip if requested - avoids batch size mismatch issue)
        # The batch size mismatch occurs because M4Depth maintains temporal state
        # between forward passes. Training with batch_size=2 creates state with that
        # shape, then validation with batch_size=1 causes a shape mismatch error.
        if args.skip_validation:
            val_loss = train_loss  # Use train loss as proxy
            print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f} (validation skipped)")
        else:
            val_loss = evaluate(model, val_dataset)
            print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
        val_losses.append(val_loss)

        # TensorBoard logging
        with train_writer.as_default():
            tf.summary.scalar('loss/train', train_loss, step=epoch)
            if not args.skip_validation:
                tf.summary.scalar('loss/val', val_loss, step=epoch)

        # Save checkpoint
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.save(os.path.join(args.output_dir, 'train', 'ckpt'))

        # Save best model (use train loss when validation is skipped)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint.save(os.path.join(args.output_dir, 'best', 'ckpt'))
            print(f"  -> New best model! (loss={val_loss:.4f})")

    # Save training history
    history = {
        'train_loss': [float(x) for x in train_losses],
        'val_loss': [float(x) for x in val_losses],
        'epochs': list(range(1, args.epochs + 1))
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
