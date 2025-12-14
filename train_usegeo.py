"""
fine-tune m4depth on usegeo dataset using midair pretrained checkpoint.
this script handles the sim-to-real transfer from synthetic (midair) to real (usegeo) data.

usage:
    # fine-tune from midair checkpoint
    python train_usegeo.py --epochs=50 --batch_size=2 --from_midair_ckpt

    # train from scratch (for comparison)
    python train_usegeo.py --epochs=50 --batch_size=2 --from_scratch
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import argparse
import json
import numpy as np
from datetime import datetime

from m4depth_network import M4Depth
from m4depth_options import M4DepthOptions
from dataloaders import get_loader, DataloaderParameters
# Callbacks not used - custom training loop with tf.GradientTape


def create_model(depth_type, nbre_levels=6):
    """create m4depth model"""
    model = M4Depth(
        depth_type=depth_type,
        nbre_levels=nbre_levels,
        is_training=True
    )
    return model


def load_midair_checkpoint(model, ckpt_dir):
    """load weights from midair pretrained checkpoint"""
    checkpoint = tf.train.Checkpoint(model)
    weights_dir = os.path.join(ckpt_dir, "best")
    latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is None:
        weights_dir = os.path.join(ckpt_dir, "train")
        latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is not None:
        print(f"loading midair checkpoint: {latest_ckpt}")
        checkpoint.restore(latest_ckpt).expect_partial()
        return True
    else:
        print("warning: no midair checkpoint found")
        return False


def train_step(model, optimizer, batch, metrics_tracker):
    """single training step"""
    # prepare data in model format
    seq_len = batch["depth"].shape[1]
    traj_samples = [{} for _ in range(seq_len)]
    attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]

    for key in attribute_list:
        value_list = tf.unstack(batch[key], axis=1)
        for i, item in enumerate(value_list):
            traj_samples[i][key] = item

    camera = batch["camera"]

    with tf.GradientTape() as tape:
        # forward pass
        predictions = model([traj_samples, camera], training=True)

        # compute loss - predictions is a nested list: predictions[scale][time_output]
        # each scale may have different number of outputs, so we only use the last frame's depth
        # as ground truth and compute loss at each scale
        total_loss = tf.constant(0.0)
        gt_depth_full = traj_samples[-1]["depth"]  # use last frame as ground truth

        for scale_idx, scale_preds in enumerate(predictions):
            # get the last prediction at this scale (usually there's just one per scale)
            pred = scale_preds[-1] if isinstance(scale_preds, list) else scale_preds
            pred_depth = pred["depth"]

            # resize gt to match prediction scale
            gt_depth = gt_depth_full
            if pred_depth.shape[1:3] != gt_depth.shape[1:3]:
                gt_depth = tf.image.resize(gt_depth, pred_depth.shape[1:3])

            # mask invalid depths (zeros and very large values)
            valid_mask = (gt_depth > 0.1) & (gt_depth < 200.0)

            # l1 loss on valid pixels
            l1_loss = tf.abs(pred_depth - gt_depth)
            l1_loss = tf.where(valid_mask, l1_loss, tf.zeros_like(l1_loss))
            loss = tf.reduce_sum(l1_loss) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-8)

            # weight by scale (lower resolution = lower weight)
            scale_weight = 0.5 ** scale_idx
            total_loss += loss * scale_weight

    # compute gradients and update
    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # update metrics
    metrics_tracker.update_state(total_loss)

    return total_loss


def evaluate(model, dataset, num_batches=None):
    """evaluate model on dataset"""
    losses = []
    batch_count = 0

    for batch in dataset:
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]

        for key in attribute_list:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item

        camera = batch["camera"]

        # forward pass (no training)
        predictions = model([traj_samples, camera], training=False)

        # compute loss on final scale only
        pred_depth = predictions[-1][0]["depth"]
        gt_depth = traj_samples[-1]["depth"]

        if pred_depth.shape[1:3] != gt_depth.shape[1:3]:
            gt_depth = tf.image.resize(gt_depth, pred_depth.shape[1:3])

        valid_mask = (gt_depth > 0.1) & (gt_depth < 200.0)
        l1_loss = tf.abs(pred_depth - gt_depth)
        l1_loss = tf.where(valid_mask, l1_loss, tf.zeros_like(l1_loss))
        loss = tf.reduce_sum(l1_loss) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + 1e-8)

        losses.append(loss.numpy())
        batch_count += 1

        if num_batches and batch_count >= num_batches:
            break

    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune M4Depth on UseGeo')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (lower for fine-tuning)')
    parser.add_argument('--from_midair_ckpt', action='store_true', help='start from midair checkpoint')
    parser.add_argument('--from_scratch', action='store_true', help='train from scratch')
    parser.add_argument('--midair_ckpt_dir', type=str, default='./ckpt', help='path to midair checkpoint')
    parser.add_argument('--output_dir', type=str, default='./ckpt_usegeo', help='output directory')
    parser.add_argument('--log_dir', type=str, default='./ckpt_usegeo/summaries', help='tensorboard log directory')
    args = parser.parse_args()

    # validate arguments
    if not args.from_midair_ckpt and not args.from_scratch:
        print("please specify --from_midair_ckpt or --from_scratch")
        return

    # create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # load dataset config
    with open('datasets_location.json') as f:
        db_config = json.load(f)

    # create dataloader
    print("loading usegeo dataset...")
    loader = get_loader('usegeo')

    train_settings = DataloaderParameters(
        db_path_config=db_config,
        records_path='data/usegeo/train_data',
        db_seq_len=4,
        seq_len=2,
        augment=True
    )
    train_dataset = loader.get_dataset('train', train_settings, batch_size=args.batch_size)
    train_dataset = loader.dataset
    print(f"training samples: {loader.length}")

    # create validation dataloader
    val_loader = get_loader('usegeo')
    val_settings = DataloaderParameters(
        db_path_config=db_config,
        records_path='data/usegeo/test_data',
        db_seq_len=4,
        seq_len=2,
        augment=False
    )
    val_dataset = val_loader.get_dataset('eval', val_settings, batch_size=1)
    val_dataset = val_loader.dataset
    print(f"validation samples: {val_loader.length}")

    # create model
    print("creating model...")
    model = create_model(depth_type=loader.depth_type)

    # build model with one forward pass
    for batch in train_dataset.take(1):
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        for key in ["depth", "RGB_im", "new_traj", "rot", "trans"]:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item
        _ = model([traj_samples, batch["camera"]], training=False)
    print("model built")

    # load midair checkpoint if requested
    if args.from_midair_ckpt:
        loaded = load_midair_checkpoint(model, args.midair_ckpt_dir)
        if not loaded:
            print("could not load midair checkpoint, starting from scratch")
    else:
        print("training from scratch")

    # create optimizer with lower learning rate for fine-tuning
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr, clipnorm=1.0)

    # tensorboard writer
    train_summary_writer = tf.summary.create_file_writer(args.log_dir)

    # training loop
    print(f"\nstarting training for {args.epochs} epochs...")
    print(f"learning rate: {args.lr}")
    print(f"batch size: {args.batch_size}")
    print("-" * 50)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # training
        epoch_losses = []
        batch_count = 0

        for batch in train_dataset:
            loss_tracker = tf.keras.metrics.Mean()
            loss = train_step(model, optimizer, batch, loss_tracker)
            epoch_losses.append(loss.numpy())
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"  epoch {epoch+1}, batch {batch_count}, loss: {loss.numpy():.4f}")

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # validation
        val_loss = evaluate(model, val_dataset, num_batches=50)
        val_losses.append(val_loss)

        # log to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss/train', train_loss, step=epoch)
            tf.summary.scalar('loss/val', val_loss, step=epoch)

        print(f"epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # save checkpoint
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.save(os.path.join(args.output_dir, 'train', 'ckpt'))
        model.save_weights(os.path.join(args.output_dir, 'train', 'latest_ckpt.weights.h5'))

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.join(args.output_dir, 'best'), exist_ok=True)
            checkpoint.save(os.path.join(args.output_dir, 'best', 'ckpt'))
            model.save_weights(os.path.join(args.output_dir, 'best', 'latest_ckpt.weights.h5'))
            print(f"  new best model saved (val_loss={val_loss:.4f})")

        # garbage collection for mac memory management
        import gc
        gc.collect()

    # save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs': list(range(1, args.epochs + 1))
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    print("\ntraining complete!")
    print(f"best validation loss: {best_val_loss:.4f}")
    print(f"checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
