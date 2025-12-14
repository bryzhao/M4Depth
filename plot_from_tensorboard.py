"""
Quick script to plot training curves from TensorBoard event files.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
from pathlib import Path


def load_tensorboard_data(logdir):
    """Load tensor data from TensorBoard event files."""
    import tensorflow as tf

    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Get available tensor tags (Keras stores metrics as tensors, not scalars)
    tags = ea.Tags().get('tensors', [])

    # Filter to just the metrics we care about
    metric_tags = [t for t in tags if 'loss' in t.lower() or 'rmse' in t.lower()]
    print(f"Available metrics: {metric_tags}")

    data = {}
    for tag in metric_tags:
        events = ea.Tensors(tag)
        steps = [e.step for e in events]
        # Tensor values need to be converted from TensorProto
        values = [tf.make_ndarray(e.tensor_proto).item() for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data


def plot_training_curves(data, output_path):
    """Plot training curves."""
    # Find loss and RMSE metrics (prefer epoch-level over batch-level)
    loss_key = None
    rmse_key = None

    for key in data.keys():
        if 'epoch_loss' in key.lower():
            loss_key = key
        elif 'loss' in key.lower() and loss_key is None:
            loss_key = key
        if 'epoch_rmse' in key.lower():
            rmse_key = key
        elif 'rmse' in key.lower() and rmse_key is None:
            rmse_key = key

    print(f"Using loss key: {loss_key}")
    print(f"Using RMSE key: {rmse_key}")

    if not loss_key and not rmse_key:
        print("No loss or RMSE metrics found. Available keys:")
        for k in data.keys():
            print(f"  - {k}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    if loss_key and data[loss_key]['values']:
        ax1 = axes[0]
        steps = data[loss_key]['steps']
        values = data[loss_key]['values']
        ax1.plot(steps, values, 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Training Loss ({loss_key})', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Print summary
        print(f"\nLoss Summary:")
        print(f"  Initial: {values[0]:.4f}")
        print(f"  Final: {values[-1]:.4f}")
        print(f"  Min: {min(values):.4f}")
        print(f"  Reduction: {((values[0] - values[-1]) / values[0] * 100):.1f}%")

    # Plot RMSE
    if rmse_key and data[rmse_key]['values']:
        ax2 = axes[1]
        steps = data[rmse_key]['steps']
        values = data[rmse_key]['values']
        ax2.plot(steps, values, 'r-', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('RMSE Log', fontsize=12)
        ax2.set_title(f'RMSE Log ({rmse_key})', fontsize=14)
        ax2.grid(True, alpha=0.3)

        print(f"\nRMSE Log Summary:")
        print(f"  Initial: {values[0]:.4f}")
        print(f"  Final: {values[-1]:.4f}")
        print(f"  Min: {min(values):.4f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs')
    parser.add_argument('--logdir', type=str, default='./checkpoints/summaries/train',
                        help='TensorBoard log directory')
    parser.add_argument('--output', type=str, default='training_curves.png',
                        help='Output plot filename')
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: log directory not found: {logdir}")
        return

    print(f"Loading TensorBoard data from: {logdir}")
    data = load_tensorboard_data(str(logdir))

    if not data:
        print("No data found in TensorBoard logs")
        return

    plot_training_curves(data, args.output)


if __name__ == '__main__':
    main()
