"""
plot training loss from training.log file.
generates a plot showing loss progression across epochs.
"""

import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_training_log(log_path):
    """
    extract epoch-level loss values from the training log.
    looks for the final loss reported at the end of each epoch.
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # pattern matches the epoch summary line at the end of each epoch
    # example: "479/479 [...] - 1668s 3s/step - RMSE_log: 0.4469 - loss: 1.2606 - lr: 1.0000e-04"
    epoch_pattern = r'Epoch (\d+)/(\d+)'
    loss_pattern = r'\d+/\d+ \[=+\] - \d+s \d+s/step - RMSE_log: ([\d.]+) - loss: ([\d.]+) - lr:'

    epochs = []
    losses = []
    rmse_logs = []

    # split by epoch markers and extract final loss for each
    epoch_blocks = re.split(r'(Epoch \d+/\d+)', content)

    current_epoch = 0
    for i, block in enumerate(epoch_blocks):
        epoch_match = re.match(r'Epoch (\d+)/(\d+)', block)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        # find the final loss line in this epoch block (the one with "======" complete bar)
        final_matches = re.findall(loss_pattern, block)
        if final_matches and current_epoch > 0:
            # take the last match which is the epoch-end summary
            rmse_log, loss = final_matches[-1]
            epochs.append(current_epoch)
            losses.append(float(loss))
            rmse_logs.append(float(rmse_log))

    return epochs, losses, rmse_logs


def plot_training_curves(epochs, losses, rmse_logs, output_path):
    """
    create and save training loss plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss plot
    ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss vs Epoch', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # rmse log plot
    ax2.plot(epochs, rmse_logs, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('RMSE (log)', fontsize=12)
    ax2.set_title('RMSE Log vs Epoch', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"plot saved to: {output_path}")

    # also display if possible
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument('--log', type=str, default='training.log',
                        help='path to training log file')
    parser.add_argument('--output', type=str, default='training_loss.png',
                        help='output plot filename')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"error: log file not found: {log_path}")
        return

    print(f"parsing {log_path}...")
    epochs, losses, rmse_logs = parse_training_log(log_path)

    if not epochs:
        print("error: no completed epochs found in log file")
        return

    print(f"\ntraining progress summary:")
    print(f"  completed epochs: {len(epochs)}")
    print(f"  epoch range: {min(epochs)} - {max(epochs)}")
    print(f"  initial loss: {losses[0]:.4f}")
    print(f"  current loss: {losses[-1]:.4f}")
    print(f"  loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print()

    # print epoch-by-epoch summary
    print("epoch-by-epoch:")
    for e, l, r in zip(epochs, losses, rmse_logs):
        print(f"  epoch {e:3d}: loss={l:.4f}, rmse_log={r:.4f}")
    print()

    plot_training_curves(epochs, losses, rmse_logs, args.output)


if __name__ == '__main__':
    main()
