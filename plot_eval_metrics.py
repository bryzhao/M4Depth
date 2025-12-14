"""
Plot evaluation metrics comparing our results vs paper results.
Creates a bar chart showing all 7 standard depth estimation metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Paper's reported results (Table 2, MidAir)
PAPER_RESULTS = {
    'Abs Rel': 0.105,
    'Sq Rel': 3.454,
    'RMSE': 7.043,
    'RMSE Log': 0.186,
    'δ<1.25': 0.919,
    'δ<1.25²': 0.953,
    'δ<1.25³': 0.969,
}

def load_our_results(filepath='./checkpoints/perfs-midair.txt'):
    """Load our evaluation results from perfs file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    values = [float(line.strip()) for line in lines if line.strip()]

    return {
        'Abs Rel': values[0],
        'Sq Rel': values[1],
        'RMSE': values[2],
        'RMSE Log': values[3],
        'δ<1.25': values[4],
        'δ<1.25²': values[5],
        'δ<1.25³': values[6],
    }

def plot_comparison(our_results, paper_results, output_path='eval_comparison.png'):
    """Create comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Split metrics into two groups
    # Lower is better: Abs Rel, Sq Rel, RMSE, RMSE Log
    # Higher is better: δ metrics

    lower_better = ['Abs Rel', 'Sq Rel', 'RMSE', 'RMSE Log']
    higher_better = ['δ<1.25', 'δ<1.25²', 'δ<1.25³']

    # Plot 1: Error metrics (lower is better)
    ax1 = axes[0]
    x = np.arange(len(lower_better))
    width = 0.35

    ours_vals = [our_results[k] for k in lower_better]
    paper_vals = [paper_results[k] for k in lower_better]

    bars1 = ax1.bar(x - width/2, ours_vals, width, label='Ours', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, paper_vals, width, label='Paper', color='#3498db', alpha=0.8)

    ax1.set_ylabel('Value (↓ lower is better)')
    ax1.set_title('Error Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lower_better, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, ours_vals):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, paper_vals):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    # Plot 2: Accuracy metrics (higher is better)
    ax2 = axes[1]
    x = np.arange(len(higher_better))

    ours_vals = [our_results[k] for k in higher_better]
    paper_vals = [paper_results[k] for k in higher_better]

    bars1 = ax2.bar(x - width/2, ours_vals, width, label='Ours', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x + width/2, paper_vals, width, label='Paper', color='#3498db', alpha=0.8)

    ax2.set_ylabel('Value (↑ higher is better)')
    ax2.set_title('Accuracy Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(higher_better)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.85, 1.0)  # Zoom in on relevant range

    # Add value labels
    for bar, val in zip(bars1, ours_vals):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, paper_vals):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    plt.suptitle('M4Depth Evaluation: Our Results vs Paper (MidAir Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Metric':<12} {'Ours':>10} {'Paper':>10} {'Diff':>10} {'Status':>10}")
    print("-"*60)

    for metric in list(lower_better) + list(higher_better):
        ours = our_results[metric]
        paper = paper_results[metric]
        diff = ours - paper

        # Determine if better or worse
        if metric in lower_better:
            status = "✓ better" if diff < 0 else ("~ same" if abs(diff/paper) < 0.05 else "worse")
        else:
            status = "✓ better" if diff > 0 else ("~ same" if abs(diff/paper) < 0.05 else "worse")

        print(f"{metric:<12} {ours:>10.4f} {paper:>10.4f} {diff:>+10.4f} {status:>10}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation metrics comparison')
    parser.add_argument('--results', type=str, default='./checkpoints/perfs-midair.txt',
                        help='Path to our results file')
    parser.add_argument('--output', type=str, default='eval_comparison.png',
                        help='Output plot filename')
    args = parser.parse_args()

    print(f"Loading results from: {args.results}")
    our_results = load_our_results(args.results)

    plot_comparison(our_results, PAPER_RESULTS, args.output)


if __name__ == '__main__':
    main()
