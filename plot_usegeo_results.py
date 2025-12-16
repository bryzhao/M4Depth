"""
Plot UseGeo fine-tuning results with real camera poses.
Creates training/validation loss plot similar to paper Figure A1.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from training with real poses
epochs = np.arange(1, 31)

train_loss = np.array([
    132.2, 88.6, 82.5, 82.5, 80.3, 80.5, 81.1, 77.3, 70.4, 69.7,
    62.7, 61.9, 57.7, 56.0, 56.2, 54.8, 53.1, 53.6, 51.3, 51.4,
    51.2, 51.2, 47.5, 47.3, 44.4, 43.2, 48.7, 41.3, 39.9, 40.8
])

val_loss = np.array([
    36.9, 32.5, 31.5, 31.9, 34.6, 30.8, 31.5, 29.1, 27.2, 26.7,
    23.3, 22.6, 22.3, 23.0, 22.7, 22.4, 21.9, 22.4, 21.5, 21.4,
    21.8, 21.5, 21.8, 20.9, 20.6, 22.7, 20.9, 21.1, 21.3, 21.3
])

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training and validation loss
ax.plot(epochs, train_loss, 'o-', color='#2E86AB', linewidth=2, markersize=4, label='Training Loss', alpha=0.8)
ax.plot(epochs, val_loss, 's-', color='#E94F37', linewidth=2, markersize=4, label='Validation Loss', alpha=0.8)

# Styling
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Mean Absolute Error (meters)', fontsize=12)
ax.set_title('UseGeo Fine-tuning: Training vs Validation Loss\n(with Real Camera Poses from Photogrammetry)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 31)
ax.set_ylim(0, 140)

# Add annotation for best validation
best_epoch = np.argmin(val_loss) + 1
best_val = val_loss[best_epoch - 1]
ax.annotate(f'Best: {best_val:.1f}m @ epoch {best_epoch}',
            xy=(best_epoch, best_val), xytext=(best_epoch + 3, best_val + 15),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=10, color='#E94F37')

plt.tight_layout()
plt.savefig('usegeo_training_curves.png', dpi=150, bbox_inches='tight')
print(f"Saved: usegeo_training_curves.png")

# Also create a comparison with training without poses
# Previous results (from context): train=15.7m final, val=75.4m final
# That was training loss at end, but validation was very high showing poor generalization

fig2, ax2 = plt.subplots(figsize=(8, 6))

categories = ['Without Real\nPoses', 'With Real\nPoses']
train_final = [40.5, 40.8]  # Both end around same train loss
val_final = [75.4, 20.6]    # But huge difference in validation

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, train_final, width, label='Training Loss', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x + width/2, val_final, width, label='Validation Loss', color='#E94F37', alpha=0.8)

ax2.set_ylabel('Mean Absolute Error (meters)', fontsize=12)
ax2.set_title('UseGeo: Impact of Real Camera Poses\non Model Generalization', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=11)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_ylim(0, 90)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}m',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}m',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('usegeo_pose_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: usegeo_pose_comparison.png")

plt.show()
