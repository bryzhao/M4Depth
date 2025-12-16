"""
visualize depth predictions on usegeo dataset.
generates side-by-side comparisons of rgb, predicted depth, and ground truth.

usage:
    # visualize with midair pretrained model (before fine-tuning)
    python visualize_usegeo.py --ckpt_dir=./ckpt --output_dir=visualizations_usegeo_pretrained

    # visualize with usegeo fine-tuned model
    python visualize_usegeo.py --ckpt_dir=./ckpt_usegeo --output_dir=visualizations_usegeo_finetuned
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from m4depth_network import M4Depth
from dataloaders import get_loader, DataloaderParameters


def visualize_samples(model, dataset, num_samples=10, output_dir="visualizations_usegeo", stride=1):
    """
    generate side-by-side comparisons of rgb, predicted depth, and ground truth.
    stride: skip every N samples to get variety
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_count = 0
    iter_count = 0

    for batch in dataset:
        iter_count += 1

        # skip samples based on stride
        if stride > 1 and (iter_count - 1) % stride != 0:
            continue
        if sample_count >= num_samples:
            break

        # prepare batch for model
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
        for key in attribute_list:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item

        camera = batch["camera"]

        # get predictions
        predictions = model([traj_samples, camera], training=False)

        # extract data
        rgb_images = batch["RGB_im"].numpy()  # (batch, seq, h, w, 3)
        depth_gt = batch["depth"].numpy()      # (batch, seq, h, w, 1)

        # get predicted depth
        if isinstance(predictions, dict):
            depth_pred = predictions["depth"].numpy()
        else:
            depth_pred = predictions[-1][0]["depth"].numpy()

        batch_size = rgb_images.shape[0]

        for b in range(batch_size):
            if sample_count >= num_samples:
                break

            # get last frame in sequence
            rgb = rgb_images[b, -1]
            gt = depth_gt[b, -1, :, :, 0]

            # resize prediction to match gt
            pred = tf.image.resize(
                depth_pred[b:b+1],
                gt.shape[:2],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ).numpy()[0, :, :, 0]

            # create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # rgb image
            axes[0].imshow(rgb)
            axes[0].set_title("RGB Input")
            axes[0].axis('off')

            # ground truth depth (usegeo depth is in meters, ~10-150m range)
            gt_clipped = np.clip(gt, 1, 200)
            im1 = axes[1].imshow(np.log(gt_clipped), cmap='plasma')
            axes[1].set_title(f"Ground Truth Depth\nmin={gt.min():.1f}m, max={gt.max():.1f}m")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            # predicted depth
            pred_clipped = np.clip(pred, 1, 200)
            im2 = axes[2].imshow(np.log(pred_clipped), cmap='plasma')
            axes[2].set_title(f"Predicted Depth\nmin={pred.min():.1f}m, max={pred.max():.1f}m")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            # error map
            valid_mask = (gt > 0.1) & (gt < 200)
            error = np.abs(pred - gt)
            error[~valid_mask] = 0

            im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=50)
            mean_error = error[valid_mask].mean() if valid_mask.any() else 0
            axes[3].set_title(f"Absolute Error\nmean={mean_error:.2f}m")
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

            plt.tight_layout()

            # save
            save_path = os.path.join(output_dir, f"sample_{sample_count:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"saved: {save_path}")
            sample_count += 1

    print(f"\n{sample_count} visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize UseGeo depth predictions')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples')
    parser.add_argument('--output_dir', type=str, default='visualizations_usegeo', help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help='checkpoint directory')
    parser.add_argument('--stride', type=int, default=5, help='skip every N samples')
    parser.add_argument('--use_train_data', action='store_true', help='use training data')
    parser.add_argument('--data_dir', type=str, default='data/usegeo', help='data directory (use data/usegeo_poses for real poses)')
    args = parser.parse_args()

    print("loading model and data...")

    # load dataset config
    with open('datasets_location.json') as f:
        db_config = json.load(f)

    # create dataloader
    loader = get_loader('usegeo')

    records_path = f'{args.data_dir}/test_data'
    if args.use_train_data:
        records_path = f'{args.data_dir}/train_data'

    settings = DataloaderParameters(
        db_path_config=db_config,
        records_path=records_path,
        db_seq_len=4,
        seq_len=4,
        augment=False
    )

    loader.get_dataset('eval', settings, batch_size=1)
    dataset = loader.dataset

    # create model
    model = M4Depth(
        depth_type=loader.depth_type,
        nbre_levels=6,
        is_training=True
    )

    # build model
    for batch in dataset.take(1):
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        for key in ["depth", "RGB_im", "new_traj", "rot", "trans"]:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item
        _ = model([traj_samples, batch["camera"]], training=False)

    # load checkpoint
    checkpoint = tf.train.Checkpoint(model)
    weights_dir = os.path.join(args.ckpt_dir, "best")
    latest_ckpt = tf.train.latest_checkpoint(weights_dir)
    if latest_ckpt is None:
        weights_dir = os.path.join(args.ckpt_dir, "train")
        latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is not None:
        print(f"loading checkpoint: {latest_ckpt}")
        checkpoint.restore(latest_ckpt).expect_partial()
    else:
        print("warning: no checkpoint found, using random initialization")

    # generate visualizations
    visualize_samples(model, dataset, args.num_samples, args.output_dir, stride=args.stride)

    print(f"\ndone! {args.num_samples} visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
