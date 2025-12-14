"""
visualize model predictions vs ground truth depth maps.
helps debug whether the model is learning anything useful.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse

# import model and dataloader
from m4depth_network import M4Depth
from m4depth_options import M4DepthOptions
from dataloaders import get_loader


def visualize_samples(model, dataset, num_samples=10, output_dir="visualizations", stride=1):
    """
    generate side-by-side comparisons of rgb, predicted depth, and ground truth.
    stride: skip every N samples to spread across different trajectories/scenes
    """
    os.makedirs(output_dir, exist_ok=True)

    # iterate through dataset
    sample_count = 0
    iter_count = 0
    for batch in dataset:
        iter_count += 1
        # skip samples based on stride to spread across trajectories
        if stride > 1 and (iter_count - 1) % stride != 0:
            continue
        if sample_count >= num_samples:
            break

        # rearrange batch data into the format the model expects
        # model.call() expects [traj_samples, camera] where traj_samples is a list of frame dicts
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

        # extract data from batch
        rgb_images = batch["RGB_im"].numpy()  # shape: (batch, seq, h, w, 3)
        depth_gt = batch["depth"].numpy()      # shape: (batch, seq, h, w, 1)

        # get predicted depth (last frame in sequence)
        if isinstance(predictions, dict):
            depth_pred = predictions["depth"].numpy()
        else:
            # predictions might be nested list from training mode
            depth_pred = predictions[-1][0]["depth"].numpy()

        batch_size = rgb_images.shape[0]

        for b in range(batch_size):
            if sample_count >= num_samples:
                break

            # get last frame in sequence
            rgb = rgb_images[b, -1]  # last frame
            gt = depth_gt[b, -1, :, :, 0]

            # resize prediction to match gt if needed
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

            # ground truth depth (log scale for better visualization)
            gt_clipped = np.clip(gt, 1, 200)
            im1 = axes[1].imshow(np.log(gt_clipped), cmap='plasma')
            axes[1].set_title(f"Ground Truth Depth\nmin={gt.min():.1f}, max={gt.max():.1f}")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            # predicted depth (log scale)
            pred_clipped = np.clip(pred, 1, 200)
            im2 = axes[2].imshow(np.log(pred_clipped), cmap='plasma')
            axes[2].set_title(f"Predicted Depth\nmin={pred.min():.1f}, max={pred.max():.1f}")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            # error map
            # mask invalid depths
            valid_mask = (gt > 0) & (gt < 200)
            error = np.abs(pred - gt)
            error[~valid_mask] = 0

            im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=50)
            axes[3].set_title(f"Absolute Error\nmean={error[valid_mask].mean():.2f}")
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

            plt.tight_layout()

            # save figure
            save_path = os.path.join(output_dir, f"sample_{sample_count:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"saved: {save_path}")
            sample_count += 1

    print(f"\n{sample_count} visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize depth predictions')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='output directory for images')
    parser.add_argument('--use_train_data', action='store_true',
                        help='visualize on training data instead of test')
    parser.add_argument('--stride', type=int, default=1,
                        help='skip every N samples to spread across trajectories (e.g. 100 to sample different scenes)')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle dataset before sampling (mixes environments)')
    parser.add_argument('--environment', type=str, default=None,
                        help='filter to specific environment (e.g. Kite_training/cloudy, PLE_training/fall)')

    # add m4depth options to parser (includes --ckpt_dir, --records_path, etc)
    model_opts = M4DepthOptions(parser)
    args, _ = parser.parse_known_args()

    print("loading model and data...")

    # get dataloader
    chosen_dataloader = get_loader("midair")

    # configure dataloader settings
    records_path = args.records_path if args.records_path else 'data/midair/test_data'
    if args.use_train_data:
        records_path = 'data/midair/train_data'

    # filter to specific environment if requested
    if args.environment:
        records_path = os.path.join(records_path, args.environment)
        print(f"filtering to environment: {args.environment}")

    model_opts.dataloader_settings = model_opts.dataloader_settings._replace(
        records_path=records_path,
        db_seq_len=4
    )

    # load dataset
    chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
    dataset = chosen_dataloader.dataset

    # apply shuffle if requested (mixes samples from different trajectories/environments)
    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    # create model with 6 levels to match paper architecture
    # use is_training=True to match training model architecture for checkpoint loading
    model = M4Depth(
        depth_type=chosen_dataloader.depth_type,
        nbre_levels=6,
        is_training=True
    )

    # checkpoint directory
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else './ckpt'

    # build model first by running one forward pass with properly formatted data
    for batch in dataset.take(1):
        seq_len = batch["depth"].shape[1]
        traj_samples = [{} for _ in range(seq_len)]
        attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
        for key in attribute_list:
            value_list = tf.unstack(batch[key], axis=1)
            for i, item in enumerate(value_list):
                traj_samples[i][key] = item
        camera = batch["camera"]
        _ = model([traj_samples, camera], training=False)

    # load weights using tf.train.Checkpoint (same as training restore)
    # try best/ first, fall back to train/
    checkpoint = tf.train.Checkpoint(model)
    weights_dir = os.path.join(ckpt_dir, "best")
    latest_ckpt = tf.train.latest_checkpoint(weights_dir)
    if latest_ckpt is None:
        weights_dir = os.path.join(ckpt_dir, "train")
        latest_ckpt = tf.train.latest_checkpoint(weights_dir)

    if latest_ckpt is not None:
        print(f"loading checkpoint from: {latest_ckpt}")
        checkpoint.restore(latest_ckpt).expect_partial()
    else:
        print(f"warning: no checkpoint found, using random initialization")

    # generate visualizations
    visualize_samples(model, dataset, args.num_samples, args.output_dir, stride=args.stride)

    print(f"\ndone! {args.num_samples} visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
