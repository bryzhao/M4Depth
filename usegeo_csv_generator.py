"""
generate csv files for usegeo dataset in the same format as midair.
since usegeo doesn't have camera pose data, we use identity rotation (quaternion [1,0,0,0])
and zero translation. this allows the model to do single-frame depth estimation.
"""

import os
import argparse

def get_matching_pairs(usegeo_path, dataset_name):
    """
    find all rgb images that have corresponding depth maps.
    returns list of (rgb_relative_path, depth_relative_path) tuples.
    """
    images_dir = os.path.join(usegeo_path, dataset_name, "undistorted_images")
    depth_dir = os.path.join(usegeo_path, dataset_name, "depth_maps")

    if not os.path.isdir(images_dir) or not os.path.isdir(depth_dir):
        return []

    # get all depth map basenames (without _depth_res.tiff suffix)
    depth_files = set()
    for f in os.listdir(depth_dir):
        if f.endswith("_depth_res.tiff"):
            # extract base: 2021-04-23_13-17-12_S2223314_DxO from 2021-04-23_13-17-12_S2223314_DxO_depth_res.tiff
            base = f.replace("_depth_res.tiff", "")
            depth_files.add(base)

    # find matching images
    pairs = []
    for f in sorted(os.listdir(images_dir)):
        if f.endswith("_res.jpg"):
            # extract base: 2021-04-23_13-17-12_S2223314_DxO from 2021-04-23_13-17-12_S2223314_DxO_res.jpg
            base = f.replace("_res.jpg", "")
            if base in depth_files:
                rgb_rel = os.path.join(dataset_name, "undistorted_images", f)
                depth_rel = os.path.join(dataset_name, "depth_maps", base + "_depth_res.tiff")
                pairs.append((rgb_rel, depth_rel))

    return pairs


def write_csv(pairs, output_path, train_ratio=0.8):
    """
    write csv files for train and test splits.
    uses identity pose (no motion) since usegeo has no pose data.
    """
    # split into train/test
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    # identity quaternion (w, x, y, z) and zero translation
    identity_quat = (1.0, 0.0, 0.0, 0.0)
    zero_trans = (0.0, 0.0, 0.0)

    for split_name, split_pairs in [("train_data", train_pairs), ("test_data", test_pairs)]:
        if not split_pairs:
            continue

        split_dir = os.path.join(output_path, split_name, "UseGeo")
        os.makedirs(split_dir, exist_ok=True)

        csv_path = os.path.join(split_dir, "traj_0000.csv")

        with open(csv_path, 'w') as f:
            # write header (same format as midair)
            f.write("id\tcamera_l\tdisp\tqw\tqx\tqy\tqz\ttx\tty\ttz\n")

            for idx, (rgb_path, depth_path) in enumerate(split_pairs):
                qw, qx, qy, qz = identity_quat
                tx, ty, tz = zero_trans
                f.write(f"{idx}\t{rgb_path}\t{depth_path}\t{qw:.6f}\t{qx:.6f}\t{qy:.6f}\t{qz:.6f}\t{tx:.6f}\t{ty:.6f}\t{tz:.6f}\n")

        print(f"wrote {len(split_pairs)} samples to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate CSV files for UseGeo dataset')
    parser.add_argument("--usegeo_path", default="/Users/bryanzhao/development/code/wpi/rbe_577/machine_learning_robotics/final_project/UseGeo",
                        help="path to UseGeo dataset root")
    parser.add_argument("--output_dir", default="data/usegeo",
                        help="output directory for csv files")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="ratio of data to use for training (default: 0.8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_pairs = []

    # process each dataset
    for dataset_name in ["Dataset-1", "Dataset-2", "Dataset-3"]:
        pairs = get_matching_pairs(args.usegeo_path, dataset_name)
        print(f"{dataset_name}: found {len(pairs)} image-depth pairs")
        total_pairs.extend(pairs)

    print(f"\ntotal: {len(total_pairs)} pairs")

    if total_pairs:
        write_csv(total_pairs, args.output_dir, args.train_ratio)
        print(f"\ndone! csv files written to {args.output_dir}/")
    else:
        print("error: no matching image-depth pairs found")


if __name__ == "__main__":
    main()
