"""
Generate UseGeo CSV files with REAL camera poses.
Converts absolute UTM coordinates to relative frame-to-frame poses.
Marks trajectory breaks where drone jumps between flight strips.
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os

def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    """Inverse of quaternion (w, x, y, z format)."""
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    return np.array([w, -x, -y, -z]) / norm_sq

def compute_relative_pose(q1, t1, q2, t2):
    """
    Compute relative pose from frame 1 to frame 2.
    Returns relative rotation (quaternion) and translation.
    """
    # Relative rotation: q_rel = q1^-1 * q2
    q1_inv = quaternion_inverse(q1)
    q_rel = quaternion_multiply(q1_inv, q2)

    # Relative translation in frame 1's coordinate system
    # t_rel = R1^-1 * (t2 - t1)
    t_diff = t2 - t1
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])  # scipy uses x,y,z,w
    t_rel = r1.inv().apply(t_diff)

    return q_rel, t_rel

def process_dataset(traj_path, dataset_name):
    """Process a single dataset trajectory file."""
    df = pd.read_csv(traj_path, sep='\t')
    print(f"  {dataset_name}: {len(df)} frames")

    # Compute frame-to-frame distances to find trajectory breaks
    dtx = df.tx.diff()
    dty = df.ty.diff()
    dtz = df.tz.diff()
    dist = np.sqrt(dtx**2 + dty**2 + dtz**2)

    # Mark trajectory breaks (distance > 50m indicates new flight strip)
    BREAK_THRESHOLD = 50.0
    is_break = dist > BREAK_THRESHOLD
    is_break.iloc[0] = True  # First frame is always a trajectory start

    # Compute relative poses
    rel_qw, rel_qx, rel_qy, rel_qz = [], [], [], []
    rel_tx, rel_ty, rel_tz = [], [], []
    new_traj_flags = []

    for i in range(len(df)):
        if i == 0 or is_break.iloc[i]:
            # First frame of trajectory - identity rotation, small forward translation
            # Use non-zero translation to avoid division by zero in parallax computation
            rel_qw.append(1.0)
            rel_qx.append(0.0)
            rel_qy.append(0.0)
            rel_qz.append(0.0)
            rel_tx.append(0.0)
            rel_ty.append(0.0)
            rel_tz.append(0.5)  # Small forward translation to prevent NaN
            new_traj_flags.append(1)
        else:
            # Compute relative pose from previous frame
            q1 = np.array([df.iloc[i-1].qw, df.iloc[i-1].qx, df.iloc[i-1].qy, df.iloc[i-1].qz])
            t1 = np.array([df.iloc[i-1].tx, df.iloc[i-1].ty, df.iloc[i-1].tz])
            q2 = np.array([df.iloc[i].qw, df.iloc[i].qx, df.iloc[i].qy, df.iloc[i].qz])
            t2 = np.array([df.iloc[i].tx, df.iloc[i].ty, df.iloc[i].tz])

            q_rel, t_rel = compute_relative_pose(q1, t1, q2, t2)

            rel_qw.append(q_rel[0])
            rel_qx.append(q_rel[1])
            rel_qy.append(q_rel[2])
            rel_qz.append(q_rel[3])
            rel_tx.append(t_rel[0])
            rel_ty.append(t_rel[1])
            rel_tz.append(t_rel[2])
            new_traj_flags.append(0)

    # Create output dataframe with dataset prefix in paths
    output_df = pd.DataFrame({
        'id': range(len(df)),
        'camera_l': dataset_name + '/' + df['camera_l'],
        'disp': dataset_name + '/' + df['disp'],
        'qw': rel_qw,
        'qx': rel_qx,
        'qy': rel_qy,
        'qz': rel_qz,
        'tx': rel_tx,
        'ty': rel_ty,
        'tz': rel_tz,
        'new_traj': new_traj_flags
    })

    return output_df


def main():
    # Process all 3 datasets
    print("Loading trajectory files...")
    datasets = [
        ('/home/bryan/dev/final_project/UseGeo/dataset1_trajectory.csv', 'Dataset-1'),
        ('/home/bryan/dev/final_project/UseGeo/dataset2_trajectory.csv', 'Dataset-2'),
        ('/home/bryan/dev/final_project/UseGeo/dataset3_trajectory.csv', 'Dataset-3'),
    ]

    all_dfs = []
    for traj_path, dataset_name in datasets:
        df = process_dataset(traj_path, dataset_name)
        all_dfs.append(df)

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['id'] = range(len(combined_df))
    print(f"\nTotal frames: {len(combined_df)}")

    # Assign trajectory IDs based on new_traj flags
    combined_df['traj_id'] = combined_df['new_traj'].cumsum() - 1
    n_trajs = combined_df['traj_id'].max() + 1
    print(f"Total trajectories: {n_trajs}")

    # Print statistics
    non_break_df = combined_df[combined_df['new_traj'] == 0]
    rel_dist = np.sqrt(non_break_df['tx']**2 + non_break_df['ty']**2 + non_break_df['tz']**2)
    print(f"\nRelative translation stats (excluding trajectory starts):")
    print(f"  Mean: {rel_dist.mean():.2f}m")
    print(f"  Std:  {rel_dist.std():.2f}m")
    print(f"  Min:  {rel_dist.min():.2f}m")
    print(f"  Max:  {rel_dist.max():.2f}m")

    # Split into train/val (80/20 by trajectory)
    unique_trajs = combined_df.traj_id.unique()
    np.random.seed(42)
    np.random.shuffle(unique_trajs)

    n_train_trajs = int(len(unique_trajs) * 0.8)
    train_trajs = unique_trajs[:n_train_trajs]
    val_trajs = unique_trajs[n_train_trajs:]

    train_df = combined_df[combined_df.traj_id.isin(train_trajs)].copy()
    val_df = combined_df[combined_df.traj_id.isin(val_trajs)].copy()

    # Reset IDs within each split
    train_df['id'] = range(len(train_df))
    val_df['id'] = range(len(val_df))

    # Remove traj_id column (not needed by dataloader)
    train_df = train_df.drop('traj_id', axis=1)
    val_df = val_df.drop('traj_id', axis=1)

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_df)} frames ({len(train_trajs)} trajectories)")
    print(f"  Val:   {len(val_df)} frames ({len(val_trajs)} trajectories)")

    # Save CSVs
    os.makedirs('data/usegeo_poses/train_data', exist_ok=True)
    os.makedirs('data/usegeo_poses/test_data', exist_ok=True)

    train_df.to_csv('data/usegeo_poses/train_data/usegeo_train.csv', index=False, sep='\t')
    val_df.to_csv('data/usegeo_poses/test_data/usegeo_val.csv', index=False, sep='\t')

    print(f"\nSaved:")
    print(f"  data/usegeo_poses/train_data/usegeo_train.csv")
    print(f"  data/usegeo_poses/test_data/usegeo_val.csv")

    # Show sample rows
    print(f"\nSample train data (first 3 rows):")
    print(train_df.head(3).to_string())

if __name__ == '__main__':
    main()
