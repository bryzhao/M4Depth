"""
dataloader for usegeo real-world drone depth dataset.
usegeo has rgb jpg images and tiff depth maps.
poses are computed from photogrammetry (omega, phi, kappa) and converted
to quaternion + relative translation format.
"""

import tensorflow as tf
from .generic import *


class DataLoaderUseGeo(DataLoaderGeneric):
    """
    dataloader for the usegeo dataset.
    key differences from midair:
    - depth is float32 tiff (direct meters) not 16-bit png disparity
    - image resolution is 1320x1989 (not 1024x1024)
    - poses from photogrammetry trajectory (quaternion + relative translation)
    """

    def __init__(self, out_size=[384, 384], crop=False):
        super(DataLoaderUseGeo, self).__init__('usegeo')
        self.in_size = [1320, 1989]  # usegeo native resolution
        self.depth_type = "map"  # direct depth values in meters

    def _set_output_size(self, out_size=[384, 384]):
        self.out_size = out_size
        self.long_edge = 0 if out_size[0] >= out_size[1] else 1
        if self.crop:
            self.intermediate_size = [out_size[self.long_edge], out_size[self.long_edge]]
        else:
            self.intermediate_size = out_size

        # camera intrinsics - using normalized values since usegeo doesn't provide intrinsics
        # these are approximate values assuming ~90 degree fov
        self.fx = 0.5 * self.intermediate_size[1]
        self.fy = 0.5 * self.intermediate_size[0]
        self.cx = 0.5 * self.intermediate_size[1]
        self.cy = 0.5 * self.intermediate_size[0]

    def get_dataset(self, usecase, settings, batch_size=3, out_size=[384, 384], crop=False):
        self.crop = crop
        if (usecase == "eval" or usecase == "predict") and self.crop:
            return AttributeError("crop option should be disabled when evaluating or predicting samples")
        super(DataLoaderUseGeo, self).get_dataset(usecase, settings, batch_size=batch_size, out_size=out_size)

    @tf.function
    def _decode_samples(self, data_sample):
        # load rgb image
        rgb_path = tf.strings.join([self.db_path, data_sample['camera_l']], separator='/')
        file = tf.io.read_file(rgb_path)
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32) / 255.

        # camera intrinsics
        camera_data = {
            "f": tf.convert_to_tensor([self.fx, self.fy]),
            "c": tf.convert_to_tensor([self.cx, self.cy]),
        }

        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(
            tf.image.resize(rgb_image, self.intermediate_size),
            self.intermediate_size + [3]
        )

        # pose data - real poses from photogrammetry trajectory
        out_data['rot'] = tf.cast(
            tf.stack([data_sample['qw'], data_sample['qx'], data_sample['qy'], data_sample['qz']], 0),
            dtype=tf.float32
        )
        # Real relative translation from frame-to-frame pose computation
        out_data['trans'] = tf.cast(
            tf.stack([data_sample['tx'], data_sample['ty'], data_sample['tz']], 0),
            dtype=tf.float32
        )
        # new_traj flag indicates trajectory start (from CSV)
        out_data['new_traj'] = tf.cast(data_sample['new_traj'], tf.bool)

        # load depth data if available
        if 'disp' in data_sample:
            depth_path = tf.strings.join([self.db_path, data_sample['disp']], separator='/')

            # usegeo depth is float32 tiff - decode using opencv through py_function
            # (tensorflow doesn't have native tiff support)
            depth = tf.py_function(self._load_tiff_depth, [depth_path], tf.float32)
            depth = tf.ensure_shape(depth, [None, None, 1])

            # resize to match output size
            out_data['depth'] = tf.reshape(
                tf.image.resize(depth, self.intermediate_size, method='nearest'),
                self.intermediate_size + [1]
            )

        return out_data

    @staticmethod
    def _load_tiff_depth(depth_path):
        """
        load tiff depth file using opencv.
        called as tf.py_function since tf doesn't support tiff natively.
        """
        import cv2
        import numpy as np

        # convert tensor to string
        path_str = depth_path.numpy().decode('utf-8')

        # read tiff as-is (float32)
        depth = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)

        if depth is None:
            # return zeros if file not found
            return np.zeros((1320, 1989, 1), dtype=np.float32)

        # ensure correct shape
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=-1)

        return depth.astype(np.float32)

    def _perform_augmentation(self):
        """
        perform data augmentation for usegeo.
        includes color augmentation. geometric transforms are possible now
        that we have real poses, but keeping simple for now.
        """
        # color augmentation
        self._augmentation_step_color()

        # crop if needed
        if self.crop:
            if self.long_edge == 0:
                diff = self.intermediate_size[1] - self.out_size[1]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(
                    self.out_data['RGB_im'],
                    [0, 0, offset, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 3]
                )
                self.out_data['depth'] = tf.slice(
                    self.out_data['depth'],
                    [0, 0, offset, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 1]
                )
                self.out_data['camera']['c'] = tf.convert_to_tensor([
                    self.out_data['camera']['c'][0] - tf.cast(offset, tf.float32),
                    self.out_data['camera']['c'][1]
                ])
            else:
                diff = self.intermediate_size[0] - self.out_size[0]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(
                    self.out_data['RGB_im'],
                    [0, offset, 0, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 3]
                )
                self.out_data['depth'] = tf.slice(
                    self.out_data['depth'],
                    [0, offset, 0, 0],
                    [self.seq_len, self.out_size[0], self.out_size[1], 1]
                )
                self.out_data['camera']['c'] = tf.convert_to_tensor([
                    self.out_data['camera']['c'][0],
                    self.out_data['camera']['c'][1] - tf.cast(offset, tf.float32)
                ])

            self.out_data['RGB_im'] = tf.reshape(
                self.out_data['RGB_im'],
                [self.seq_len, self.out_size[0], self.out_size[1], 3]
            )
            self.out_data['depth'] = tf.reshape(
                self.out_data['depth'],
                [self.seq_len, self.out_size[0], self.out_size[1], 1]
            )
