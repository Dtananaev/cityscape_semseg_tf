#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2021 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
from io_file import load_dataset_list
from parameters import Parameters
from tqdm import tqdm


class SemsegDataset:
    """
    This is dataset layer for semseg experiment
    Arguments:
        param_settings: parameters of experiment
        dataset_file: name of .dataset file
        augmentation: apply augmentation True/False
        shuffle: shuffle the data True/False
    """

    def __init__(self, param_settings, dataset_file, shuffle=False):
        # Private methods
        self.seed = param_settings["seed"]

        self.param_settings = param_settings
        self.dataset_file = dataset_file
        self.inputs_list = load_dataset_list(self.param_settings["dataset_dir"], dataset_file)
        self.num_samples = len(self.inputs_list)
        self.num_it_per_epoch = int(self.num_samples / self.param_settings["batch_size"])
        self.output_types = [tf.float32, tf.float32]

        ds = tf.data.Dataset.from_tensor_slices(self.inputs_list)

        if shuffle:
            ds = ds.shuffle(self.num_samples)
        ds = ds.map(
            map_func=lambda x: tf.py_function(self.load_image_semseg, [x], Tout=self.output_types),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.batch(self.param_settings["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = ds

    def load_image_semseg(self, data_input):
        """
        Loads image and semseg and resizes it
        Note: This is numpy function.
        """
        image_file, semseg_file = np.asarray(data_input).astype("U")
        img = np.asarray(
            Image.open(image_file).resize((self.param_settings["data_width"], self.param_settings["data_height"]), Image.NEAREST),
            dtype=np.float32,
        )

        semseg = np.asarray(
            Image.open(semseg_file).resize((self.param_settings["data_width"], self.param_settings["data_height"]), Image.NEAREST),
            dtype=np.float32,
        )
        semseg = np.expand_dims(semseg, axis=-1)
        mask = semseg >= self.param_settings["num_classes"] - 1  # subtract backgound class
        semseg[mask] = 19.0  # Map all labels which is not trainig for to the background class

        return img, semseg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetLayer.")
    parser.add_argument("--dataset_file", type=str, help="creates .dataset file", default="train.datalist")
    args = parser.parse_args()

    param_settings = Parameters().settings
    train_dataset = SemsegDataset(param_settings, args.dataset_file)

    for samples in tqdm(train_dataset.dataset, total=train_dataset.num_it_per_epoch):
        image, semseg = samples
        print("image {}, semseg {}".format(image.shape, semseg.shape))
