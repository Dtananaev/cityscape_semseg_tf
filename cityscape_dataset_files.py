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

import os
import glob
import numpy as np
import argparse
from io_file import save_dataset_list


class CreateDatasetCityScape:
    """
    The class to create .dataset filelist for CityScape
    Arguments:
        dataset_dir: path to CityScape dataset
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    
    def remove_prefix(self, x):
        return x[len(self.dataset_dir)+1:]

    def get_data_pair(self, split="train"):
        """
        Creates the list of image semseg pairs.
        Arguments:
            split: train, val, test split
        Returns:
            pair: list of image semseg pairs
        """
        labels_string = os.path.join(self.dataset_dir, "*", "gtFine", split, "*", "*_labelTrainIds.png")
        img_string = os.path.join(self.dataset_dir, "*", "leftImg8bit", split, "*", "*_leftImg8bit.png")

        # Get list of labels
        lbl_list = sorted(glob.glob(labels_string))
        lbl_list = np.asarray([self.remove_prefix(x) for x in lbl_list])
        # Get list of images
        img_list = sorted(glob.glob(img_string))
        img_list = np.asarray([self.remove_prefix(x) for x in img_list])

        pair = np.concatenate((img_list[:, None], lbl_list[:, None]), axis=1)
        pair = [";".join(x) for x in pair]
        return pair

    def create_datasets_file(self, split="train"):
        """
        Creates  <split>.dataset file
        """
        filename = os.path.join(self.dataset_dir, split + ".dataset")
        data_list = self.get_data_pair(split)
        save_dataset_list(filename, data_list)
        print(f"The dataset of the size {len(data_list)} saved in {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetFileCreator.")
    parser.add_argument("--dataset_dir", type=str, help="creates .dataset file", default="dataset")
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    data_creator = CreateDatasetCityScape(args.dataset_dir)
    for split in splits:
        data_creator.create_datasets_file(split)
