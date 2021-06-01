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
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json


def save_plot_to_image(file_to_save, figure):
    """
    Save matplotlib figure to image and close
    """
    plt.savefig(file_to_save)
    plt.close(figure)


def save_to_json(json_filename, dict_to_save):
    """
    Save to json file
    """
    with open(json_filename, "w") as f:
        json.dump(dict_to_save, f, indent=2)


def load_from_json(json_filename):
    """
    load from json file
    """
    with open(json_filename) as f:
        data = json.load(f)
        return data


def load_dataset_list(dataset_dir, dataset_file, delimiter=";"):
    """
    The function loads list of data from dataset
    file.
    Args:
     dataset_file: path to the .dataset file.
    Returns:
     dataset_list: list of data.
    """

    def add_path_prefix(item):
        """
        Add full path to the data entry
        """
        return os.path.join(dataset_dir, item)

    file_path = os.path.join(dataset_dir, dataset_file)
    dataset_list = []
    with open(file_path) as f:
        dataset_list = f.readlines()
    dataset_list = [x.strip().split(delimiter) for x in dataset_list]
    dataset_list = [list(map(add_path_prefix, x)) for x in dataset_list]

    return dataset_list


def save_dataset_list(dataset_file, data_list):
    """
    Saves dataset list to file.
    """
    with open(dataset_file, "w") as f:
        for item in data_list:
            f.write("%s\n" % item)
