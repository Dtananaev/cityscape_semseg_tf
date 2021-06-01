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
from io_file import save_to_json
import os
import numpy as np


def confusion_matrix_statistics(confusion_matrix):
    """
    The function computes statistics over confusion matrix
    """

    true_positives = tf.cast(tf.linalg.diag_part(confusion_matrix), dtype=tf.float32)

    sum_over_col = tf.cast(tf.reduce_sum(confusion_matrix, axis=1), dtype=tf.float32)
    false_positive = sum_over_col - true_positives

    sum_over_row = tf.cast(tf.reduce_sum(confusion_matrix, axis=0), dtype=tf.float32)
    false_negative = sum_over_row - true_positives

    return true_positives, false_positive, false_negative


def compute_per_class_iou(confusion_matrix):
    """
    The function computes the intersection-over-union via the confusion matrix.
    """

    # IoU = true_positives / (true_positives + false_positive + false_negative)
    true_positives, false_positive, false_negative = confusion_matrix_statistics(confusion_matrix)
    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = true_positives + false_positive + false_negative
    per_class_iou = tf.math.divide_no_nan(true_positives, denominator)

    return per_class_iou


class EpochMetrics:
    """
    The class computes the mean IoU, accuracy and loss
    for train and validation step
    Arguments:
        number_classes: the number of semseg classes
    """
    def __init__(self, number_classes):
        # class_names: numpy array where class_names[label_id] = "<label_name>"
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.train_mIoU = tf.keras.metrics.MeanIoU(num_classes=number_classes, name="train_mIoU")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
        self.val_mIoU = tf.keras.metrics.MeanIoU(num_classes=number_classes, name="val_mIoU")

    def reset(self):
        """
        Reset all metrics to zero (need to do each epoch)
        """
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        self.train_mIoU.reset_states()
        self.val_mIoU.reset_states()

    def save_to_json(self, dir_to_save):
        """
        Save all metrics to the json file
        """

        # Check that folder is exitsts or create it
        os.makedirs(dir_to_save, exist_ok=True)
        json_filename = os.path.join(dir_to_save, "epoch_metrics.json")
        # fill the dict
        metrics_dict = {
            "train_mIoU": str(self.train_mIoU.result().numpy()),
            "val_mIoU": str(self.val_mIoU.result().numpy()),
            "train_loss": str(self.train_loss.result().numpy()),
            "val_loss": str(self.val_loss.result().numpy()),
            "train_accuracy": str(self.train_accuracy.result().numpy()),
            "val_accuracy": str(self.val_accuracy.result().numpy()),
            "train_confusion_matrix": self.train_mIoU.total_cm.numpy().tolist(),
            "val_confusion_matrix": self.val_mIoU.total_cm.numpy().tolist(),
        }
        save_to_json(json_filename, metrics_dict)

    def print_metrics(self):
        """
        Print all metrics
        """
        train_loss = np.around(self.train_loss.result().numpy(), decimals=2)
        val_loss = np.around(self.val_loss.result().numpy(), decimals=2)
        train_mIoU = np.around(self.train_mIoU.result().numpy() * 100, decimals=2)
        val_mIoU = np.around(self.val_mIoU.result().numpy() * 100, decimals=2)
        train_accuracy = np.around(self.train_accuracy.result().numpy() * 100, decimals=2)
        val_accuracy = np.around(self.val_accuracy.result().numpy() * 100, decimals=2)

        template = "train_loss {}, val_loss {}".format(train_loss, val_loss)
        template += "\ntrain_mIoU: {} %, val_mIoU: {} %".format(train_mIoU, val_mIoU)
        template += "\ntrain_accuracy: {} %, val_accuracy: {} %".format(train_accuracy, val_accuracy)
        print(template)
