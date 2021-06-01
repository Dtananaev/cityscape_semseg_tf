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

from io_file import save_to_json
from cityscape_helpers import (
    get_label_name,
    get_color_palette,
)
import os


class Parameters(object):
    """
    The class contains experiment parameters.
    """

    def __init__(self):


        self.settings = {
            # The directory for checkpoints
            "dataset_dir": "dataset",
            "data_height": 256,
            "data_width": 512,
            "batch_size": 16,
            # The checkpoint related
            "checkpoints_dir": "log/checkpoints",
            "train_summaries": "log/summaries/train",
            "eval_summaries": "log/summaries/val",
            # Update tensorboard train images each step_summaries iterations
            "step_summaries": None,  # to turn off make it None
            # General settings
            "seed": 2021,
            "max_epochs": 1000,
            "weight_decay": 1e-4,
            # Semseg related
            "num_classes": 20,  # 19 + 1 background
        }

        # Set special parameters
        self.settings["optimizer"] = "adamw"
        self.settings["scheduler"] =  "one_cycle"
        # Semseg related
        self.settings["label_colors"] = get_color_palette()  # python  list label_colors[label_id] = [r,g,b]
        self.settings["label_names"] = get_label_name()  # python  list where label_names[label_id] = "<label_name>"

        # Automatically defined during training parameters
        self.settings["train_size"] = None  # the size of train set
        self.settings["val_size"] = None  # the size of val set


    def save_to_json(self, dir_to_save):
        """
        Save parameters to .json
        """
        # Check that folder is exitsts or create it
        os.makedirs(dir_to_save, exist_ok=True)
        json_filename = os.path.join(dir_to_save, "parameters.json")
        save_to_json(json_filename, self.settings)