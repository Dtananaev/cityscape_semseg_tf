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
from train import train_step
from parameters import Parameters
from semseg_dataset import SemsegDataset
from inception_v1 import InceptionV1Model
from training_tools import initialize_model
from summary_helpers import train_summaries
from tqdm import tqdm
import tensorflow_addons as tfa


def get_lr_finder_scheduler(min_lr, max_lr, max_epochs, iter_per_epoch):
    """
    lr = initial_learning_rate * decay_rate ^ (step / decay_steps)
    """
    scheduler = {
        "name": "exponential_scheduler",
        "initial_learning_rate": min_lr,
        "decay_steps": max_epochs * iter_per_epoch,  # Number epochs
        "decay_rate": max_lr / min_lr,  # increase lr
        "staircase": False,  # Decay lr at discrete intervals
    }
    scheduler_type = tf.keras.optimizers.schedules.ExponentialDecay

    return scheduler_type(**scheduler)


def lr_finder():
    """
    The function searches for the maximum learning rate,
    which can be used for cycling schedulers like triangular lr policy
    or warm restards.
    Usual otpimal lr = 0.5 * max_lr;
    Usual minimal lr = 0.1 * max_lr;
    In order to get best max lr check lr for the smallest loss in tensorboard.
    """
    param = Parameters()
    max_epochs = 1
    param.settings["training_summaries"] = "log/lr_finder"
    param.settings["step_summaries"] = None  # Turn off image summarues



    # Init label colors and label names
    tf.random.set_seed(param.settings["seed"])

    train_dataset = SemsegDataset(param.settings, "train.dataset", shuffle=True)
    param.settings["train_size"] = train_dataset.num_samples
    # Init model
    model = InceptionV1Model(weight_decay=param.settings["weight_decay"], num_classes=param.settings["num_classes"])
    input_shape = [1, param.settings["data_height"], param.settings["data_width"], 3]
    initialize_model(model, input_shape)
    lr_scheduler = get_lr_finder_scheduler(min_lr=1e-6, max_lr=0.1, max_epochs=max_epochs, iter_per_epoch=train_dataset.num_it_per_epoch)
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=param.settings["weight_decay"])

    # Semseg loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for epoch in range(0, max_epochs):
        for train_samples in tqdm(train_dataset.dataset, desc=f"Lr finder epoch {epoch}", total=train_dataset.num_it_per_epoch):
            train_outputs = train_step(param.settings, train_samples, model, loss_object, optimizer)
            train_summaries(train_outputs, optimizer, param.settings)


if __name__ == "__main__":
    lr_finder()
