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
from parameters import Parameters
from semseg_dataset import SemsegDataset
from semseg_helpers import get_semseg_image
from metrics import EpochMetrics
from summary_helpers import train_summaries, epoch_metrics_summaries
from inception_v1 import InceptionV1Model
from training_tools import load_model, initialize_model, setup_gpu
from tqdm import tqdm
import argparse
import os
import tensorflow_addons as tfa
from one_cycle_tf import OneCycle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure train runs on gpu 0


def get_optimizer_lr(wd, num_iter_per_epoch):
    max_lr = 4.5e-4
    cycle_size = 40 * num_iter_per_epoch
    shift_peak = 0.1
    final_lr_scale = 1.0
    min_wd = 0.0
    max_wd = wd
    min_momentum = 0.85
    max_momentum = 0.95

    lr_scheduler = OneCycle(initial_learning_rate=max_lr/25.0,
                            maximal_learning_rate=max_lr,
                            cycle_size=cycle_size, 
                            shift_peak=shift_peak,
                            final_lr_scale=final_lr_scale
                            )
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=max_wd)
    momentum_scheduler = OneCycle(initial_learning_rate=max_momentum,
                                maximal_learning_rate=min_momentum,
                                cycle_size=cycle_size,
                                shift_peak=shift_peak,
                                final_lr_scale=final_lr_scale
                                )
    wd_scheduler = OneCycle(initial_learning_rate=max_wd,
                            maximal_learning_rate=min_wd,
                            cycle_size=cycle_size, 
                            shift_peak=shift_peak,
                            final_lr_scale=final_lr_scale
                        )
    optimizer._set_hyper("beta_1", lambda: momentum_scheduler(optimizer.iterations))
    optimizer._set_hyper("weight_decay", lambda: wd_scheduler(optimizer.iterations))
    return lr_scheduler, optimizer


@tf.function
def train_step(param_settings, samples, model, loss_object, optimizer, epoch_metrics=None):

    with tf.GradientTape() as tape:
        images, semseg = samples
        # The training entry needed for the layers with different
        # behaviour in train/test time like batchnorm, dropout etc.
        predictions = model(images, training=True)

        # CityScape balancing dataset based on distribution of pixels labels in %
        semseg = tf.squeeze(tf.cast(semseg, tf.int32), axis=-1)

        softmax_loss = loss_object(semseg, predictions)
        total_loss = softmax_loss

        # Get L2 losses for weight decay
        # For all optimizers without decoupled weight decay
        # For adamW and FTRL this is not needed
        decoupled_wd = ["adamw"]
        if param_settings["optimizer"].lower() not in decoupled_wd:
            total_loss += tf.add_n(model.losses)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch_metrics is not None:
        epoch_metrics.train_loss(softmax_loss)
        epoch_metrics.train_accuracy(semseg, predictions)
        epoch_metrics.train_mIoU.update_state(semseg, get_semseg_image(predictions))
    train_outputs = {
        "total_loss": total_loss,
        "softmax_loss": softmax_loss,
        "images": images,
        "semseg": semseg,
        "predictions": predictions,
    }

    return train_outputs


@tf.function
def val_step(samples, model, loss_object, epoch_metrics=None):

    images, semseg = samples
    predictions = model(images, training=False)
    loss = loss_object(semseg, predictions)
    if epoch_metrics is not None:
        epoch_metrics.val_loss(loss)
        epoch_metrics.val_accuracy(semseg, predictions)
        epoch_metrics.val_mIoU.update_state(semseg, get_semseg_image(predictions))


def train(resume=False):
    setup_gpu()
    # General parameters
    param = Parameters()

    # Init label colors and label names
    tf.random.set_seed(param.settings["seed"])

    train_dataset = SemsegDataset(
        param.settings,
        "train.dataset",
        shuffle=True,
    )
    param.settings["train_size"] = train_dataset.num_samples
    val_dataset = SemsegDataset(param.settings, "val.dataset", shuffle=False)
    param.settings["val_size"] = val_dataset.num_samples

    # Init model
    model = InceptionV1Model(
        weight_decay=param.settings["weight_decay"],
        num_classes=param.settings["num_classes"],
    )
    input_shape = [1, param.settings["data_height"], param.settings["data_width"], 3]
    initialize_model(model, input_shape)
    model.summary()
    start_epoch, model = load_model(param.settings["checkpoints_dir"], model, resume)
    model_path = os.path.join(param.settings["checkpoints_dir"], "{model}-{epoch:04d}")
    
    #optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=param.settings["weight_decay"])
    lr, optimizer = get_optimizer_lr(param.settings["weight_decay"], train_dataset.num_it_per_epoch)
    epoch_metrics = EpochMetrics(param.settings["num_classes"])

    # Semseg loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(start_epoch, param.settings["max_epochs"]):
        save_dir = model_path.format(model=model.name, epoch=epoch)
        epoch_metrics.reset()
        for train_samples in tqdm(
            train_dataset.dataset,
            desc=f"Epoch {epoch}",
            total=train_dataset.num_it_per_epoch,
        ):
            train_outputs = train_step(
                param.settings,
                train_samples,
                model,
                loss_object,
                optimizer,
                epoch_metrics,
            )
            train_summaries(train_outputs, optimizer, param.settings)
        for val_samples in tqdm(val_dataset.dataset, desc="Validation", total=val_dataset.num_it_per_epoch):
            val_step(val_samples, model, loss_object, epoch_metrics)
        epoch_metrics_summaries(param.settings, epoch_metrics, epoch)
        epoch_metrics.print_metrics()
        # Save all
        param.save_to_json(save_dir)
        epoch_metrics.save_to_json(save_dir)
        model.save(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN.")
    parser.add_argument(
        "--resume",
        type=lambda x: x,
        nargs="?",
        const=True,
        default=False,
        help="Activate nice mode.",
    )
    args = parser.parse_args()
    train(resume=args.resume)
