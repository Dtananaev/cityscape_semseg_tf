#
# Author: Denis Tananaev
# Date: 04.04.2020
#

import tensorflow as tf
from semseg_helpers import get_semseg_image, tinted_image
from metrics import compute_per_class_iou
import matplotlib.pyplot as plt
import io
import numpy as np
import itertools


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    buf.close()
    return image


def plot_confusion_matrix(cm, class_names, cmap="Oranges", figsize=(15, 15), dpi=100, title="Confusion matrix"):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Arguments:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        cmap: colormap (see matplotlib options)
        figsize: the size of figure
        dpi: pixels per 1 unit of size
        title: title of plot
    Returns:
        figure: confusion matrix figure to plot
    """
    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_class_distribution(
    class_distribution,
    class_names,
    color="tab:orange",
    figsize=(7, 7),
    dpi=100,
    text_size=7,
    title="Class distribution",
    show_mean=True,
):
    """
    Returns a matplotlib figure containing the plotted histograms.

    Arguments:
        class_distribution: list with class ditribution over different classes
        class_names: list with class names
        color: histogram color (see matplotlib colors)
        figsize: size of figure
        dpi: pixels per 1 unit of size
        title: title of plot
        show_mean: show mean as dotted red line
    Returns:
        figure: histograms to plot
    """
    y_pos = np.arange(len(class_names))
    class_distribution *= 100
    label = np.around(class_distribution, decimals=1).astype("str")
    figure = plt.figure(figsize=figsize, dpi=dpi)
    plt.bar(y_pos, class_distribution, color=color)
    plt.xticks(y_pos, class_names, rotation=90)
    for i in range(len(y_pos)):
        plt.text(x=y_pos[i] - 0.4, y=class_distribution[i], s=label[i], size=text_size)
    plt.ylabel("Value, %")
    plt.xlabel("Class")
    plt.title(title)
    if show_mean:
        mean = np.sum(class_distribution) / len(class_distribution)
        plt.axhline(mean, 0, 1, linestyle=":", color="tab:red", alpha=1.0, label="Mean: {:.1f}%".format(mean))
        plt.legend()

    return figure

def show_optimizer_info(optimizer, writer):
    with writer.as_default():
        tf.summary.scalar("beta_1", optimizer._get_hyper("beta_1"), step=optimizer.iterations)
        tf.summary.scalar("weight_decay", optimizer._get_hyper("weight_decay"), step=optimizer.iterations)
        tf.summary.scalar("learning_rate", optimizer._get_hyper("learning_rate")(optimizer.iterations), step=optimizer.iterations)


def train_summaries(train_out, optimizer, param_settings):
    """
    Visualizes  the train outputs in tensorboards
    """
    writer = tf.summary.create_file_writer(param_settings["train_summaries"])

    with writer.as_default():
        # Losses
        with tf.name_scope("Training losses"):
            tf.summary.scalar("1.Total loss", train_out["total_loss"], step=optimizer.iterations)
            tf.summary.scalar("2.Softmax loss", train_out["softmax_loss"], step=optimizer.iterations)
        with tf.name_scope("Optimizer info"):
            show_optimizer_info(optimizer, writer)

        # Show images
        if param_settings["step_summaries"] is not None and optimizer.iterations % param_settings["step_summaries"] == 0:
            input_images = train_out["images"] / 255.0
            # Show Inputs
            with tf.name_scope("1-Inputs"):
                tf.summary.image("Input images", input_images, step=optimizer.iterations)
            # Show GT semseg
            with tf.name_scope("2-Ground truth semantic segmentation"):
                semseg = tinted_image(
                    input_images.numpy(),
                    train_out["semseg"].numpy(),
                    param_settings["label_colors"],
                )
                tf.summary.image("2. input semseg", semseg, step=optimizer.iterations)
            # Show prediction semseg
            with tf.name_scope("2-Prediction"):
                predictions = get_semseg_image(train_out["predictions"])
                predictions = tinted_image(
                    input_images.numpy(),
                    predictions.numpy(),
                    param_settings["label_colors"],
                )
                tf.summary.image("3. Predicted semseg", predictions, step=optimizer.iterations)


def epoch_metrics_summaries(param_settings, epoch_metrics, epoch):
    """
    Visualizes epoch metrics
    """
    train_confusion_matrix = epoch_metrics.train_mIoU.total_cm
    val_confusion_matrix = epoch_metrics.val_mIoU.total_cm
    label_names = param_settings["label_names"]

    # Train results
    writer = tf.summary.create_file_writer(param_settings["train_summaries"])
    with writer.as_default():
        # Show epoch metrics for train
        with tf.name_scope("Epoch metrics"):
            tf.summary.scalar("1. Loss", epoch_metrics.train_loss.result().numpy(), step=epoch)
            tf.summary.scalar("2. MeanIoU", epoch_metrics.train_mIoU.result().numpy(), step=epoch)
            tf.summary.scalar("3. Accuracy", epoch_metrics.train_accuracy.result().numpy(), step=epoch)
        # Show per class metrics
        with tf.name_scope("Epoch metrics per class"):
            train_per_class_iou = compute_per_class_iou(train_confusion_matrix)
            for i in range(param_settings["num_classes"]):
                class_name = f"{i} {label_names[i]}"
                tf.summary.scalar(class_name, train_per_class_iou[i], step=epoch)
        # Show confusion matrix
        with tf.name_scope("Epoch metrics: Confusion matrix"):
            matrix = plot_confusion_matrix(train_confusion_matrix.numpy(), label_names, cmap="Oranges")
            tf.summary.image("Train Confusion Matrix", plot_to_image(matrix), step=epoch)

    # Val results
    writer = tf.summary.create_file_writer(param_settings["eval_summaries"])
    with writer.as_default():
        # Show epoch metrics for train
        with tf.name_scope("Epoch metrics"):
            tf.summary.scalar("1. Loss", epoch_metrics.val_loss.result().numpy(), step=epoch)
            tf.summary.scalar("2. MeanIoU", epoch_metrics.val_mIoU.result().numpy(), step=epoch)
            tf.summary.scalar("3. Accuracy", epoch_metrics.val_accuracy.result().numpy(), step=epoch)
        # Show per class metrics
        with tf.name_scope("Epoch metrics per class"):
            val_per_class_iou = compute_per_class_iou(val_confusion_matrix)
            for i in range(param_settings["num_classes"]):
                class_name = f"{i} {label_names[i]}"
                tf.summary.scalar(class_name, val_per_class_iou[i], step=epoch)
        # Show confusion matrix
        with tf.name_scope("Epoch metrics: Confusion matrix"):
            matrix = plot_confusion_matrix(val_confusion_matrix.numpy(), label_names, cmap="Blues")
            tf.summary.image("Val Confusion Matrix", plot_to_image(matrix), step=epoch)
