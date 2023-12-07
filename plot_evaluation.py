import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

        
# This is the function to plot training, validation loss and accuracy
def plot_loss_and_acc(trainer_name, epoch_list, history):
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, history["train_loss"], label="Train Loss")
    plt.plot(epoch_list, history["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    # Set x-axis ticks without decimal points
    plt.xticks(epoch_list, [int(epoch) for epoch in epoch_list])

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, np.array([tensor.item() for tensor in history["train_acc"]], dtype=np.float64), label="Train Accuracy")
    plt.plot(epoch_list, np.array([tensor.item() for tensor in history["valid_acc"]], dtype=np.float64), label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    # Set x-axis ticks without decimal points
    plt.xticks(epoch_list, [int(epoch) for epoch in epoch_list])

    plt.tight_layout()
    plt.savefig(f"res/{trainer_name}__acc_and_loss.png")
    plt.show()


# This is the function to plot time and memory usage
def plot_time_and_memory_usage(trainer_name, epoch_list, history):
    plt.figure(figsize=(12, 5))

    # Plot epoch elapsed time
    plt.subplot(1, 2, 1)
    plt.bar(epoch_list, history["epo_elapsed_time"], label="Elapsed Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Elapsed Time (s)")
    plt.legend()
    plt.title("Epoch Elapsed Time")
    plt.xticks(epoch_list, [int(epoch) for epoch in epoch_list])

    # Plot memory usage
    plt.subplot(1, 2, 2)
    plt.bar(epoch_list, history["max_alloc"], label="Max GPU Allocated Memory (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("GPU Memory (MB)")
    plt.legend()
    plt.title("GPU Memory Allocated Memory")
    plt.xticks(epoch_list, [int(epoch) for epoch in epoch_list])

    plt.tight_layout()
    plt.gcf().set_dpi(600)
    plt.savefig(f"res/{trainer_name}__time_and_memory_usage.png", dpi=600)
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.gcf().set_dpi(600)
    plt.savefig(f"res/confusion_matrix—_{time.time()}.png", dpi=600)
    plt.show()