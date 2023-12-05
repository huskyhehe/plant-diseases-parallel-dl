import numpy as np
import matplotlib.pyplot as plt


def plot_time_and_speedup(mode: str, num_list, time_list, speedup_list) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(num_list, time_list, marker="o", label="Elapsed Time")
    plt.title("Elapsed Time")
    plt.xlabel("Number of Workers")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.xticks(num_list, [int(n) for n in num_list])

    plt.subplot(1, 2, 2)
    plt.plot(num_list, speedup_list, marker="o", label="Speedup")
    plt.title("Speedup")
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.xticks(num_list, [int(n) for n in num_list])

    plt.tight_layout()
    plt.gcf().set_dpi(600)
    plt.savefig(f"res/{mode}_effi_time_speedup.png", dpi=600)
    plt.show()


def plot_gpu_usage(mode: str, num_list, max_alloc_list) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.bar(num_list, max_alloc_list, label="Max GPU Allocated Memory (MB)")
    plt.title("Max GPU Allocated Memory")
    plt.xlabel("Number of Workers")
    plt.ylabel("GPU Memory Usage (MB)")
    plt.grid(True)
    plt.xticks(num_list, [int(n) for n in num_list])

    plt.tight_layout()
    plt.gcf().set_dpi(600)
    plt.savefig(f"res/{mode}_effi_gpu_mem.png", dpi=600)
    plt.show()