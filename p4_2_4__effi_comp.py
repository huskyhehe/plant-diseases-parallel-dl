import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from plot_comparison import plot_time_and_speedup, plot_gpu_usage


# This function is to load pickle file
def load_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
        return loaded_data


def workers_comp():
    workers_2__trainer = load_file('res/workers_2__trainer.pkl')
    workers_4__trainer = load_file('res/workers_4__trainer.pkl')
    workers_8__trainer = load_file('res/workers_8__trainer.pkl')

    trainer_list = [workers_0__trainer, workers_2__trainer, workers_4__trainer, workers_8__trainer]
    worker_list = [0, 2, 4, 8]
    
    time_list = [t.total_time for t in trainer_list]
    speedup_list = [time_list[0] / t for t in time_list]
    max_alloc_list = [sum(t.history["max_alloc"]) / len(t.history["max_alloc"]) for t in trainer_list]

    plot_time_and_speedup("workers", worker_list, time_list, speedup_list)
    plot_gpu_usage("workers", worker_list, max_alloc_list)


def dp_comp():
    dp_1__trainer = load_file('res/dp_1__trainer.pkl')
    dp_2__trainer = load_file('res/dp_2__trainer.pkl')
    dp_4__trainer = load_file('res/dp_4__trainer.pkl')

    trainer_list = [dp_1__trainer, dp_2__trainer, dp_4__trainer]
    gpu_list = [1, 2, 4]
    
    time_list = [t.total_time for t in trainer_list]
    speedup_list = [time_list[0] / t for t in time_list]
    max_alloc_list = [sum(t.history["max_alloc"]) / len(t.history["max_alloc"]) for t in trainer_list]

    plot_time_and_speedup("dp", gpu_list, time_list, speedup_list)
    plot_gpu_usage("dp", gpu_list, max_alloc_list)



def ddp_comp():
    ddp_2__trainer = load_file('res/ddp_2__trainer.pkl')
    ddp_4__trainer = load_file('res/ddp_4__trainer.pkl')

    trainer_list = [workers_0__trainer, ddp_2__trainer, ddp_4__trainer]
    gpu_list = [1, 2, 4]
    
    time_list = [t.total_time for t in trainer_list]
    speedup_list = [time_list[0] / t for t in time_list]
    max_alloc_list = [sum(t.history["max_alloc"]) / len(t.history["max_alloc"]) for t in trainer_list]

    plot_time_and_speedup("ddp", gpu_list, time_list, speedup_list)
    plot_gpu_usage("ddp", gpu_list, max_alloc_list)


def main():
    workers_0__trainer = load_file('res/workers_0__trainer.pkl')
    workers_comp()
    dp_comp()


if __name__ == "__main__":
    main()


    