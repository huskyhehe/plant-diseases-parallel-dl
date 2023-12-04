import platform
import cpuinfo
import psutil
import GPUtil

def display_cpu_info():
    info = cpuinfo.get_cpu_info()
    virtual_memory = psutil.virtual_memory()
    print("CPU:")
    print(f"  CPU: {info['brand_raw']}")
    print(f"  Architecture: {info['arch']}")


def display_gpu_info():
    gpus = GPUtil.getGPUs()
    print("GPU:")
    print(f"  GPU count: {len(gpus)}")

    for gpu in gpus:
        memory_total_gb = gpu.memoryTotal / 1024
        memory_free_gb = gpu.memoryFree / 1024
        print(f"  Name: {gpu.name}, Memory Total: {memory_total_gb:.2f} GB, Memory Free: {memory_free_gb:.2f} GB")
