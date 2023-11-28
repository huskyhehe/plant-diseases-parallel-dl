import platform
import cpuinfo
import psutil
import GPUtil


def display_sys_info():
    print("System")
    print(f"  OS: {platform.system()} {platform.version()}")


def display_cpu_info():
    info = cpuinfo.get_cpu_info()
    virtual_memory = psutil.virtual_memory()
    print("\nCPU")
    print(f"  CPU: {info['brand_raw']}")
    print(f"  Architecture: {info['arch']}")
    print(f"  CPU Count: {info['count']}")
    print(f"  CPU Speed: {info['hz_actual_friendly']}")
    print(f"  Memory Total: {virtual_memory.total / (1024 ** 3):.2f} GB")


def display_gpu_info():
    gpus = GPUtil.getGPUs()
    print("\nGPU:")
    print(f"  GPU count: {len(gpus)}")

    for gpu in gpus:
        memory_total_gb = gpu.memoryTotal / 1024
        memory_free_gb = gpu.memoryFree / 1024
        print(f"  Name: {gpu.name}, Memory Total: {memory_total_gb:.2f} GB, Memory Free: {memory_free_gb:.2f} GB")


display_sys_info()
display_cpu_info()
display_gpu_info()
