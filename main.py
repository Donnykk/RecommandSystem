import os
import time
import psutil
from Analyze import AnalyzeData
from Train import train


def show_info():
    # 计算消耗内存
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    return memory
    # print(f'{start} 一共占用{memory:.2f}MB')


if __name__ == '__main__':
    AnalyzeData()
    start_time = time.time()
    start_memory = show_info()
    train()
    end_time = time.time()
    end_memory = show_info()
    print("共花费", end_time - start_time, "秒")
    print(f'一共占用 {end_memory - start_memory} MB')
