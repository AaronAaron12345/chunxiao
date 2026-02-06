#!/usr/bin/env python3
"""
52_launcher.py - 多GPU启动器
为每个GPU启动独立的Python进程
"""

import subprocess
import sys
import os
import time

def main():
    print("=" * 60)
    print("52_launcher.py - 多GPU并行启动器")
    print("=" * 60)
    
    # 数据集分配到8个GPU
    # GPU 0: datasets 0, 8
    # GPU 1: datasets 1, 9  
    # GPU 2: datasets 2, 10
    # GPU 3: datasets 3
    # GPU 4: datasets 4
    # GPU 5: datasets 5
    # GPU 6: datasets 6
    # GPU 7: datasets 7
    
    # GPU 0,6,7 被占用，使用 1-5
    gpu_assignments = {
        1: [0, 5, 10],
        2: [1, 6],
        3: [2, 7],
        4: [3, 8],
        5: [4, 9],
    }
    
    python_path = "/data1/condaproject/dinov2/bin/python3"
    
    processes = []
    log_dir = "/data2/image_identification/src/output"
    os.makedirs(log_dir, exist_ok=True)
    
    # 为每个GPU启动独立进程
    for gpu_id, datasets in gpu_assignments.items():
        datasets_str = ",".join(map(str, datasets))
        log_file = os.path.join(log_dir, f"gpu{gpu_id}_log.txt")
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {python_path} /data2/image_identification/src/final/52_worker.py {datasets_str} > {log_file} 2>&1"
        print(f"启动 GPU {gpu_id}: datasets {datasets}")
        
        p = subprocess.Popen(cmd, shell=True)
        processes.append((gpu_id, p))
    
    print(f"\n已启动 {len(processes)} 个GPU进程")
    print("等待所有进程完成...")
    
    # 等待所有进程
    start_time = time.time()
    for gpu_id, p in processes:
        p.wait()
        print(f"GPU {gpu_id} 完成 (返回码: {p.returncode})")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总所有GPU结果")
    print("=" * 80)
    
    for gpu_id in range(8):
        log_file = os.path.join(log_dir, f"gpu{gpu_id}_log.txt")
        if os.path.exists(log_file):
            print(f"\n--- GPU {gpu_id} ---")
            with open(log_file, 'r') as f:
                content = f.read()
                # 打印最后100行
                lines = content.strip().split('\n')
                for line in lines[-30:]:
                    print(line)


if __name__ == "__main__":
    main()
