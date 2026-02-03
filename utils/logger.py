#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块
============
统一管理所有程序的日志输出

功能：
1. 控制台输出
2. 文件日志存储
3. 结果JSON保存
4. 运行时间记录
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path


def get_output_dir():
    """获取output目录路径"""
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_log_dir():
    """获取logs目录路径"""
    script_dir = Path(__file__).parent.parent
    log_dir = script_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    return log_dir


def setup_logger(name, log_file=None):
    """
    设置日志记录器
    
    Args:
        name: 日志名称（通常是脚本名）
        log_file: 日志文件名（可选）
    
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_path = get_log_dir() / log_file
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件: {log_path}")
    
    return logger


def save_results(results, name, extra_info=None):
    """
    保存运行结果到JSON文件
    
    Args:
        results: 结果字典
        name: 结果名称
        extra_info: 额外信息（可选）
    
    Returns:
        result_path: 保存的文件路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'{name}_{timestamp}.json'
    result_path = get_output_dir() / result_file
    
    # 添加元信息
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'name': name,
        **results
    }
    
    if extra_info:
        full_results['extra_info'] = extra_info
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    return result_path


class ExperimentTracker:
    """实验跟踪器 - 记录完整的实验信息"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.logger = setup_logger(experiment_name)
        self.results = {}
        
        self.logger.info("=" * 60)
        self.logger.info(f"实验开始: {experiment_name}")
        self.logger.info("=" * 60)
    
    def log(self, message, level='info'):
        """记录日志"""
        getattr(self.logger, level)(message)
    
    def log_config(self, config):
        """记录配置信息"""
        self.logger.info("配置参数:")
        for k, v in config.items():
            self.logger.info(f"  {k}: {v}")
        self.results['config'] = config
    
    def log_data_info(self, n_samples, n_features, n_folds):
        """记录数据信息"""
        self.logger.info(f"数据: {n_samples} 样本, {n_features} 特征")
        self.logger.info(f"交叉验证: {n_folds} folds")
        self.results['data_info'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_folds': n_folds
        }
    
    def log_progress(self, current, total, extra_info=''):
        """记录进度"""
        pct = current / total * 100
        self.logger.info(f"进度: {current}/{total} ({pct:.1f}%) {extra_info}")
    
    def log_results(self, mean_acc, std_acc, overall_acc=None, auc=None):
        """记录最终结果"""
        self.logger.info("=" * 60)
        self.logger.info("[最终结果]")
        self.logger.info(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
        if overall_acc:
            self.logger.info(f"  整体准确率: {overall_acc:.2f}%")
        if auc:
            self.logger.info(f"  AUC: {auc:.4f}")
        
        self.results['mean_accuracy'] = mean_acc
        self.results['std_accuracy'] = std_acc
        if overall_acc:
            self.results['overall_accuracy'] = overall_acc
        if auc:
            self.results['auc'] = auc
    
    def finish(self):
        """完成实验，保存结果"""
        end_time = datetime.now()
        elapsed = (end_time - self.start_time).total_seconds()
        
        self.logger.info(f"总用时: {elapsed:.1f}秒")
        self.logger.info("=" * 60)
        
        self.results['elapsed_time'] = elapsed
        self.results['start_time'] = self.start_time.isoformat()
        self.results['end_time'] = end_time.isoformat()
        
        # 保存结果
        result_path = save_results(self.results, self.experiment_name)
        self.logger.info(f"结果已保存: {result_path}")
        
        return result_path


if __name__ == '__main__':
    # 测试
    tracker = ExperimentTracker('test_experiment')
    tracker.log_config({'param1': 100, 'param2': 0.5})
    tracker.log_data_info(26, 4, 325)
    tracker.log_results(85.5, 12.3, 86.0, 0.92)
    tracker.finish()
