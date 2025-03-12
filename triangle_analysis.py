import math
import numpy as np
from collections import defaultdict
import json

def is_non_triangle_border(a, b, c, threshold=1):
    """判断是否为非三角形的边界数据
    
    参数:
        a, b, c: 三角形三边长
        threshold: 边长判断的相对阈值（默认1%）
    
    返回:
        is_border: 是否是边界数据
    """
    # 计算三边之和的关系
    side_sums = [
        (a + b - c),  # a + b 与 c 的关系
        (b + c - a),  # b + c 与 a 的关系
        (a + c - b)   # a + c 与 b 的关系
    ]
    
    # 使用相对误差判断
    for i, sum_pair in enumerate(side_sums):
        if i == 0:
            relative_error = abs(sum_pair) # / (a + b)
        elif i == 1:
            relative_error = abs(sum_pair) # / (b + c)
        else:
            relative_error = abs(sum_pair) # / (a + c)
            
        if relative_error < threshold:
            return True
            
    return False

def is_normal_triangle_border(a, b, c, threshold=1):
    """判断是否为普通三角形的边界数据
    
    参数:
        a, b, c: 三角形三边长
        threshold: 边长判断的相对阈值（默认2%）
    
    返回:
        is_border: 是否是边界数据
    """
    # 计算两边之差的相对误差
    sides = [a, b, c]
    max_side = max(sides)
    
    # 1. 判断是否接近不能构成三角形的情况
    # 对于三角形，任意两边之和必须大于第三边
    # 如果两边之和接近第三边，说明是边界数据
    side_sums = [
        (a + b, c),  # a + b 与 c 的关系
        (b + c, a),  # b + c 与 a 的关系
        (a + c, b)   # a + c 与 b 的关系
    ]
    
    for sum_pair, third_side in side_sums:
        # 计算两边之和与第三边的差值的相对误差
        relative_diff = abs(sum_pair - third_side) #/ sum_pair
        
        if relative_diff < threshold:
            return True
    
    # 2. 判断是否接近等腰三角形
    # 如果任意两边接近相等，就认为是边界数据
    for i in range(3):
        for j in range(i + 1, 3):
            relative_diff = abs(sides[i] - sides[j]) # / max_side
            if relative_diff < threshold:
                return True
    
    return False

def analyze_dataset(data_path='./data_4_class/train.jsonl'):
    """分析数据集中的边界情况，只分析非三角形和普通三角形"""
    # 存储每个类别的边界数据
    border_cases = defaultdict(list)
    total_cases = defaultdict(int)
    
    # 读取数据
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            label = item['label']
            
            # 只处理非三角形(0)和普通三角形(1)
            if label not in [0, 1]:
                continue
                
            a, b, c = item['a'], item['b'], item['c']
            total_cases[label] += 1
            
            # 根据类别使用不同的边界判断函数
            is_border = False
            if label == 0:  # 非三角形
                is_border = is_non_triangle_border(a, b, c)
            elif label == 1:  # 普通三角形
                is_border = is_normal_triangle_border(a, b, c)
            
            if is_border:
                border_cases[label].append({
                    'sides': [a, b, c],
                    'index': len(border_cases[label])
                })
    
    # 打印分析结果
    class_names = ['非三角形', '普通三角形']
    print("\n数据集边界情况分析:")
    print("-" * 50)
    
    for label in [0, 1]:
        n_borders = len(border_cases[label])
        total = total_cases[label]
        print(f"\n{class_names[label]}:")
        print(f"总样本数: {total}")
        print(f"边界样本数: {n_borders} ({n_borders/total:.2%})")
    
    return border_cases, total_cases

if __name__ == "__main__":
    # 分析数据集
    # border_cases, total_cases = analyze_dataset()
    total = 0
    boundary = 0
    for i, j, k in [(i, j, k) for i in range(1, 101) for j in range(1, 101) for k in range(1, 101)]:
        if i == j or i == k or j == k:
            continue
        total += 1
        if is_normal_triangle_border(i, j, k, 3):
            boundary += 1
    print(f"总样本数: {total}")
    print(f"边界样本数: {boundary} ({boundary/total:.2%})")

