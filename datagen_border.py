import random
import json
import os
from triangle_analysis import is_non_triangle_border, is_normal_triangle_border

# 标签映射：将三角形类型转换为整数
label_map = {
    'non-triangle': 0,
    'scalene': 1,
    'isosceles': 2,
    'equilateral': 3
}

# 判断是否为三角形
def is_triangle(a, b, c):
    return a + b > c and a + c > b and b + c > a

# 判断三角形类型
def triangle_type(a, b, c):
    if not is_triangle(a, b, c):
        return 'non-triangle'
    elif a == b == c:
        return 'equilateral'
    elif a == b or b == c or a == c:
        return 'isosceles'
    else:
        return 'scalene'

def generate_data(space_size=100):
    border_size_n = 0  # non-triangle 的边界计数
    border_size_s = 0  # scalene 的边界计数
    data = []
    for a in range(1, space_size + 1):
        for b in range(1, space_size + 1):
            for c in range(1, space_size + 1):
                t_type = triangle_type(a, b, c)
                # if t_type == 'isosceles':
                #     continue
                # if t_type == 'equilateral':
                #     continue

                if t_type == 'non-triangle':
                    border = is_non_triangle_border(a, b, c)
                    border_size_n += 1 if border else 0 
                elif t_type == 'scalene':
                    border = is_normal_triangle_border(a, b, c)
                    border_size_s += 1 if border else 0
                else:
                    # 如果还存在其他类型，也可在此处理
                    border = False

                data.append({
                    "a": a,
                    "b": b,
                    "c": c,
                    "label": label_map[t_type],  # 'non-triangle' 或 'scalene'
                    "border": border
                })

    print(f"Generated {len(data)} samples.")
    print(f"Generated {border_size_n} non-triangle border samples.")
    print(f"Generated {border_size_s} scalene border samples.")
    return data

# =============== 2. 将数据按类别、border 拆分 ===============
def split_by_label_and_border(data):
    """
    将所有数据拆分成:
    - non_triangle_border
    - non_triangle_non_border
    - scalene_border
    - scalene_non_border

    返回一个 dict，方便后续按需抽样。
    """
    non_triangle_border = []
    non_triangle_non_border = []
    scalene_border = []
    scalene_non_border = []
    isosceles = []
    equilateral = []
    for d in data:
        if d['label'] == 0:
            if d['border']:
                non_triangle_border.append(d)
            else:
                non_triangle_non_border.append(d)
        elif d['label'] == 1:
            if d['border']:
                scalene_border.append(d)
            else:
                scalene_non_border.append(d)
        elif d['label'] == 2:
            isosceles.append(d)
        elif d['label'] == 3:
            equilateral.append(d)

    return {
        'non_triangle_border': non_triangle_border,
        'non_triangle_non_border': non_triangle_non_border,
        'scalene_border': scalene_border,
        'scalene_non_border': scalene_non_border,
        'isosceles': isosceles,
        'equilateral': equilateral
    }

# =============== 3. 根据给定的“边界占比”抽取样本 ===============
def sample_data(split_dict, label_name, total_num=5000, border_ratio=0.0):
    """
    从指定类别的数据里（non_triangle 或 scalene），随机抽取 total_num 条样本，
    其中 border_ratio（例如0.05表示5%）对应的数量来自“border”数据，剩下的来自“non-border”数据。

    - split_dict：由 split_by_label_and_border 返回的 dict
    - label_name: 'non_triangle' 或 'scalene'
    - total_num: 抽样总量，比如 5000
    - border_ratio: 边界数据占总量的比例，0.0 ~ 1.0
    """
    # 根据 label_name 找到对应的 border / non-border 列表
    if label_name == 0:
        border_list = split_dict['non_triangle_border']
        non_border_list = split_dict['non_triangle_non_border']
    elif label_name == 1:
        border_list = split_dict['scalene_border']
        non_border_list = split_dict['scalene_non_border']
    elif label_name == 2:
        border_list = split_dict['isosceles']
        non_border_list = []
    elif label_name == 3:
        total_num = len(split_dict['equilateral'])
        border_list = split_dict['equilateral']
        non_border_list = []

    if label_name == 2 or label_name == 3:
        return random.sample(border_list, total_num)

    # 计算需要 border 的数量
    border_count = int(total_num * border_ratio)
    # 其余由 non-border 补足
    non_border_count = total_num - border_count

    if border_count > len(border_list):
        raise ValueError(
            f"边界样本不够，想要 {border_count} 条，实际只有 {len(border_list)} 条。"
        )
    if non_border_count > len(non_border_list):
        raise ValueError(
            f"非边界样本不够，想要 {non_border_count} 条，实际只有 {len(non_border_list)} 条。"
        )

    border_samples = random.sample(border_list, border_count)
    non_border_samples = random.sample(non_border_list, non_border_count)

    return border_samples + non_border_samples

def split_train_eval_test_from_train(data, eval_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)
    total = len(data)
    eval_end = int(eval_ratio * total)
    test_end = eval_end + int(test_ratio * total)
    eval_data = data[:eval_end]
    test_data = data[eval_end:test_end]
    train_data = data[test_end:]
    return train_data, eval_data, test_data

def save_data_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    # random.seed(42)  # 固定随机数种子，便于结果复现；可去掉

    # 1. 生成数据
    data = generate_data(space_size=100)  # 这里你可用更大的 space_size
    print(f"Generated {len(data)} samples.")

    # 2. 拆分数据
    split_dict = split_by_label_and_border(data)

    # 3. 多次抽样，示例使用 0%、5%、10% 三种边界占比
    border_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ratio in border_ratios:
        # 3.1 在 non-triangle 中抽取
        try:
            sampled_non_triangle = sample_data(
                split_dict, label_name=0,
                total_num=2000, border_ratio=ratio
            )
            print(f"[non-triangle] border_ratio={ratio*100}% 抽取到 {len(sampled_non_triangle)} 条数据")
            # TODO: 这里你可以将 sampled_non_triangle 写到文件 / 后续处理

        except ValueError as e:
            print(f"[non-triangle] border_ratio={ratio*100}% 抽样失败: {e}")

        # 3.2 在 scalene 中抽取
        try:
            sampled_scalene = sample_data(
                split_dict, label_name=1,
                total_num=5000, border_ratio=ratio
            )
            print(f"[scalene] border_ratio={ratio*100}% 抽取到 {len(sampled_scalene)} 条数据")
            # TODO: 这里你可以将 sampled_scalene 写到文件 / 后续处理

        except ValueError as e:
            print(f"[scalene] border_ratio={ratio*100}% 抽样失败: {e}")

        # 3.3 在 isoceles 中抽取
        try:
            sampled_isoceles = sample_data(
                split_dict, label_name=2,
                total_num=5000, border_ratio=ratio
            )
            print(f"[isoceles] border_ratio={ratio*100}% 抽取到 {len(sampled_isoceles)} 条数据")

        except ValueError as e:
            print(f"[isoceles] border_ratio={ratio*100}% 抽样失败: {e}")

        # 3.4 在 equilateral 中抽取
        try:
            sampled_equilateral = sample_data(
                split_dict, label_name=3,
                total_num=100, border_ratio=ratio
            )
            print(f"[equilateral] border_ratio={ratio*100}% 抽取到 {len(sampled_equilateral)} 条数据")

        except ValueError as e:
            print(f"[equilateral] border_ratio={ratio*100}% 抽样失败: {e}")

        data = sampled_non_triangle + sampled_scalene + sampled_isoceles + sampled_equilateral
        train_data, eval_data, test_data = split_train_eval_test_from_train(data)
        data_dir = 'data/ratio_' + str(ratio)
        os.makedirs(data_dir, exist_ok=True)
        save_data_to_jsonl(train_data, os.path.join(data_dir, 'train.jsonl'))
        save_data_to_jsonl(eval_data, os.path.join(data_dir, 'eval.jsonl'))
        save_data_to_jsonl(test_data, os.path.join(data_dir, 'test.jsonl'))

# =============== 主流程示例 ===============
if __name__ == '__main__':
    
    main()
