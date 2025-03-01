import random
import json
import os
from triangle_analysis import is_triangle,

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

# 生成所有点的三角形数据
def generate_data(space_size=100):
    data = []
    for a in range(1, space_size + 1):
        for b in range(1, space_size + 1):
            for c in range(1, space_size + 1):
                t_type = triangle_type(a, b, c)
                data.append({"a": a, "b": b, "c": c, "label": label_map[t_type]})
    return data

# 将训练数据随机拆分出验证集和测试集
def split_train_eval_test_from_train(data, eval_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)
    total = len(data)
    eval_end = int(eval_ratio * total)
    test_end = eval_end + int(test_ratio * total)
    eval_data = data[:eval_end]
    test_data = data[eval_end:test_end]
    train_data = data
    return train_data, eval_data, test_data

# 保存数据为jsonl文件
def save_data_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# 生成并保存train, eval, test数据集
if __name__ == "__main__":
    append_path = "data/"
    if not os.path.exists(append_path):
        os.makedirs(append_path)
    space_size = 100  # 100 * 100 * 100 的空间
    eval_ratio = 0.1  # 验证集占总数据的 10%
    test_ratio = 0.1  # 测试集占总数据的 10%
    # 生成完整的小空间数据
    data = generate_data(space_size=space_size)
    # 从训练数据中拆分验证集和测试集
    train_data, eval_data, test_data = split_train_eval_test_from_train(data, eval_ratio=eval_ratio, test_ratio=test_ratio)
    # 保存数据为JSONL文件
    save_data_to_jsonl(train_data, os.path.join(append_path, 'train.jsonl'))
    save_data_to_jsonl(eval_data, os.path.join(append_path, 'eval.jsonl'))
    save_data_to_jsonl(test_data, os.path.join(append_path, 'test.jsonl'))

    print(f"Data generated and saved as train.jsonl, eval.jsonl, and test.jsonl.")
    #print(f"Added {num_errors} erroneous samples to the training dataset.")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(eval_data)}")
    print(f"Test data size: {len(test_data)}")

    # visualize_data = generate_data(num_samples_per_class=1000, equilateral_samples=1000)
    # save_data_to_jsonl(visualize_data, os.path.join(append_path, 'visualize.jsonl'))