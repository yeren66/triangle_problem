import datagen_border
import train
import os
import json

for i in range(10):
    datagen_border.main()
    border_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ratio in border_ratios:
        train.main(f"ratio_{ratio}", i)

# result_dir = "/Users/mac/Desktop/三角形分类/border_case"
# def read_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line.strip()))
#     return data

# border_ratios = {"ratio_0.0": 0, "ratio_0.05": 0, "ratio_0.1": 0, "ratio_0.2": 0, "ratio_0.3": 0, "ratio_0.4": 0, "ratio_0.5": 0, "ratio_0.6": 0, "ratio_0.7": 0, "ratio_0.8": 0, "ratio_0.9": 0, "ratio_1.0": 0}

# for i in range(10):
#     path = os.path.join(result_dir, f"results_{i}/test_results_{i}.jsonl")
#     data = read_jsonl(path)
#     for item in data:
#         border_ratios[item["ratio"]] = border_ratios[item["ratio"]] * i / (i + 1) + item["test_acc"] / (i + 1)

# print(border_ratios)