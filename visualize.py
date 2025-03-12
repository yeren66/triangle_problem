import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from PIL import Image
import numpy as np
import random

def load_data_from_jsonl(file_path):
    """
    从 JSONL 文件中加载数据，每行是一条 JSON 记录。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def sample_data(data, n=10000, seed=42):
    """
    从 data 中随机选取 n 个元素，并保证结果的可复现性。
    
    参数：
    data: 输入数据（列表或数组形式）
    n: 需要选取的数量
    seed: 随机种子
    
    返回：
    一个包含 n 个元素的列表/数组
    """
    # 设定随机种子
    random.seed(seed)
    # data.sort()
    # 使用 random.sample 从列表中无放回地随机选取 n 个元素
    sampled_data = random.sample(data, n)
    return sampled_data

# def plot_3d_points(data):
#     """
#     使用 matplotlib 在 3D 空间中绘制数据点，不同 label 使用不同颜色。
#     data: [{'a': float, 'b': float, 'c': float, 'label': str}, ...]
#     """
#     # 先收集所有 label
#     labels = set(d['label'] for d in data)

#     # 为每个 label 生成一种颜色，这里直接使用 matplotlib 自带的调色板
#     color_map = {}
#     palette = plt.cm.Set1.colors  # 这里可以更换成你喜欢的调色板
#     for i, label in enumerate(labels):
#         color_map[label] = palette[i % len(palette)]

#     # 将同一 label 的点分组，方便一起绘制并只加一个图例
#     grouped_points = {}
#     for item in data:
#         lbl = item['label']
#         grouped_points.setdefault(lbl, []).append((item['a'], item['b'], item['c']))

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制每一类点
#     for lbl, points in grouped_points.items():
#         xs = [p[0] for p in points]
#         ys = [p[1] for p in points]
#         zs = [p[2] for p in points]
#         ax.scatter(xs, ys, zs, c=[color_map[lbl]], label=lbl)

#     ax.set_xlabel('a')
#     ax.set_ylabel('b')
#     ax.set_zlabel('c')
#     ax.legend()
#     plt.show()

def plot_3d_points(data, sample_size=10000):
    """
    使用 matplotlib 在 3D 空间中绘制数据点。
    
    - 如果数据条目数大于 sample_size，则会随机抽样指定数量的数据点。
    - border=True 的点用绿色（green），border=False 的点用红色（red）。
    - 不同的 label 会在图例中区分显示。
    - 先绘制绿色，再绘制红色。
    """
    # ---- 随机抽样，控制最大绘图数据量 ----
    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    # 将图中所有可能的 label 找出来（主要用于图例显示时避免重复）
    all_labels = sorted(set(d['true_label'] for d in data))

    # 先拆分数据，以便分别绘制绿色( border=True ) 和红色( border=False )  
    data_green = [d for d in data if d['match'] == True]
    data_red   = [d for d in data if d['match'] == False]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ---- 先绘制 border=True (绿色) ----
    # for lbl in all_labels:
    #     # 当前 label & border=True 的点
    #     points = [(d['a'], d['b'], d['c']) for d in data_green if d['label'] == lbl]
    #     if not points:
    #         continue
    #     xs = [p[0] for p in points]
    #     ys = [p[1] for p in points]
    #     zs = [p[2] for p in points]

    #     ax.scatter(xs, ys, zs, c='green', label=f"{lbl} (border=True)")

    # ---- 再绘制 border=False (红色) ----
    for lbl in all_labels:
        # 当前 label & border=False 的点
        points = [(d['a'], d['b'], d['c']) for d in data_red if d['true_label'] == lbl]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        ax.scatter(xs, ys, zs, c='red', label=f"{lbl} (border=False)")

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    # 去除重复图例
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()


def visualize_predictions_net_gif(data, output_file='3d_scatter_plot.gif', duration=10):
    """
    生成并保存 3D 散点图动画（GIF）。
    
    参数：
    - data: 预测数据，包含 'a', 'b', 'c' 和 'match' 字段
    - output_file: 输出 GIF 文件名
    - duration: 动画时长（秒）
    """
    # 分别保存匹配和不匹配的点
    true_points = {'x': [], 'y': [], 'z': []}
    false_points = {'x': [], 'y': [], 'z': []}

    for item in data:
        # if random.random() > 0.1:
        #     continue
        x, y, z = item['a'], item['b'], item['c']
        if item['match']:  # 如果预测正确
            true_points['x'].append(x)
            true_points['y'].append(y)
            true_points['z'].append(z)
        else:  # 如果预测错误
            # if item['label'] == 1 or item['label'] == 0:
            #     continue
            false_points['x'].append(x)
            false_points['y'].append(y)
            false_points['z'].append(z)

    # 创建 3D 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(true_points['x'], true_points['y'], true_points['z'], c='green', label='Correct Prediction (True)', marker='x')

    # 画出不匹配的点（红色）
    ax.scatter(false_points['x'], false_points['y'], false_points['z'], c='red', label='Incorrect Prediction (False)', marker='x')

    # 设置坐标轴标签
    ax.set_xlabel('a (x)')
    ax.set_ylabel('b (y)')
    ax.set_zlabel('c (z)')
    ax.set_title('3D Scatter Plot of Predictions')
    ax.legend()

    # 旋转动画函数
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)  # elev 是仰角，azim 是方位角

    # 生成动画
    frames = int(duration * 20)  # 40 fps * duration 秒
    angles = np.linspace(0, 90, frames)  # 旋转 360°
    ani = animation.FuncAnimation(fig, rotate, frames=angles, interval=100)

    # 保存为 GIF
    ani.save(output_file, writer='pillow', fps=10)

    print(f"旋转 3D 散点图已保存至 {output_file}")

if __name__ == '__main__':
    # 假设数据存放在 data.jsonl 文件中
    # data_file = '/home/yeren/triangle/data/ratio_0.5/train.jsonl'
    data_file = "/Users/mac/Desktop/三角形分类/border_case/boundary_3/results_1/ratio_1.0.jsonl"
    data = load_data_from_jsonl(data_file)
    data = sample_data(data, n=10000)
    # plot_3d_points(data)
    visualize_predictions_net_gif(data, output_file='fig/b_3-1-ratio_1.0.gif', duration=10)
