import os
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def normal_plot(border_ratios):
    custom_x_ticks = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Convert the existing ratio keys to numerical values and sort them
    ratios = sorted(border_ratios.keys(), key=lambda x: float(x.split("_")[1]))
    test_accuracies = [border_ratios[r] for r in ratios]

    # Convert ratio keys to percentage-based values for the x-axis
    ratio_values = [float(r.split("_")[1]) * 100 for r in ratios]

    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.plot(ratio_values, test_accuracies, marker='o', linestyle='-', color='b')

    # Set custom x-axis ticks
    plt.xticks(custom_x_ticks)

    # Labels and title
    plt.xlabel("Ratio (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Ratio")
    plt.grid(True)
    plt.show()

def std_dev_plot(border_ratios):
    border_ratios_raw = {r: [] for r in border_ratios.keys()}  # Store all values per ratio

    for i in range(10):
        path = os.path.join(result_dir, f"results_{i}/test_results_{i}.jsonl")
        if os.path.exists(path):
            data = read_jsonl(path)
            for item in data:
                if item["ratio"] in border_ratios_raw:
                    temp = item["test_acc"]
                    # temp = item["per_class_acc"]["0"]
                    border_ratios_raw[item["ratio"]].append(temp)

    # Compute mean and standard deviation for each ratio
    ratios = sorted(border_ratios_raw.keys(), key=lambda x: float(x.split("_")[1]))
    ratio_values = [float(r.split("_")[1]) * 100 for r in ratios]
    means = [np.mean(border_ratios_raw[r]) if border_ratios_raw[r] else 0 for r in ratios]
    std_devs = [np.std(border_ratios_raw[r]) if border_ratios_raw[r] else 0 for r in ratios]

    # Plot the mean with error bars
    plt.figure(figsize=(8, 5))
    plt.plot(ratio_values, means, marker='o', linestyle='-', color='b', label="Mean Accuracy")
    plt.fill_between(ratio_values, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs), 
                    color='b', alpha=0.2, label="±1 Std Dev")

    # Set custom x-axis ticks
    plt.xticks([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Labels and title
    plt.xlabel("Ratio (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Total Test Accuracy")
    # plt.title("Non-Triangle Test Accuracy")
    # plt.title("Scalene Test Accuracy")
    # plt.title("Isoceles Test Accuracy")
    # plt.title("Equilateral Test Accuracy")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("fig/b_2-total-plot.png")

def std_dev_smooth_plot(border_ratios):
    border_ratios_raw = {r: [] for r in border_ratios.keys()}  # Store all values per ratio

    for i in range(10):
        path = os.path.join(result_dir, f"results_{i}/test_results_{i}.jsonl")
        if os.path.exists(path):
            data = read_jsonl(path)
            for item in data:
                if item["ratio"] in border_ratios_raw:
                    border_ratios_raw[item["ratio"]].append(item["test_acc"])

    # Compute mean and standard deviation for each ratio
    ratios = sorted(border_ratios_raw.keys(), key=lambda x: float(x.split("_")[1]))
    ratio_values = [float(r.split("_")[1]) * 100 for r in ratios]
    means = [np.mean(border_ratios_raw[r]) if border_ratios_raw[r] else 0 for r in ratios]
    std_devs = [np.std(border_ratios_raw[r]) if border_ratios_raw[r] else 0 for r in ratios]

    # Use cubic spline interpolation to smooth the curve
    smooth_x = np.linspace(min(ratio_values), max(ratio_values), 300)  # More points for a smooth curve
    spline = scipy.interpolate.make_interp_spline(ratio_values, means, k=1)  # k=3 for cubic smoothing
    smooth_y = spline(smooth_x)

    # Smooth the standard deviation area
    spline_std_low = scipy.interpolate.make_interp_spline(ratio_values, np.array(means) - np.array(std_devs), k=1)
    spline_std_high = scipy.interpolate.make_interp_spline(ratio_values, np.array(means) + np.array(std_devs), k=1)
    smooth_std_low = spline_std_low(smooth_x)
    smooth_std_high = spline_std_high(smooth_x)

    # Plot the smoothed curve with standard deviation shading
    plt.figure(figsize=(8, 5))
    plt.plot(smooth_x, smooth_y, linestyle='-', color='b', label="Mean Accuracy")
    plt.fill_between(smooth_x, smooth_std_low, smooth_std_high, color='b', alpha=0.2, label="±1 Std Dev")

    # Set custom x-axis ticks
    plt.xticks([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Labels and title
    plt.xlabel("Ratio (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Ratio (Smoothed Curve)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    border_ratios = {"ratio_0.0": 0, "ratio_0.05": 0, "ratio_0.1": 0, "ratio_0.2": 0, "ratio_0.3": 0, "ratio_0.4": 0, "ratio_0.5": 0, "ratio_0.6": 0, "ratio_0.7": 0, "ratio_0.8": 0, "ratio_0.9": 0, "ratio_1.0": 0}

    result_dir = "/Users/mac/Desktop/三角形分类/border_case/boundary_2"
    
    for i in range(10):
        path = os.path.join(result_dir, f"results_{i}/test_results_{i}.jsonl")
        data = read_jsonl(path)
        for item in data:
            # temp = item["test_acc"]
            temp = item["per_class_acc"]["0"]
            border_ratios[item["ratio"]] = border_ratios[item["ratio"]] * i / (i + 1) + temp / (i + 1)


    # normal_plot(border_ratios)
    std_dev_plot(border_ratios)
    # std_dev_smooth_plot(border_ratios)

    print(border_ratios)