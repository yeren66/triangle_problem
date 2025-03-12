import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Net_1
from collections import defaultdict
from tqdm import tqdm
import os

# ========== 1. 定义数据集类，用于读取 JSONL 文件并返回 (x, y) ==========
class JSONLDataset(Dataset):
    def __init__(self, file_path):
        """
        file_path: JSONL 文件路径，每行形如：
            {"a": 51, "b": 80, "c": 95, "label": 1, "border": false}
        """
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回 (features, label)，其中 features 是 (a, b, c) 的张量，label 是 int
        """
        sample = self.data[idx]
        x = torch.tensor([sample['a'], sample['b'], sample['c']], dtype=torch.float)
        y = torch.tensor(sample['label'], dtype=torch.long)
        return x, y, sample

# ========== 2. 训练和评估流程函数 ==========
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个epoch，返回平均loss和accuracy。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y,z in dataloader:
        # 将数据放到device上（如CPU或GPU）
        x = x.to(device)
        y = y.to(device)

        # 前向计算
        outputs = model(x)
        loss = loss_fn(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, loss_fn, device):
    """
    在验证集或测试集上评估模型，返回平均loss和accuracy。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, z in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ========== 3. 主流程：读取数据、训练模型、测试并保存结果 ==========
def main(ratio="ratio_0.0", times=1):
    # 超参数
    batch_size = 64
    lr = 1e-3
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 构建Dataset和DataLoader
    test_dir = "data/train.jsonl"
    data_dir = "data/" + str(ratio) + "/"
    train_dataset = JSONLDataset(data_dir + "train.jsonl")
    eval_dataset = JSONLDataset(data_dir + "eval.jsonl")
    test_dataset = JSONLDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2) 实例化模型、定义损失函数和优化器
    model = Net_1().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 用于保存最佳模型的变量
    best_eval_acc = 0.0
    best_model_weights = None

    # 3) 训练
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        eval_loss, eval_acc = evaluate(model, eval_loader, loss_fn, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")

        # 如果当前验证精度高于历史最佳，保存该模型
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            # 深拷贝当前模型的权重
            best_model_weights = model.state_dict()

    # 训练完成后，恢复最佳模型
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Loaded best model weights with eval_acc={best_eval_acc:.4f}")

    # 4) 在测试集上测试，并将预测结果保存到文件
    print("Evaluating on test set...")
    # test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 将每条测试数据的预测结果保存到一个文件中
    
    append_dir = "results_" + str(times) + "/"
    os.makedirs(append_dir, exist_ok=True)
    append_path = append_dir + str(ratio) + ".jsonl"
    test_loss, test_acc, per_class_acc = evaluate_and_save_predictions(model, test_loader, device, loss_fn, append_path)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Per-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"Class {cls}: {acc:.4f}")

    with open(append_dir + "test_results_" + str(times) + ".jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            "ratio": ratio,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "per_class_acc": per_class_acc
        }, ensure_ascii=False) + "\n")
    print("Test results have been saved to 'test_results.jsonl'.")


def evaluate_and_save_predictions(model, dataloader, device, loss_fn, output_file):
    """
    既像 evaluate 那样计算测试集（或验证集）整体 loss/acc，
    又像 save_test_predictions 那样逐条写结果，并额外计算分类别准确率。

    返回: (test_loss, test_acc, per_class_acc_dict)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 分类别记录每个类别的预测正确数与该类别总数
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # 收集结果后一次性写文件，避免频繁 IO
    results = []

    with torch.no_grad():
        for x_batch, y_batch, raw_item_batch in tqdm(dataloader):
            # ========== 关键修改：如果 raw_item_batch 是字典，则把它“还原”成列表 ========== #
            if isinstance(raw_item_batch, dict):
                # 默认 collate_fn 将多个 sample 的同名键合并成一个 list/张量
                # 这里把它拆分还原为一个列表，每个元素是原始字典
                batch_size = x_batch.size(0)
                raw_list = []
                for i in range(batch_size):
                    one_item = {}
                    for key, val_list in raw_item_batch.items():
                        one_item[key] = val_list[i]
                    raw_list.append(one_item)
                raw_item_batch = raw_list
            # ======================================================================== #

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)       # [batch_size, num_classes]
            loss = loss_fn(outputs, y_batch)

            # 累加整体 loss，用于计算平均
            running_loss += loss.item() * x_batch.size(0)

            # 取预测值
            _, preds = torch.max(outputs, dim=1)

            # 计算整体 accuracy
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

            # 分类别准确率
            for i in range(len(x_batch)):
                true_label = y_batch[i].item()
                pred_label = preds[i].item()

                class_total[true_label] += 1
                if pred_label == true_label:
                    class_correct[true_label] += 1

                # 构造要写入文件的结果
                raw_data = raw_item_batch[i]
                one_result = {
                    "a": raw_data["a"].item(),
                    "b": raw_data["b"].item(),
                    "c": raw_data["c"].item(),
                    "true_label": raw_data["label"].item(),
                    "pred_label": pred_label,
                    "match": (pred_label == true_label)
                }
                results.append(one_result)

    # 计算整体 loss / accuracy
    test_loss = running_loss / total
    test_acc = correct / total

    # 计算分类别准确率
    per_class_acc = {}
    for cls_label, tot_cnt in class_total.items():
        per_class_acc[cls_label] = class_correct[cls_label] / tot_cnt if tot_cnt > 0 else 0.0

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return test_loss, test_acc, per_class_acc


if __name__ == "__main__":
    border_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ratio in border_ratios:
        main(f"ratio_{ratio}")

    # main("ratio_0.0")
