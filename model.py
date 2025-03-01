import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class Net_1(nn.Module):
     def __init__(self):
         super(Net_1, self).__init__()
         self.l1 = nn.Linear(3, 64)
         self.l2 = nn.Linear(64, 32)
         self.l3 = nn.Linear(32, 4)

     def forward(self, x):
         x = torch.relu(self.l1(x))
         x = torch.relu(self.l2(x))
         return self.l3(x)  # 输出不加 softmax，因为 CrossEntropyLoss 内部会处理

class Net_2(torch.nn.Module):
     def __init__(self):
         super(Net_2, self).__init__()
         self.l1 = torch.nn.Linear(3, 64)
         self.l2 = torch.nn.Linear(64, 256)
         self.l3 = torch.nn.Linear(256, 128)
         self.l4 = torch.nn.Linear(128, 64)
         self.l5 = torch.nn.Linear(64, 4)

     def forward(self, x):
         x = F.relu(self.l1(x))
         x = F.relu(self.l2(x))
         x = F.relu(self.l3(x))
         x = F.relu(self.l4(x))
         x = self.l5(x)
         return x
