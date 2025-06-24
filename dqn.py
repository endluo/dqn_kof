import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=10):
        super(DQN, self).__init__()

        # 输入 (4, 240, 240)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=1)   # → (32, 59, 59)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)               # → (64, 30, 30)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)               # → (64, 15, 15)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)              # → (128, 8, 8)
        self.bn4 = nn.BatchNorm2d(128)

        flatten_dim = 128 * 8 * 8  # = 8192

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_actions)

    def forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, 59, 59)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 30, 30)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, 15, 15)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 128, 8, 8)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)            # (B, 8192)
        x = F.relu(self.fc1(x))              # (B, 256)
        x = F.relu(self.fc2(x))              # (B, 128)
        return self.out(x)               # (B, num_actions)

def select_action(model, state,device):
    """
    输入: state (4, 120, 120)
    输出: 动作编号 (0 ~ 7)
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0  # -> (1, 4, 800, 600)
    state_tensor = state_tensor.to(device)
    with torch.no_grad():
        q_values = model(state_tensor)  # -> (1, 8)
        action = q_values.argmax(dim=1).item()
    return action
if __name__=="__main__":
    model = DQN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 随机生成一个符合输入要求的 state 数据，形状为 (4, 800, 600)
    state = np.random.rand(4, 240, 240)
    print(select_action(model, state,device))