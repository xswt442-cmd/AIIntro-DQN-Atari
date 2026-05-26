import torch
import torch.nn as nn


class DQN(nn.Module):
    """CNN Q 网络，输入 4 帧 84×84 灰度图，输出每个动作的 Q 值"""

    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        # 三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 用 dummy 输入自动推导卷积输出尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0  # 归一化到 [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
