import random

import numpy as np
import torch
import torch.nn as nn

from .model import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN 智能体：ε-greedy 探索、target 网络、经验回放"""

    def __init__(
        self,
        input_shape,
        n_actions,
        lr=2.5e-4,
        gamma=0.99,
        buffer_size=100000,
        batch_size=32,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=1_000_000,
        target_update=10000,
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = device

        # 策略网络和目标网络
        self.policy_net = DQN(input_shape, n_actions).to(device)
        self.target_net = DQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss，对异常值不敏感

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def act(self, state, training=True):
        """ε-greedy 动作选择：训练时指数衰减探索率"""
        if training:
            eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
                -self.steps_done / self.eps_decay
            )
            self.steps_done += 1
        else:
            eps = 0.01  # 评估时小概率随机，避免卡死

        if random.random() < eps:
            return random.randrange(self.n_actions)

        state_t = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        """从回放池采样，计算 TD 误差并更新策略网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # 当前状态的 Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # TD target
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  # 梯度裁剪
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将策略网络权重复制到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """保存检查点（含网络、优化器、步数）"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        """加载检查点"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.target_net.eval()
