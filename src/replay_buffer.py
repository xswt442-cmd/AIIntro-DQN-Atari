import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """经验回放池：定长队列，随机采样 batch 用于打破样本相关性"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 满后自动弹出最旧数据

    def push(self, state, action, reward, next_state, done):
        """存入一条转移 (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一个 batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
