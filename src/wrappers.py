from collections import deque

import cv2
import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """随机 NOOP 若干帧再开始，增加初始状态多样性"""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, term, trunc, _ = self.env.step(self.noop_action)
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """需要按 FIRE 开始游戏的环境（如 Pong），reset 时自动发射"""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc:
            return self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(2)
        if term or trunc:
            return self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """跳过中间帧，取最后两帧像素最大值，减少计算量并消除闪烁"""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if term or trunc:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, term, trunc, info


class ResizeGrayscaleFrame(gym.ObservationWrapper):
    """转为灰度图并缩放至 84×84"""

    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self._shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self._shape, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """堆叠连续 k 帧作为网络输入，让 CNN 感知运动信息"""

    def __init__(self, env, k=4):
        super().__init__(env)
        self._k = k
        self._frames = deque(maxlen=k)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], k, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self):
        return np.array(self._frames)


class ClipRewardEnv(gym.RewardWrapper):
    """奖励裁剪：只保留符号 {-1, 0, +1}，稳定训练"""

    def reward(self, reward):
        return np.sign(reward)


def make_env(env_id, render_mode=None):
    """构建完整预处理后的 Atari 环境"""
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeGrayscaleFrame(env)
    env = FrameStack(env, k=4)
    env = ClipRewardEnv(env)
    return env
