"""DQN 评估 / 演示脚本 —— 加载模型观看或录制 Atari 游戏"""

import argparse
import os
import sys
import time

import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import DQNAgent
from src.wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on Atari")
    parser.add_argument("checkpoint", help="Path to checkpoint .pth file")
    parser.add_argument("--env", default="PongNoFrameskip-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true", help="Disable real-time rendering")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--record-dir", default=None, help="Directory to save mp4 videos")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record_dir else ("human" if not args.no_render else None)
    env = make_env(args.env, render_mode=render_mode)

    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.record_dir,
            episode_trigger=lambda ep: True,
            name_prefix="dqn",
        )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(input_shape=obs_shape, n_actions=n_actions, device=args.device)
    agent.load(args.checkpoint)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(obs, training=False)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            steps += 1

            if not args.record_dir and not args.no_render:
                time.sleep(1.0 / args.fps)  # 控制播放帧率

            if term or trunc:
                print(f"Episode {ep}: reward = {total_reward:.1f}, steps = {steps}")
                break

    env.close()
    if args.record_dir:
        print(f"Videos saved to {args.record_dir}")


if __name__ == "__main__":
    main()
