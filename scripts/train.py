"""DQN 训练脚本 —— 从项目根目录运行: python scripts/train.py"""

import argparse
import os
import sys
from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import torch

# 允许从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import DQNAgent
from src.wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Atari")
    parser.add_argument("--env", default="PongNoFrameskip-v4")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4) # Adam 通常推荐使用 1e-4
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000) # 将 1,000,000 降为 10w 防止家用电脑 OOM
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-decay", type=int, default=250_000)
    parser.add_argument("--target-update", type=int, default=10000)
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")

    env = make_env(args.env)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"Environment: {args.env}, observations: {obs_shape}, actions: {n_actions}")

    agent = DQNAgent(
        input_shape=obs_shape,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        eps_decay=args.eps_decay,
        target_update=args.target_update,
        device=args.device,
    )

    episode_rewards = []
    episode_losses = []
    start_time = time()
    step = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        while True:
            action = agent.act(obs, training=True)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # 存经验 → 学习
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # 加入预热期并降低训练频率
            if step >= 10000 and step % 4 == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

            obs = next_obs
            total_reward += reward
            step += 1

            if step % agent.target_update == 0:
                agent.update_target()

            if done:
                break

        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        elapsed = str(timedelta(seconds=int(time() - start_time)))
        print(
            f"Ep {ep:5d} | reward: {total_reward:7.1f} | loss: {avg_loss:.5f} | "
            f"eps: {max(agent.eps_end, agent.eps_start - agent.steps_done * (agent.eps_start - agent.eps_end) / agent.eps_decay):.3f} | "
            f"steps: {step} | time: {elapsed}"
        )

        if ep % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"dqn_ep{ep}.pth")
            agent.save(save_path)
            plot_path = os.path.join(args.save_dir, "training_curve.png")
            _plot_curve(episode_rewards, episode_losses, plot_path)

    env.close()
    final_path = os.path.join(args.save_dir, "dqn_final.pth")
    agent.save(final_path)
    plot_path = os.path.join(args.save_dir, "training_curve.png")
    _plot_curve(episode_rewards, episode_losses, plot_path)
    print(f"Done. Model saved to {final_path}")


def _plot_curve(rewards, losses, path):
    """绘制 reward 和 loss 曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax2.plot(losses)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Average Loss per Episode")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
