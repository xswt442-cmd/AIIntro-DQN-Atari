"""Quick diagnostic: check if DQN is actually learning."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.wrappers import make_env

env = make_env("PongNoFrameskip-v4")
obs, _ = env.reset()

# Test 1: Q-values on a real observation
from src.model import DQN
model = DQN((4, 84, 84), 6)
model.eval()

obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).float() / 255.0
with torch.no_grad():
    q = model(obs_t).squeeze().numpy()

print("=== 诊断 ===")
print(f"obs 值范围: [{obs.min()}, {obs.max()}]")
print(f"Q-values: {q}")
print(f"Q 值差异 (max-min): {q.max() - q.min():.6f}")
print(f"→ {'正常' if abs(q.max() - q.min()) > 0.01 else '异常: Q值几乎相同，网络在输出常数!'}")

# Test 2: take some random actions, check rewards
rewards = []
for _ in range(200):
    _, r, _, _, _ = env.step(np.random.randint(6))
    rewards.append(r)
print(f"\n随机动作的 reward 分布: min={min(rewards)}, max={max(rewards)}, nonzero={np.count_nonzero(rewards)}")
print(f"→ {'正常' if np.count_nonzero(rewards) > 0 else '异常: 所有reward都是0!'}")

# Test 3: check if a trained checkpoint predicts varying Q-values
import glob
ckpts = glob.glob("results/dqn_ep*.pth")
if ckpts:
    ckpt = torch.load(sorted(ckpts)[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["policy_net"])
    model.eval()
    with torch.no_grad():
        q_trained = model(obs_t).squeeze().numpy()
    print(f"\n训练后的 Q-values: {q_trained}")
    print(f"训练后 Q 值差异: {q_trained.max() - q_trained.min():.6f}")
    print(f"→ {'正常' if abs(q_trained.max() - q_trained.min()) > 0.01 else '异常: 训练后网络仍在输出常数!'}")
else:
    print("\n未找到 checkpoint")

env.close()
