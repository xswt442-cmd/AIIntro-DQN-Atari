# AIIntro-DQN-Atari

基于 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（Mnih et al., 2013）实现的 DQN 深度强化学习算法，用于 Atari 游戏。

## 环境配置

```bash
# 有 NVIDIA 显卡：先装 CUDA 版 PyTorch，再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 无 GPU：直接安装全部
pip install -r requirements.txt
```

> 注意：gymnasium 需锁定在 0.29.x 版本（`<1.0.0`）。如果之前装过 1.x，建议重建环境或新建环境安装依赖。  

## 项目结构

```
├── src/
│   ├── model.py          # CNN Q 网络
│   ├── replay_buffer.py  # 经验回放池
│   ├── wrappers.py       # Atari 环境预处理 wrapper
│   └── agent.py          # DQN 智能体
├── scripts/
│   ├── train.py          # 训练脚本
│   └── eval.py           # 评估 / 演示脚本
├── results/              # 保存模型和训练曲线
├── requirements.txt
└── README.md
```

## 训练

```bash
python scripts/train.py --env PongNoFrameskip-v4 --episodes 5000
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--env` | `PongNoFrameskip-v4` | Atari 环境 ID |
| `--episodes` | `5000` | 训练回合数 |
| `--lr` | `2.5e-4` | 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--batch-size` | `32` | 批次大小 |
| `--buffer-size` | `100000` | 经验回放池容量 |
| `--target-update` | `10000` | target 网络更新间隔（步数） |
| `--save-dir` | `results` | 模型和曲线图输出目录 |
| `--device` | 自动检测 | `cuda` 或 `cpu` |

训练输出：
- `results/dqn_ep{N}.pth` — 定期保存的检查点
- `results/dqn_final.pth` — 最终模型
- `results/training_curve.png` — reward 和 loss 曲线

## 评估

```bash
python scripts/eval.py results/dqn_final.pth --episodes 5
```
