# AIIntro-DQN-Atari

基于 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（Mnih et al., 2013）实现的 DQN 深度强化学习算法，用于 Atari 游戏。

## 项目结构

```
├── src/
│   ├── model.py          # CNN Q 网络
│   ├── replay_buffer.py  # 经验回放池
│   ├── wrappers.py       # Atari 预处理
│   └── agent.py          # DQN 智能体
├── scripts/
│   ├── train.py          # 训练脚本
│   ├── eval.py           # 评估 / 录像脚本
│   └── diagnose.py       # 诊断脚本（检查 Q 值、reward 是否正常）
├── results/              # 训练产出（模型、曲线图）
├── requirements.txt
└── README.md
```

## 环境配置

```bash
# 有 NVIDIA 显卡：先装 CUDA 版 PyTorch，再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 无 GPU：直接安装全部
pip install -r requirements.txt
```

> 注意：gymnasium 需锁在 0.29.x（`<1.0.0`），如果之前装过 1.x 建议重建环境。

## 训练

```bash
python scripts/train.py
```

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--env` | `PongNoFrameskip-v4` | Atari 环境 ID |
| `--episodes` | `5000` | 训练回合数 |
| `--lr` | `1e-4` | Adam 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--batch-size` | `32` | 批次大小 |
| `--buffer-size` | `100000` | 经验回放池容量 |
| `--eps-decay` | `250000` | 探索率线性衰减步数 |
| `--target-update` | `10000` | target 网络更新间隔 |
| `--save-interval` | `500` | 检查点保存间隔（回合） |
| `--save-dir` | `results` | 输出目录 |
| `--device` | 自动检测 | `cuda` 或 `cpu` |

训练输出：
- `results/dqn_ep{N}.pth` — 每 500 轮的检查点
- `results/dqn_final.pth` — 最终模型
- `results/training_curve.png` — reward / loss 曲线

## 评估

```bash
# 命令行查看结果
python scripts/eval.py results/dqn_final.pth --episodes 5 --no-render

# 弹窗实时观看
python scripts/eval.py results/dqn_final.pth --episodes 5

# 录制视频
python scripts/eval.py results/dqn_final.pth --episodes 3 --record-dir results/videos
```

## 诊断

如果训练不收敛，运行诊断脚本检查 Q 值和 reward 是否正常：

```bash
python scripts/diagnose.py
```
