# build_dataset.py
import gymnasium as gym
import numpy as np
import torch
from agents.cartpole_dqn import DQNSolver, DQNConfig

ENV_NAME = "CartPole-v1"
DATASET_PATH = "datasets/cartpole_offline.npz"
NUM_EPISODES = 300
MAX_STEPS = 500


def build_dataset():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 使用一个训练好的 DQN policy（或随机 policy 也行但效果会差）
    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    agent.load("D:\人工智能b\AIB_Project\models\DQN_lr0.0005_gamma0.99_eps0.99.torch")
    agent.exploration_rate = 0.05  # 近似 greedy

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for ep in range(NUM_EPISODES):
        s, _ = env.reset(seed=ep)
        for _ in range(MAX_STEPS):
            s_in = s.reshape(1, -1)
            a = agent.act(s_in, evaluation_mode=True)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)
            dones.append(done)

            s = s2
            if done:
                break

        if ep % 20 == 0:
            print(f"[Dataset] Episode {ep}/{NUM_EPISODES}")

    env.close()

    np.savez(
        DATASET_PATH,
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        next_states=np.array(next_states, dtype=np.float32),
        dones=np.array(dones, dtype=np.float32),
    )

    print(f"✅ Offline dataset saved to {DATASET_PATH}")
    print(f"Total transitions: {len(states)}")


if __name__ == "__main__":
    build_dataset()
