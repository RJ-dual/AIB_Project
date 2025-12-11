"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
Unified training/evaluation for DQN, PPO (and future algorithms).
- Saves models to standardized paths: `cartpole_{algo}.torch`
- Evaluation auto-detects algorithm from filename if not specified.
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from agents.cartpole_actorcritic import ActorCriticConfig, ActorCriticNet, ActorCriticSolver
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


# ----------------------------
# Agent Factory & Path Utils
# ----------------------------
def create_agent(algorithm: str, obs_dim: int, act_dim: int):
    """Create agent by algorithm name."""
    algorithm = algorithm.lower()
    if algorithm == "dqn":
        return DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm == "ppo":
        return PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    elif algorithm == "actorcritic":
        return ActorCriticSolver(obs_dim, act_dim, cfg=ActorCriticConfig()) 
    else:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'. Choose from ['dqn', 'ppo'].")


def get_model_path(algorithm: str) -> str:
    """Return standardized model save path."""
    algorithm = algorithm.lower()
    if algorithm in ["dqn", "ppo","actorcritic"]:
        return os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")
    else:
        return os.path.join(MODEL_DIR, f"cartpole_{algorithm}.torch")



# ----------------------------
# Unified Training Function
# ----------------------------
def train(
    algorithm: str = "dqn",
    num_episodes: int = 200,
    terminal_penalty: bool = True,
    render: bool = False,
    seed_offset: int = 0,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = create_agent(algorithm, obs_dim, act_dim)
    model_path = get_model_path(algorithm)

    print(f"[Train] Algorithm: {algorithm.upper()}, Device: {agent.device}")
    print(f"[Train] Model will be saved to: {model_path}")

    logger = ScoreLogger(ENV_NAME)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed_offset + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        if hasattr(agent, 'reset_buffers'):
            agent.reset_buffers()

        while not done:
            action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0

            next_state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            agent.step(state, action, reward, next_state, done)
            state = next_state

            if render:
                time.sleep(1 / 60)

        logger.add_score(steps, ep)

        # Get algorithm-specific statistics
        if algorithm.lower() == "dqn":
            epsilon = agent.exploration_rate
            print(f"[{algorithm.upper():4s}] Ep {ep:3d} | Steps: {steps:3d} | ε: {epsilon:.3f}")
        elif algorithm.lower() in ["ppo", "actorcritic"]:
            stats = agent.get_stats()
            print(f"[{algorithm.upper():4s}] Ep {ep:3d} | Steps: {steps:3d} ")

    env.close()
    agent.save(model_path)
    print(f"[Train] ✅ Saved {algorithm.upper()} model to: {model_path}")
    return agent
# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate(
    model_path: str | None = None,
    algorithm: str | None = None,
    episodes: int = 5,
    render: bool = True,
    fps: int = 60,
):
    """
    Evaluate a trained model.

    If `model_path` is None: picks first .torch file in models/.
    If `algorithm` is None: infers from filename (e.g., '*ppo*' → 'ppo').
    """
    # Resolve model path
    if model_path is None:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No .torch model found in '{MODEL_DIR}'. Please train first.")
        model_path = os.path.join(MODEL_DIR, candidates[0])
        print(f"[Eval] Auto-selected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # Infer algorithm from filename if not given
    if algorithm is None:
        basename = os.path.basename(model_path).lower()
        if "ppo" in basename:
            algorithm = "ppo"
        elif "dqn" in basename:
            algorithm = "dqn"
        elif "actorcritic" in basename:
            algorithm = "actorcritic"
        else:
            raise ValueError(
                f"Cannot auto-detect algorithm from filename '{basename}'. "
                f"Please specify `algorithm=` explicitly (e.g., algorithm='ppo')."
            )
    print(f"[Eval] Algorithm: {algorithm.upper()}")

    # Create environment
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create agent
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "ppo":
        agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    elif algorithm.lower() == "actorcritic":
        agent = ActorCriticSolver(obs_dim, act_dim, cfg=ActorCriticConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Load model
    agent.load(model_path)
    print(f"[Eval] ✅ Loaded {algorithm.upper()} model from: {model_path}")

    # Run evaluation
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep:2d}: steps = {steps}")

    env.close()
    avg = np.mean(scores) if scores else 0.0
    print(f"\n[Eval] 📊 Average over {episodes} episodes: {avg:.2f}")
    return scores


# ----------------------------
# Convenience Aliases (Optional)
# ----------------------------
def train_dqn(**kwargs):
    return train(algorithm="dqn", **kwargs)

def train_ppo(**kwargs):
    return train(algorithm="ppo", **kwargs)

def train_actorcritic(**kwargs):
    return train(algorithm="actorcritic", **kwargs)


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    # 🔁 Example workflows:

    # ✅ Train DQN
    # train(algorithm="dqn", num_episodes=300)

    # ✅ Train PPO
    # train(algorithm="ppo", num_episodes=500)
    
    # ✅ Train Actor-Critic
    train(algorithm="actorcritic", num_episodes=500)

    # ✅ Evaluate the latest model (auto-detects algo)
    # evaluate(episodes=100, render=False)

    # ✅ Or evaluate specific model & algo:
    evaluate(model_path="models/cartpole_actorcritic.torch", algorithm="actorcritic", episodes=100, render=True)
