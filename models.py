"""DRL model training: A2C, PPO, DDPG using Stable Baselines 3."""

import os
import numpy as np
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from config import A2C_PARAMS, PPO_PARAMS, DDPG_PARAMS, TIMESTEPS


MODELS_DIR = "trained_models"
LOG_DIR = "tensorboard_log"


def get_model(algorithm, env, model_kwargs=None):
    """Initialize a DRL model."""
    os.makedirs(LOG_DIR, exist_ok=True)

    if algorithm == "a2c":
        params = {**A2C_PARAMS, **(model_kwargs or {})}
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=LOG_DIR, **params)
    elif algorithm == "ppo":
        params = {**PPO_PARAMS, **(model_kwargs or {})}
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=LOG_DIR, **params)
    elif algorithm == "ddpg":
        params = {**DDPG_PARAMS, **(model_kwargs or {})}
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model = DDPG("MlpPolicy", env, action_noise=action_noise,
                      verbose=0, tensorboard_log=LOG_DIR, **params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


def train_model(model, algorithm, timesteps=None):
    """Train a DRL model."""
    if timesteps is None:
        timesteps = TIMESTEPS.get(algorithm, 100000)
    print(f"Training {algorithm.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, tb_log_name=algorithm)
    print(f"Training complete for {algorithm.upper()}.")
    return model


def save_model(model, algorithm):
    """Save trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, algorithm)
    model.save(path)
    print(f"Model saved to {path}")


def load_model(algorithm, env):
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, algorithm)
    if algorithm == "a2c":
        return A2C.load(path, env=env)
    elif algorithm == "ppo":
        return PPO.load(path, env=env)
    elif algorithm == "ddpg":
        return DDPG.load(path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def predict_with_model(model, env):
    """Run a trained model on the environment and return portfolio stats."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_portfolio_stats()
