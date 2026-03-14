from collections import deque

import os
import argparse
import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import matplotlib.pyplot as plt

from buffers import EpisodicBuffer, FunctionBuffer
from logger import SimpleLogger
from stable_crl import ContinuousFeedForwardPolicy, ContrastiveQf, StableContrastiveRL


if __name__ == "__main__":

    env_name = "FetchPush-v2"
    exp_name = "fetchPush_SGCRL_hard_neg_close_simple"
    use_single_goal = True
    max_episode_steps = 50
    device = "cuda"
    repr_dim = 64

    env = gym.make(env_name, render_mode="human",
                   max_episode_steps=max_episode_steps)

    if use_single_goal:
        folder = "results/{}/SGCRL/{}/".format(env_name, exp_name)
    else:
        folder = f"results/{env_name}/ContrastiveRL/{exp_name}/"

    action_shape = env.action_space.shape[0]
    observation_shape = env.observation_space['observation'].shape[0]
    goal_shape = observation_shape

    policy = ContinuousFeedForwardPolicy(
        size_in=observation_shape+goal_shape, size_out=action_shape).to(device)
    qf = ContrastiveQf(hidden_sizes=[256, 256], representation_dim=repr_dim,
                       action_dim=action_shape, goal_dim=goal_shape, obs_dim=observation_shape).to(device)
    crl = StableContrastiveRL(device=device, policy=policy, qf=qf,
                              entropy_coefficient=None, target_entropy=0.0, bc_coef=0.05)

    crl.load(folder)

    """Evaluates the policy in the environment."""

    distance_threshold = 0.05
    num_episodes = 100
    final_distance = []
    successes = []
    steps = []
    print(f"Evaluation for desired goals using",
          "single goal" if use_single_goal else "multiple goals", "for training")
    hit_till_end = False
    if hit_till_end:
        for _ in range(100):
            obs, _ = env.reset()
            env.render()
            done = False
            state = obs["observation"]
            goal_np = obs["desired_goal"]
            goal = torch.zeros((1, observation_shape),
                               device=device, dtype=torch.float32)
            goal[0, 3:6] = torch.tensor(
                goal_np, device=device, dtype=torch.float32)
            t = 0

            while not done and t < max_episode_steps:
                state = torch.tensor(state.reshape(
                    (1, -1)), device=device, dtype=torch.float32)
                action = policy.get_action(state, goal)
                next_obs, reward, done, truncated, info = env.step(
                    action.detach().cpu().numpy().reshape((-1,)))
                next_state = next_obs["observation"]
                state = next_state
                t += 1
            distance = np.linalg.norm(state[:3] - goal_np[:3])
            final_distance.append(distance)
            successes.append(1.0 * (distance < distance_threshold))
            steps.append(t)

        eval_distance, success_rate, avg_steps = np.mean(
            final_distance), np.mean(successes), np.mean(steps)
        print("avg distance: ", eval_distance, "| success rate: ",
              success_rate, "| avg steps: ", avg_steps)

    print("Testing for when hit once.")
    successes2 = deque([], maxlen=100)
    steps = []
    for _ in range(100):
        obs, _ = env.reset()
        env.render()
        done = False
        state = obs["observation"]
        goal_np = obs["desired_goal"]
        goal = torch.zeros((1, observation_shape),
                           device=device, dtype=torch.float32)
        goal[0, :3] = torch.tensor(goal_np, device=device, dtype=torch.float32)
        t = 0
        success = 0.0
        while not done and t < max_episode_steps:
            state = torch.tensor(state.reshape((1, -1)),
                                 device=device, dtype=torch.float32)
            action = policy.get_action(state, goal)
            next_obs, reward, done, truncated, info = env.step(
                action.detach().cpu().numpy().reshape((-1,)))
            next_state = next_obs["observation"]
            state = next_state
            t += 1
            distance = np.linalg.norm(state[:3] - goal_np[:3])
            if distance < distance_threshold:
                success = 1.0
                break

        final_distance.append(distance)
        successes2.append(success)
        steps.append(t)

    eval_distance, success_rate, avg_steps = np.mean(
        final_distance), sum(successes2), np.mean(steps)
    print("avg distance: ", eval_distance, "| success rate: ",
          success_rate, "| avg steps: ", avg_steps)

    print("testing only one single hard goal")
    for _ in range(100):
        obs, _ = env.reset()
        env.render()
        done = False
        state = obs["observation"]
        goal_np = np.array([1.5, 0.6, 0.4])  # obs["desired_goal"]
        goal = torch.zeros((1, observation_shape),
                           device=device, dtype=torch.float32)
        goal[0, 3:6] = torch.tensor(
            goal_np, device=device, dtype=torch.float32)
        env.env.goal = goal
        env.env.env.env.goal = goal_np
        obs["desired_goal"] = goal
        t = 0

        while not done and t < max_episode_steps:
            state = torch.tensor(state.reshape((1, -1)),
                                 device=device, dtype=torch.float32)
            action = policy.get_action(state, goal)
            next_obs, reward, done, truncated, info = env.step(
                action.detach().cpu().numpy().reshape((-1,)))
            next_state = next_obs["observation"]
            state = next_state
            t += 1
        distance = np.linalg.norm(state[:3] - goal_np[:3])
        final_distance.append(distance)
        successes.append(1.0 * (distance < distance_threshold))
        steps.append(t)

    eval_distance, success_rate, avg_steps = np.mean(
        final_distance), np.mean(successes), np.mean(steps)
    print("avg distance: ", eval_distance, "| success rate: ",
          success_rate, "| avg steps: ", avg_steps)
