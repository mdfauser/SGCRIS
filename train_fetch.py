from collections import deque

import os
import argparse
import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.buffers import EpisodicBuffer, FunctionBuffer
from utils.logger import SimpleLogger
from agents.stable_crl import ContinuousFeedForwardPolicy, ContrastiveQf, StableContrastiveRL
from agents.sgcris import ContrastiveRLwithImaginedSubgoals, LaplacePolicy, ContrastiveSubgoalPolicy
from core.metrics import Plotter
from core.data_storage import DataStorage


def goal_to_obs(goal):
    """
        Transform goal to observation. For FetchReach, the endeffector position is set to the goal position
    """
    obs = torch.zeros((1, observation_shape),
                      device=args.device, dtype=torch.float32)
    obs[0, :3] = torch.tensor(goal, device=args.device, dtype=torch.float32)
    obs[0, 3:6] = torch.tensor(goal, device=args.device, dtype=torch.float32)
    return obs


def future_sampling(batch_size, trajectory_len, to_device, keys):
    max_episode_step = 50

    dist = torch.distributions.geometric.Geometric(probs=torch.tensor(0.1))
    trajectory_len = (dist.sample().clamp(0, max_episode_step)+1).int().item()

    transitions = buffer.sample(batch_size, trajectory_len, to_device, keys)
    if transitions is None:
        return None

    future_transition = {}
    for key, val in transitions.items():
        future_transition[key] = val[:, 0:1]
    future_transition["future_goal"] = transitions["observation"][:, -1]
    return future_transition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",           default="FetchReach-v2")
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--start_timesteps",    default=1000, type=int)
    parser.add_argument("--max_timesteps",      default=1e+6, type=int)
    parser.add_argument("--max_episode_length", default=50, type=int)
    parser.add_argument("--max_episodes",       default=30000, type=int)
    parser.add_argument("--batch_size",         default=256, type=int)
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="exp_FetchReach-v2")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--tau",                default=0.005, type=float)
    parser.add_argument("--gamma",              default=0.99, type=float)
    parser.add_argument("--lr",                 default=3e-4, type=float)
    parser.add_argument("--repr_dim",           default=64, type=int)
    parser.add_argument("--samples_per_insert", default=256, type=int)
    parser.add_argument("--temperature",        default=0.7, type=float)
    parser.add_argument('--use_subgoals',       default=True, type=bool)
    parser.add_argument('--use_single_goal',    default=True, type=bool)
    parser.add_argument('--hlp_latent',         default=True, type=bool)
    parser.add_argument("--load_model",         default=True,
                        action='store_true', help="Flag to load model and continue training")
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    env = gym.make(args.env_name, max_episode_steps=args.max_episode_length)
    env_test = gym.make(args.env_name,
                        max_episode_steps=args.max_episode_length)  # used for range goals

    # action_dim = env.action_space.shape[0]
    # state_dim = env.observation_space['observation'].shape[0]
    # goal_dim = env.observation_space['desired_goal'].shape[0]

    observation_shape = env.observation_space['observation'].shape[0]
    goal_shape = observation_shape
    action_shape = 4

    buffer_shapes = {
        "observation": [observation_shape],
        "action": [action_shape],
        "next_observation": [observation_shape],
        "desired_goal": [observation_shape]
    }

    future_buffer = FunctionBuffer(sample_function=future_sampling)
    buffer = EpisodicBuffer(num_episodes=args.max_episodes,
                            max_episode_length=args.max_episode_length, shapes=buffer_shapes)
    policy = ContinuousFeedForwardPolicy(
        size_in=observation_shape+goal_shape, size_out=action_shape).to(args.device)
    qf = ContrastiveQf(hidden_sizes=[256, 256], representation_dim=args.repr_dim,
                       action_dim=action_shape, goal_dim=goal_shape, obs_dim=observation_shape).to(args.device)
    if args.hlp_latent:
        subgoal_net = ContrastiveSubgoalPolicy(
            contrastive_qf=qf, state_dim=observation_shape, repr_dim=64, device=args.device)
    else:
        subgoal_net = LaplacePolicy(
            state_dim=observation_shape, goal_dim=goal_shape, device=args.device)
    if (args.use_subgoals):
        crl = ContrastiveRLwithImaginedSubgoals(device=args.device, policy=policy, qf=qf,
                                                entropy_coefficient=None, target_entropy=0.0, bc_coef=0.05, subgoal_net=subgoal_net)
    else:
        crl = StableContrastiveRL(device=args.device, policy=policy, qf=qf,
                                  entropy_coefficient=None, target_entropy=0.0, bc_coef=0.05)

    logger = SimpleLogger()
    plotter = Plotter(args.exp_name)

    if args.use_single_goal:
        folder = "results/{}/SGCRL/{}/".format(args.env_name, args.exp_name)
        single_goal_np = [1.3, 0.75, 0.65]  # [1.35, 1.05, 0.7]
        single_goal = goal_to_obs(np.array(single_goal_np))
    else:
        folder = f"results/{args.env_name}/ContrastiveRL/{args.exp_name}/"

    if args.load_model and os.path.exists(folder):
        print(f"Loading model from {folder}")
        crl.load(folder)
    else:
        print(f"No model found in {folder}, starting training from scratch.")

    successes = deque([], maxlen=100)
    total_steps = 0
    successes2 = deque([], maxlen=100)

    for e in range(args.max_episodes):

        obs_dict, _ = env.reset()
        # env.render()
        # if args.use_single_goal:
        #     env.env.goal = np.array(single_goal_np)
        #     env.env.env.env.goal = np.array(single_goal_np)

        goal = goal_to_obs(obs_dict["desired_goal"]).reshape((1, -1))
        state = torch.tensor(obs_dict["observation"].reshape(
            (1, -1)), device=args.device, dtype=torch.float32)
        success = 0.0

        for step in range(args.max_episode_length):
            with torch.no_grad():
                total_steps += 1

                if total_steps < args.start_timesteps:
                    action = env.action_space.sample()
                    next_obs_dict = env.step(action)[0]
                    action = torch.tensor(
                        action, device=args.device, dtype=torch.float32)
                else:
                    if args.use_single_goal:
                        action = policy.get_action(state, single_goal)
                    else:
                        action = policy.get_action(state, goal)

                    next_obs_dict = env.step(
                        action.detach().cpu().numpy().reshape((-1,)))[0]

                next_state = torch.tensor(next_obs_dict["observation"].reshape(
                    (1, -1)), device=args.device, dtype=torch.float32)
                buffer.append({"observation": state, "action": action,
                              "next_observation": next_state, "desired_goal": goal})

                if args.use_single_goal:
                    distance = np.linalg.norm(
                        next_obs_dict["achieved_goal"] - single_goal_np)
                else:
                    distance = np.linalg.norm(
                        next_obs_dict["achieved_goal"] - next_obs_dict["desired_goal"])
                state = next_state

            if total_steps >= args.start_timesteps and step % 16 == 0:
                if (args.use_subgoals):
                    if not args.use_single_goal:
                        single_goal = 0
                    update_dict = crl.update(
                        future_buffer, single_goal, batch_size=args.batch_size)
                else:
                    update_dict = crl.update(
                        future_buffer, batch_size=args.batch_size)
                logger.aggregate(update_dict)
                # plotter.update(update_dict, total_steps, (sum(successes)/100))

            if distance <= 0.05:
                success = 1.0
                break

        """This loop is for testing a range of goals"""
        if args.use_single_goal:
            obs_dict2, _ = env_test.reset()
            # goal2 = goal_to_obs(np.array([1.35, 1.05, 0.7])) #[1.5, 0.6, 0.4]
            goal2 = goal_to_obs(obs_dict2["desired_goal"]).reshape((1, -1))
            state2 = torch.tensor(obs_dict2["observation"].reshape(
                (1, -1)), device=args.device, dtype=torch.float32)
            success2 = 0.0
            for step2 in range(args.max_episode_length):
                with torch.no_grad():

                    action2 = policy.get_action(state2, goal2)

                    next_obs_dict2 = env_test.step(
                        action2.detach().cpu().numpy().reshape((-1,)))[0]

                    next_state2 = torch.tensor(next_obs_dict2["observation"].reshape(
                        (1, -1)), device=args.device, dtype=torch.float32)

                    distance2 = np.linalg.norm(
                        next_obs_dict2["achieved_goal"] - next_obs_dict2["desired_goal"])
                    state2 = next_state2

                if distance2 <= 0.05:
                    success2 = 1.0
                    break  # done

            successes2.append(success2)
        successes.append(success)
        plotter.store_only(total_steps, (sum(successes)/100),
                           (sum(successes2)/100))  # (sum(successes2)/100)

        if e % 1 == 0:
            print(
                f"[{e}, {total_steps}] {step+1} steps. Success rate: {sum(successes)} " + logger.print())
            logger.flush()
            print(f"multiple goals: {sum(successes2)}")

        # if e % 50 == 0:
        #     if not os.path.isdir(folder):
        #                 os.makedirs(folder)
        #     crl.save(folder, save_optims=False)

        buffer.new_episode()
    # plotter.plot_success_rate()
