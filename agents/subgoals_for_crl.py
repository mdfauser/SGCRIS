from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from agents.stable_crl import StableContrastiveRL

""" High-level policy for imagined subgoals"""


class LaplacePolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, device, hidden_dims=[256, 256]):
        super(LaplacePolicy, self).__init__()

        # Define the fully connected layers
        fc = [nn.Linear(state_dim + goal_dim, hidden_dims[0]), nn.ReLU()]
        for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
        self.fc = nn.Sequential(*fc)

        # Define mean and log_scale layers
        self.mean = nn.Linear(hidden_dims[-1], state_dim)
        self.log_scale = nn.Linear(hidden_dims[-1], state_dim)
        self.LOG_SCALE_MIN = -20
        self.LOG_SCALE_MAX = 2
        self.device = device
        self.to(self.device)

    def forward(self, state, goal):
        # goal = torch.FloatTensor(goal).to(self.device)
        if goal.ndim == 2 and goal.shape[0] == 1:
            goal = goal.expand(state.size(0), -1)

        h = self.fc(torch.cat([state, goal], -1))
        mean = self.mean(h)
        scale = self.log_scale(h).clamp(
            min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()
        distribution = torch.distributions.laplace.Laplace(mean, scale)
        return distribution


class ContrastiveSubgoalPolicy(nn.Module):
    def __init__(self, contrastive_qf, state_dim, repr_dim, device, hidden_dims=[256, 256], max_scale=0.1):
        super().__init__()
        self.device = device
        self.max_scale = max_scale
        self.contrastive_qf = contrastive_qf  # Contrastive Q-function

        # Extract encoders from contrastive model
        self.goal_encoder = contrastive_qf._g_encoder
        self.state_action_encoder = contrastive_qf._sa_encoder

        # Subgoal selection network
        self.subgoal_selector = Mlp(hidden_dims, repr_dim, repr_dim * 2)

        # Scale adjustment for exploration
        self.scale_net = Mlp(hidden_dims, repr_dim, repr_dim * 2)

        # Decoder to map from latent to state space
        self.decoder = Mlp(hidden_dims, state_dim, repr_dim)

        self.to(device)

    def forward(self, state, action, goal):
        state, action, goal = state.to(self.device), action.to(
            self.device), goal.to(self.device)

        # Encode inputs into latent space
        latent_sa = self.state_action_encoder(
            torch.cat([state, action], dim=-1))  # (Batch, repr_dim)
        latent_goal = self.goal_encoder(goal)  # (Batch, repr_dim)

        # Compute subgoal candidate in latent space
        latent_concat = torch.cat([latent_sa, latent_goal], dim=-1)
        latent_subgoal_mean = self.subgoal_selector(latent_concat)

        # Add diversity for exploration
        latent_subgoal_mean += 0.1 * torch.randn_like(latent_subgoal_mean)

        # Compute noise scale for exploration
        scale_raw = self.scale_net(latent_concat)
        scale = torch.exp(torch.clamp(scale_raw, -10, 2))
        scale = torch.clamp(scale, max=self.max_scale)

        # Bias sampling toward reachable goal-aligned states
        latent_subgoal = latent_subgoal_mean + self.max_scale * \
            torch.tanh(scale) * (latent_goal - latent_sa)

        # Decode subgoal back to state space
        subgoal_mean = self.decoder(latent_subgoal)

        # Decode scale for state-space subgoal distribution
        state_scale = self.decoder(scale)
        state_scale = torch.clamp(state_scale, min=1e-3)

        obs_goal = torch.cat([state, subgoal_mean], -1)
        # Use contrastive Q-function for reachability filtering
        q_value = self.contrastive_qf(obs_goal, action).sigmoid()
        if q_value.mean() < 0.2:  # If reachability is low, adjust subgoal
            subgoal_mean = state + 0.1 * (goal - state)

        # Return a Normal distribution in state space
        return torch.distributions.Normal(subgoal_mean, state_scale)


class Mlp(nn.Module):
    def __init__(self, hidden_dims, repr_shape, input_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], repr_shape)
        )

    def forward(self, inp):
        return self.net(inp)


class ContrastiveRLwithImaginedSubgoals(StableContrastiveRL):
    def __init__(
            self,
            device,
            policy,
            qf,
            lr=3e-4,
            optimizer_class=optim.Adam,
            entropy_coefficient=None,
            target_entropy=0.0,
            bc_coef=0.05,
            subgoal_net=None,  # Neuer Parameter für die abgeleitete Klasse
    ):
        # Basisklasse initialisieren
        super().__init__(
            device=device,
            policy=policy,
            qf=qf,
            lr=lr,
            optimizer_class=optimizer_class,
            entropy_coefficient=entropy_coefficient,
            target_entropy=target_entropy,
            bc_coef=bc_coef,
        )
        self.subgoal_net = subgoal_net
        self.subgoal_net_optimizer = optimizer_class(
            self.subgoal_net.parameters(),
            lr=lr,
        )
        self.target_actor = policy
        self.target_actor.load_state_dict(self.policy.state_dict())

        self.target_update_intervall = 1000
        self.Lambda = 0.1
        self.n_ensemble = 10
        self.total_it = 0
        self.epsilon = 1e-16
        self.tau = 0.005

    def sample_subgoal(self, state, action, goal):
        subgoal_distribution = self.subgoal_net(state, action, goal)
        subgoal = subgoal_distribution.rsample((self.n_ensemble,))
        subgoal = torch.transpose(subgoal, 0, 1)
        return subgoal

    def value_subgoal(self, state, goal):
        # for subgoals and goals
        if goal.shape[0] == 1:
            if goal.ndim == 2 and goal.shape[0] == 1:
                goal = goal.expand(state.size(0), -1)
        dist = self.policy.get_dist(state, goal)
        obs_goal = torch.cat([state, goal], -1)

        sampled_action = dist.rsample()
        q_action = self.qf(obs_goal, sampled_action)

        return torch.diag(q_action)

    def train_highlevel_policy(self, state, action, goal, subgoal):
        ris_dict = {}
        # Compute subgoal distribution
        subgoal_dist = self.subgoal_net(state, action, goal)

        with torch.no_grad():
            # Compute target value
            new_subgoal = subgoal_dist.loc
            policy_v_1 = self.value_subgoal(state, new_subgoal)
            policy_v_2 = self.value_subgoal(new_subgoal, goal)
            policy_v = torch.cat(
                [policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

            # Compute subgoal distance loss
            v_1 = self.value_subgoal(state, subgoal)
            v_2 = self.value_subgoal(subgoal, goal)
            v = torch.cat([v_1, v_2], -1).clamp(min=-
                                                100.0, max=0.0).abs().max(-1)[0]
            # policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=-100.0, max=0.0).max(-1)[0]
            # v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).max(-1)[0]

            adv = - (v - policy_v)
            weight = F.softmax(adv/self.Lambda, dim=0)

        log_prob = subgoal_dist.log_prob(subgoal).sum(-1)
        subgoal_loss = - (log_prob * weight).mean()

        # Update network
        self.subgoal_net_optimizer.zero_grad()
        subgoal_loss.backward()
        self.subgoal_net_optimizer.step()

        ris_dict["adv"] = adv.mean()
        ris_dict["ration_adv"] = adv.ge(0.0).float().mean()

        return ris_dict

    def sample_action_and_KL(self, state, goal, action_dist, single_goal=None):
        # Sample action, subgoals and KL-divergence

        sampled_action = action_dist.rsample()

        batch_size = state.size(0)

        with torch.no_grad():
            subgoal = self.sample_subgoal(state, sampled_action, goal)

        new_state = state.unsqueeze(1).expand(
            batch_size, subgoal.size(1), self.qf._obs_dim)
        prior_action_dist = self.target_actor.get_dist(new_state, subgoal)

        prior_prob = prior_action_dist.log_prob(sampled_action.unsqueeze(1).expand(
            batch_size, subgoal.size(1), self.qf._action_dim)).sum(-1, keepdim=True).exp()

        prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)

        log_prob = action_dist.log_prob(sampled_action).sum(-1, keepdim=True)
        lp_mean = log_prob.mean()
        plp_mean = prior_log_prob.mean()
        D_KL = log_prob - prior_log_prob
        # action = torch.tanh(action)

        return sampled_action, D_KL, log_prob

    def update(self, buffer, single_goal, batch_size):

        loss_dict = {}

        with torch.no_grad():

            transitions = buffer.sample(
                batch_size,
                1,
                to_device=self.device,
                keys=None
            )
            if transitions is None:
                return {}

        action = transitions['action'].squeeze(1)
        obs = transitions['observation'].squeeze(1)

        des_goal = transitions['desired_goal'].squeeze(1)

        new_goal = transitions['future_goal'].squeeze(
            1).as_subclass(torch.Tensor)
        batch_size = obs.shape[0]

        ris_dict = self.train_highlevel_policy(
            obs, action, des_goal, new_goal)  # single_goal

        I = torch.eye(batch_size, device=self.device)
        logits, sa_repr, g_repr, sa_repr_norm, g_repr_norm = self.qf(
            torch.cat([obs, new_goal], -1), action, repr=True)

        logits_log = logits
        correct = (torch.argmax(logits_log, dim=-1) == torch.argmax(I, dim=-1))
        logits_pos = torch.sum(logits_log * I) / torch.sum(I)
        logits_neg = torch.sum(logits_log * (1 - I)) / torch.sum(1 - I)
        q_pos, q_neg = torch.sum(torch.sigmoid(logits_log) * I) / torch.sum(I), \
            torch.sum(torch.sigmoid(logits_log) * (1 - I)) / torch.sum(1 - I)
        q_pos_ratio, q_neg_ratio = q_pos / (1 - q_pos), q_neg / (1 - q_neg)
        binary_accuracy = torch.mean(((logits_log > 0) == I).float())
        categorical_accuracy = torch.mean(correct.float())

        loss_dict["logits_pos"] = logits_pos
        loss_dict["logits_neg"] = logits_neg
        loss_dict["q_pos_ratio"] = q_pos_ratio
        loss_dict["q_neg_ratio"] = q_neg_ratio
        loss_dict["bin_acc"] = binary_accuracy
        loss_dict["cat_acc"] = categorical_accuracy

        # decrease the weight of negative term to 1 / (B - 1)
        qf_loss_weights = torch.ones(
            (batch_size, batch_size), device=self.device) / (batch_size - 1)
        qf_loss_weights[torch.arange(batch_size, device=self.device), torch.arange(
            batch_size, device=self.device)] = 1
        qf_loss = self.qf_criterion(logits, I)
        qf_loss *= qf_loss_weights
        qf_loss = torch.mean(qf_loss)
        loss_dict["qf_loss"] = qf_loss

        """
        Policy and Alpha Loss
        """

        random_goal = new_goal[torch.randperm(batch_size)]
        obs_goal = torch.cat([obs, random_goal], -1)
        action_dist = self.policy.get_dist(obs, random_goal)

        sampled_action, D_KL, _ = self.sample_action_and_KL(
            obs, random_goal, action_dist)  # single_goal

        ris_dict["D_KL"] = D_KL.mean()

        log_prob = action_dist.log_prob(sampled_action).mean(dim=-1)

        if self.adaptive_entropy_coefficient:
            alpha_loss = -(self.log_alpha.exp() * (
                log_prob + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros((1,), device=self.device)
            alpha = self.entropy_coefficient

        loss_dict["alpha_loss"] = alpha_loss

        q_action = self.qf(obs_goal, sampled_action)

        min_dkl = D_KL.min()
        max_dkl = D_KL.max()
        normalized_dkl = (D_KL - min_dkl) / (max_dkl - min_dkl + 1e-6)
        # adjusted for subgoals
        q_action_min = torch.min(torch.diag(q_action), -1, keepdim=True)[0]
        actor_loss_ris = (alpha * normalized_dkl * log_prob -
                          torch.diag(q_action))  # .mean()

        actor_q_loss = alpha * log_prob - torch.diag(q_action)

        assert 0.0 <= self.bc_coef <= 1.0
        orig_action = action

        train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()

        gcbc_loss = -train_mask * \
            action_dist.log_prob(orig_action).mean(dim=-1)
        # gcbc_val_loss = -(1.0 - train_mask) * dist.log_prob(orig_action)

        actor_loss = self.bc_coef * gcbc_loss + \
            (1 - self.bc_coef) * actor_loss_ris  # actor_q_loss
        actor_loss = torch.mean(actor_loss)

        loss_dict["actor_loss_ris"] = actor_loss_ris.mean()
        loss_dict["gcbc_loss"] = gcbc_loss.mean()
        loss_dict["actor_q_loss"] = actor_q_loss.mean()
        loss_dict["actor_loss"] = actor_loss
        """
        Optimization.
        """
        if self.adaptive_entropy_coefficient:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.total_it += 1
        if self.total_it % self.target_update_intervall == 0:
            for param, target_param in zip(self.policy.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict.update(ris_dict)

        return loss_dict
