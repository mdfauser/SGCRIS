from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn


from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class ContinuousFeedForwardPolicy(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.net = Mlp(hidden_dims=[256, 256],
                       repr_shape=2*size_out, input_shape=size_in)

    def forward(self, obs, goal):
        input = torch.cat([obs, goal], dim=-1)
        out = self.net(input)
        if input.ndim == 3:
            loc, scale = out.chunk(2, dim=-1)  # was dim=1 before RIS
        else:
            loc, scale = out.chunk(2, dim=1)

        return loc, torch.exp(scale)

    def get_dist(self, obs, goal):
        loc, scale = self.forward(obs, goal)
        loc = torch.clamp(loc, -1 + 1e-6, 1 - 1e-6)  # for subgoals
        scale = torch.clamp(scale, 1e-6, 1.0)
        dist = Normal(loc, scale+1e-6)

        tanh_transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist, [tanh_transform])

        return dist

    def get_action(self, obs, goal):

        dist = self.get_dist(obs, goal)
        return torch.clamp(dist.rsample(), -1.0+1e-6, 1.0-1e-6)


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


class ContrastiveQf(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 representation_dim,
                 action_dim,
                 goal_dim,
                 obs_dim=None,
                 repr_norm=False,
                 repr_norm_temp=True,
                 repr_log_scale=None,
                 ):
        super().__init__()

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._goal_dim = goal_dim
        self._representation_dim = representation_dim
        self._repr_norm = repr_norm
        self._repr_norm_temp = repr_norm_temp

        state_dim = self._obs_dim

        self._sa_encoder = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,

        )
        self._g_encoder = Mlp(
            hidden_sizes, representation_dim, goal_dim,

        )
        self._sa_encoder2 = Mlp(
            hidden_sizes, representation_dim, state_dim + self._action_dim,

        )
        self._g_encoder2 = Mlp(
            hidden_sizes, representation_dim, goal_dim,

        )

        if self._repr_norm_temp:
            if repr_log_scale is None:
                self._repr_log_scale = nn.Parameter(
                    torch.zeros(1, requires_grad=True))
            else:
                assert isinstance(repr_log_scale, float)
                self._repr_log_scale = repr_log_scale

    @property
    def repr_norm(self):
        return self._repr_norm

    @property
    def repr_log_scale(self):
        return self._repr_log_scale

    def _compute_representation(self, obs, action, hidden=None):
        # The optional input hidden is the image representations. We include this
        # as an input for the second Q value when twin_q = True, so that the two Q
        # values use the same underlying image representation.
        if hidden is None:
            state = obs[:, :self._obs_dim]
            goal = obs[:, self._obs_dim:]
        else:
            state, goal = hidden

        if hidden is None:
            sa_repr = self._sa_encoder(torch.cat([state, action], dim=-1))
            g_repr = self._g_encoder(goal)
        else:
            sa_repr = self._sa_encoder2(torch.cat([state, action], dim=-1))
            g_repr = self._g_encoder2(goal)

        if self._repr_norm:
            sa_repr = sa_repr / torch.norm(sa_repr, dim=1, keepdim=True)
            g_repr = g_repr / torch.norm(g_repr, dim=1, keepdim=True)

            if self._repr_norm_temp:
                sa_repr = sa_repr / torch.exp(self._repr_log_scale)

        return sa_repr, g_repr, (state, goal)

    def forward(self, obs, action, repr=False):
        sa_repr, g_repr, hidden = self._compute_representation(
            obs, action)
        outer = torch.bmm(sa_repr.unsqueeze(
            0), g_repr.permute(1, 0).unsqueeze(0))[0]
        sa_repr_norm = torch.norm(sa_repr, dim=-1)
        g_repr_norm = torch.norm(g_repr, dim=-1)

        if repr:
            return outer, sa_repr, g_repr, sa_repr_norm, g_repr_norm
        else:
            return outer


class StableContrastiveRL:
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
    ):
        super().__init__()
        self.policy = policy
        self.qf = qf
        self.entropy_coefficient = entropy_coefficient
        self.adaptive_entropy_coefficient = entropy_coefficient is None
        self.target_entropy = target_entropy
        self.bc_coef = bc_coef

        self.hard_negs = False
        self.hn_simple = False
        self.hn_close = False

        if self.adaptive_entropy_coefficient:

            self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=lr,
            )

        self.qf_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=lr,
        )
        self.device = device

    def get_action(self, state, goal):
        return self.policy.get_action(state, goal)

    def save(self, folder, save_optims=False):
        torch.save(self.policy.state_dict(),		 folder + "policy.pth")
        torch.save(self.qf.state_dict(),   folder + "qf.pth")

        if save_optims:
            torch.save(self.policy_optimizer.state_dict(),
                       folder + "policy_opti.pth")
            torch.save(self.qf_optimizer.state_dict(), folder + "qf_opti.pth")
            torch.save(self.alpha_optimizer.state_dict(),
                       folder + "alpha_opti.pth")

    def load(self, folder, load_optims=False):
        self.policy.load_state_dict(torch.load(
            folder+"policy.pth", map_location=self.device, weights_only=True))
        self.qf.load_state_dict(torch.load(
            folder+"qf.pth", map_location=self.device, weights_only=True))

        if load_optims:
            self.policy_optimizer.load_state_dict(torch.load(
                folder+"policy_opti.pth", map_location=self.device, weights_only=True))
            self.qf_optimizer.load_state_dict(torch.load(
                folder+"qf_opti.pth", map_location=self.device, weights_only=True))
            self.alpha_optimizer.load_state_dict(torch.load(
                folder+"alpha_opti.pth", map_location=self.device, weights_only=True))

    def update(self, buffer, batch_size):

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

        new_goal = transitions['future_goal'].squeeze(
            1).as_subclass(torch.Tensor)
        batch_size = obs.shape[0]
        goal = transitions['desired_goal'].squeeze(1).as_subclass(torch.Tensor)

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

        if self.hard_negs:
            # "hard close" negatives
            similarity_scores = torch.norm(
                sa_repr.unsqueeze(1) - g_repr.unsqueeze(0), dim=-1)
            dist_norm = 1.0 - similarity_scores / \
                (similarity_scores.max() + 1e-8)
            if self.hn_close:
                dist_norm = 1.0 - dist_norm
            alpha = 0.5
            if self.hn_simple:
                # stable and simple approach
                hardness_weights = 1.0 + alpha * dist_norm
            else:
                # more aggressive approach
                k = 10
                hardness_weights = 1 + alpha * \
                    (torch.sigmoid(k * (dist_norm - 0.5)))  # softmax

            # avg_dig = hardness_weights.mean().detach().cpu().item()
            hardness_weights.fill_diagonal_(1)

        # decrease the weight of negative term to 1 / (B - 1)
        qf_loss_weights = torch.ones(
            (batch_size, batch_size), device=self.device) / (batch_size - 1)
        qf_loss_weights[torch.arange(batch_size, device=self.device), torch.arange(
            batch_size, device=self.device)] = 1
        # Apply hardness weights to negatives
        if self.hard_negs:
            qf_loss_weights *= hardness_weights
        qf_loss_weights_sum = qf_loss_weights.sum()

        qf_loss = self.qf_criterion(logits, I)
        qf_loss *= qf_loss_weights

        qf_loss = torch.mean(qf_loss)
        loss_dict["qf_loss"] = qf_loss

        # Compute distances
        positive_distances = torch.norm(
            sa_repr - g_repr, dim=-1)  # Diagonal (aligned pairs)
        negative_distances = torch.norm(
            sa_repr.unsqueeze(1) - g_repr.unsqueeze(0), dim=-1)
        negative_distances = negative_distances[~torch.eye(
            batch_size, dtype=bool)]
        separation_ratio = positive_distances.mean() / negative_distances.mean()
        loss_dict["sep_rat"] = separation_ratio

        """
        Policy and Alpha Loss
        """

        random_goal = goal
        obs_goal = torch.cat([obs, random_goal], -1)
        dist = self.policy.get_dist(obs, random_goal)

        sampled_action = dist.rsample()
        log_prob = dist.log_prob(sampled_action).mean(dim=-1)

        if self.adaptive_entropy_coefficient:
            alpha_loss = -(self.log_alpha.exp() * (
                log_prob + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros((1,), device=self.device)
            alpha = self.entropy_coefficient

        loss_dict["alpha_loss"] = alpha_loss

        q_action = self.qf(obs_goal, sampled_action)

        actor_q_loss = alpha * log_prob - torch.diag(q_action)

        assert 0.0 <= self.bc_coef <= 1.0
        orig_action = action

        train_mask = ((orig_action * 1E8 % 10)[:, 0] != 4).float()

        gcbc_loss = -train_mask * dist.log_prob(orig_action).mean(dim=-1)
        # gcbc_val_loss = -(1.0 - train_mask) * dist.log_prob(orig_action)

        actor_loss = self.bc_coef * gcbc_loss + \
            (1 - self.bc_coef) * actor_q_loss

        actor_loss = torch.mean(actor_loss)

        loss_dict["gcbc_loss"] = gcbc_loss.mean()
        loss_dict["actor_q_loss"] = actor_q_loss.mean()
        """
        Optimization.
        """
        if self.adaptive_entropy_coefficient:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        return loss_dict
