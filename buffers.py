from typing import Dict


import torch

class EpisodicBuffer(torch.nn.Module):
    """
    Episodic replay buffer, storing experiences from multiple episodes.
    Note that this buffer is saved when the parent nn.Module is saved!
    """

    def __init__(self, num_episodes=1000, max_episode_length=250, shapes={}):
        super().__init__()

        self.keys = []
        for key, value in shapes.items():
            self.keys.append(key)
            tensor = torch.zeros((num_episodes, max_episode_length, *value))
            self.register_buffer(key, tensor)

        idx = torch.zeros((num_episodes,), dtype=torch.int)
        episode_count = torch.zeros((num_episodes,), dtype=torch.int)

        self.register_buffer("episode_count", episode_count)
        self.register_buffer("idx", idx)
        self.register_buffer("episode", torch.tensor(0))
        self.register_buffer("max_episode", torch.tensor(0))
        self.register_buffer("episode_counter", torch.tensor(1))
        self.register_buffer("num_episodes", torch.tensor(num_episodes))
        self.register_buffer("max_episode_length", torch.tensor(max_episode_length))

        self.is_new_episode = False
        self.episode_count[0] = 1

    def append(self, data):
        self.is_new_episode = False

        episode = self.episode
        idx = self.idx[episode]
        for key, value in data.items():
            getattr(self, key)[episode, idx] = value

        self.idx[episode] = idx + 1

        # if self.idx[episode] >= self.max_episode_length:
        #     self.new_episode()

    def new_episode(self):
        self.is_new_episode = True
        self.episode = self.episode + 1
        if self.episode >= self.num_episodes:
            self.episode = torch.tensor(0, device=self.episode.device)
        else:
            self.max_episode = self.max_episode + 1  # keep counting

        self.idx[self.episode] = torch.tensor(
            0, device=self.idx.device
        )  # reset idx in episode
        self.episode_counter = self.episode_counter + 1
        self.episode_count[self.episode] = self.episode_counter

    def sample(self, batch_size, trajectory_len, to_device=None, keys=None):
        """
        Sample multiple contiguous trajectories from the buffer.
        """

        if keys is None:
            keys = self.keys

        if to_device is None:
            to_device = self.idx.device

        available_episode = torch.arange(0, self.idx.shape[0], device=self.idx.device)[
            self.idx >= trajectory_len
        ]
        if (
            available_episode.size(0) == 0
        ):  # No episode can handle the desired trajectory length
            return None

        sel = torch.randint(
            0, available_episode.shape[0], (batch_size,), device=self.idx.device
        )

        episodes = available_episode[sel]
        start = (
            (
                torch.rand((batch_size,), device=self.idx.device)
                * (self.idx[episodes] - trajectory_len + 1)
            )
            .int()
            .reshape((batch_size, 1))
        )
        episodes = episodes.reshape((-1, 1))
        indices = start + torch.arange(
            0, trajectory_len, device=self.idx.device
        ).reshape((1, trajectory_len))

        return_dict = {}


        all_indices = torch.arange(self.max_episode_length * self.num_episodes).reshape(
            self.num_episodes, self.max_episode_length
        )

        flattened_indices = all_indices[episodes, indices]

        return_dict["flattened_indices"] = flattened_indices.to(to_device)
        for key in keys:
            return_dict[key] = getattr(self, key)[episodes, indices].to(to_device)
        return return_dict
    
    def sample_last_transition(self, batch_size, to_device=None, keys=None):
        """
        Sample only the last transition of episodes to increase the likelihood of successful labels.
        """
        if keys is None:
            keys = self.keys

        if to_device is None:
            to_device = self.idx.device

        # Find episodes with at least one transition (non-empty)
        available_episode = torch.arange(0, self.idx.shape[0], device=self.idx.device)[self.idx > 0]
        if available_episode.size(0) == 0:  # No episodes available for sampling
            return None

        # Randomly select episodes for the batch
        sel = torch.randint(0, available_episode.shape[0], (batch_size,), device=self.idx.device)
        episodes = available_episode[sel].reshape((-1, 1))

        # Get the last index of each selected episode
        last_indices = self.idx[episodes].reshape((-1, 1)) - 1  # `self.idx` stores the lengths

        # Prepare return dictionary
        return_dict = {}

        # Get flattened indices for direct access
        all_indices = torch.arange(self.max_episode_length * self.num_episodes).reshape(
            self.num_episodes, self.max_episode_length
        )
        flattened_indices = all_indices[episodes, last_indices]

        # Add indices and data for specified keys
        return_dict["flattened_indices"] = flattened_indices.to(to_device)
        for key in keys:
            return_dict[key] = getattr(self, key)[episodes, last_indices].to(to_device)

        return return_dict



class FunctionBuffer:
    """
    Creates a pseudo buffer from a .sample function.
    """

    def __init__(self, sample_function, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_function = sample_function

    def sample(self, batch_size, trajectory_len, to_device, keys) -> Dict:
        transitions = self.sample_function(batch_size, trajectory_len, to_device, keys)
        # transitions_filtered = {key: transitions[key].to(to_device) for key in keys}
        return transitions