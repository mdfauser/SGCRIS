import matplotlib.pyplot as plt
import torch
from core.data_storage import DataStorage
import random


class Plotter:
    def __init__(self, filename):
        self.data = {
            'total_steps': [],
            'success_rate': [],
            'logits_pos': [],
            'logits_neg': [],
            'q_pos_ratio': [],
            'q_neg_ratio': [],
            'cat_acc': [],
            'bin_acc': [],
            'qf_loss': [],
            'alpha_loss': [],
            'gcbc_loss': [],
            'actor_loss': [],
        }
        self.storer = DataStorage(filename)
        self.storer.initialize_json()

    def update(self, update_dict, timesteps, success_rate):
        """Update the data with values from update_dict."""
        self.data['total_steps'].append(timesteps)
        self.data['success_rate'].append(success_rate)
        # if 'logits_pos' in update_dict: self.data['logits_pos'].append(update_dict['logits_pos'].cpu().item())
        # if 'logits_neg' in update_dict: self.data['logits_neg'].append(update_dict['logits_neg'].cpu().item())

        keys_to_update = [
            'logits_pos', 'logits_neg', 'pos_loss', 'neg_loss',
            'actor_loss', 'critic_loss', 'q_pos_ratio', 'q_neg_ratio',
            'cat_acc', 'bin_acc', 'qf_loss', 'alpha_loss', 'gcbc_loss'
        ]
        for key in keys_to_update:
            if key in update_dict:
                value = update_dict[key]
                if torch.is_tensor(value):
                    self.data[key].append(value.cpu().item())
                else:
                    self.data[key].append(value)

    def store_only(self, timestep, success_rate, success_rate2=None, sep_rat=None):
        if sep_rat is not None:
            sep_rat = sep_rat.cpu().item()
            sep_rat = float(f"{sep_rat:.3f}")
        self.storer.append_to_json(
            timestep, success_rate, success_rate2, sep_rat)

    def plot_success_rate(self):
        """Plot timesteps vs success rate."""
        plt.figure()
        plt.plot(self.data['total_steps'],
                 self.data['success_rate'], label='Success Rate')
        plt.xlabel('Timesteps')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Over Total Steps')
        plt.legend()
        plt.grid()
        plt.savefig("suc_fetchPush_SGCRL_base.png")
        # plt.show()

    def plot_logits(self):
        """Plot timesteps vs logits_pos and logits_neg."""
        min_length = min(len(self.data['total_steps']), len(
            self.data['logits_pos']), len(self.data['logits_neg']))

        # Trim longer lists randomly to match the shortest
        if len(self.data['total_steps']) > min_length:
            indices_to_keep = sorted(random.sample(
                range(len(self.data['total_steps'])), min_length))
            self.data['total_steps'] = [self.data['total_steps'][i]
                                        for i in indices_to_keep]

        if len(self.data['logits_pos']) > min_length:
            indices_to_keep = sorted(random.sample(
                range(len(self.data['logits_pos'])), min_length))
            self.data['logits_pos'] = [self.data['logits_pos'][i]
                                       for i in indices_to_keep]

        if len(self.data['logits_neg']) > min_length:
            indices_to_keep = sorted(random.sample(
                range(len(self.data['logits_neg'])), min_length))
            self.data['logits_neg'] = [self.data['logits_neg'][i]
                                       for i in indices_to_keep]

        plt.figure()
        plt.plot(self.data['total_steps'], self.data['logits_pos'],
                 label='Logits Pos', color='blue')
        plt.plot(self.data['total_steps'], self.data['logits_neg'],
                 label='Logits Neg', color='red')
        plt.xlabel('Timesteps')
        plt.ylabel('Logits')
        plt.title('Timesteps vs Logits')
        plt.legend()
        plt.grid()
        plt.savefig("log_fetchPush_SGCRL_base.png")
        # plt.show()

    def plot_losses(self):
        plt.figure()

        plt.xlabel('Timesteps')
        plt.ylabel('Loss')
        plt.title('Timesteps vs Losses')
        plt.legend()
        plt.grid()
        # plt.show()
