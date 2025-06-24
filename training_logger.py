import os
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    def __init__(self, save_dir="training_logs"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reward_img_path = os.path.join(save_dir, f"reward_{timestamp}.png")
        self.loss_img_path = os.path.join(save_dir, f"loss_{timestamp}.png")
        self.rewards = []
        self.losses = []
        self.save_dir = save_dir
        self.timestamp = timestamp

    def update_reward(self, reward):
        self.rewards.append(reward)
        self._plot_rewards()

    def update_loss(self, loss):
        self.losses.append(loss)
        self._plot_losses()

    def _plot_rewards(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards, label='Reward', color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.reward_img_path)
        plt.close()

    def _plot_losses(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, label='Loss', color='red')
        plt.xlabel("Train Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.loss_img_path)
        plt.close()
