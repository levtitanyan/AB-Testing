"""
  Run this file first. Uses loguru for logging outputs instead of print().
"""

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.
    Defines required methods for any subclass.
    """

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """
        Select an arm using the strategy defined by the algorithm.

        Returns:
            int: Index of the selected arm.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update internal estimates based on received reward.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the experiment over a set number of trials.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Log final results and save data to CSV:
            - Average reward
            - Cumulative regret
        """
        pass


#--------------------------------------#

class Visualization:
    """
    Visualization class for plotting learning and performance of bandit algorithms.
    """

    def __init__(self):
        self.eg_data = pd.read_csv('epsilon_greedy_rewards.csv')
        self.ts_data = pd.read_csv('thompson_sampling_rewards.csv')

    def plot1(self):
        """
        Plot average reward convergence for EpsilonGreedy and ThompsonSampling.
        """
        for df, label in [(self.eg_data, 'EpsilonGreedy'), (self.ts_data, 'ThompsonSampling')]:
            rewards = df['Reward'].values
            cumulative_avg = rewards.cumsum() / (np.arange(len(rewards)) + 1)
            plt.plot(cumulative_avg, label=label)

        plt.title('Average Reward Convergence')
        plt.xlabel('Trial')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self):
        """
        Plot cumulative reward and regret for both algorithms.
        """
        df = pd.concat([self.eg_data, self.ts_data])
        for algo in ['EpsilonGreedy', 'ThompsonSampling']:
            subset = df[df['Algorithm'] == algo]
            cumulative_reward = np.cumsum(subset['Reward'])
            regret = np.cumsum(np.max([1, 2, 3, 4]) - subset['Reward'])

            plt.plot(cumulative_reward, label=f'{algo} - Reward')
            plt.plot(regret, label=f'{algo} - Regret')

        plt.title('Cumulative Reward vs Regret')
        plt.xlabel('Trial')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()


#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Implements the Epsilon-Greedy bandit algorithm.
    """

    def __init__(self, p, epsilon=0.1, trials=20000):
        """
        Args:
            p (list): True mean rewards for each arm.
            epsilon (float): Initial exploration rate.
            trials (int): Number of trials to run.
        """
        self.p = p
        self.k = len(p)
        self.epsilon = epsilon
        self.trials = trials
        self.counts = [0] * self.k
        self.estimates = [0] * self.k
        self.rewards = []
        self.data = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        """
        Select an arm based on epsilon-greedy strategy.

        Returns:
            int: Index of selected arm.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        else:
            return np.argmax(self.estimates)

    def update(self, chosen_arm, reward):
        """
        Update the estimate for the selected arm using incremental mean.

        Args:
            chosen_arm (int): Arm index.
            reward (float): Observed reward.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.estimates[chosen_arm]
        self.estimates[chosen_arm] += (1 / n) * (reward - value)

    def experiment(self):
        """
        Run epsilon-greedy algorithm for specified number of trials.
        """
        for t in range(1, self.trials + 1):
            self.epsilon = 1 / t  # decay epsilon
            arm = self.pull()
            reward = np.random.randn() + self.p[arm]
            self.update(arm, reward)
            self.rewards.append(reward)
            self.data.append((arm, reward, 'EpsilonGreedy'))

    def report(self):
        """
        Save results to CSV and log average reward and regret.
        """
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('epsilon_greedy_rewards.csv', index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"[EpsilonGreedy] Average reward: {avg_reward}")
        logger.info(f"[EpsilonGreedy] Cumulative regret: {regret}")


#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Implements the Thompson Sampling bandit algorithm with Gaussian rewards.
    """

    def __init__(self, p, trials=20000):
        """
        Args:
            p (list): True mean rewards for each arm.
            trials (int): Number of trials to run.
        """
        self.p = p
        self.k = len(p)
        self.trials = trials
        self.alpha = [1] * self.k
        self.beta = [1] * self.k
        self.rewards = []
        self.data = []

    def __repr__(self):
        return f"ThompsonSampling()"

    def pull(self):
        """
        Sample from normal distributions for each arm and choose the best.

        Returns:
            int: Index of selected arm.
        """
        samples = [np.random.normal(self.alpha[i] / (self.beta[i] + 1e-5), 1 / (self.beta[i] + 1e-5))
                   for i in range(self.k)]
        return np.argmax(samples)

    def experiment(self):
        """
        Run Thompson Sampling for the given number of trials.
        """
        for _ in range(self.trials):
            arm = self.pull()
            reward = np.random.randn() + self.p[arm]
            self.update(arm, reward)
            self.rewards.append(reward)
            self.data.append((arm, reward, 'ThompsonSampling'))

    def update(self, chosen_arm, reward):
        """
        Update posterior estimates of the selected arm.

        Args:
            chosen_arm (int): Arm index.
            reward (float): Observed reward.
        """
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1

    def report(self):
        """
        Save results to CSV and log average reward and regret.
        """
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('thompson_sampling_rewards.csv', index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"[ThompsonSampling] Average reward: {avg_reward}")
        logger.info(f"[ThompsonSampling] Cumulative regret: {regret}")


def comparison():
    """
    Run and compare both algorithms. Generate CSVs and plots.
    """
    logger.info("Running EpsilonGreedy...")
    eg = EpsilonGreedy([1, 2, 3, 4])
    eg.experiment()
    eg.report()

    logger.info("Running ThompsonSampling...")
    ts = ThompsonSampling([1, 2, 3, 4])
    ts.experiment()
    ts.report()

    logger.info("Generating visualizations...")
    viz = Visualization()
    viz.plot1()
    viz.plot2()


if __name__ == '__main__':
    logger.debug("Starting comparison...")
    comparison()





