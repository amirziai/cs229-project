import numpy as np


class EpsilonGreedy:
    def __init__(self, n_actions: int, epsilon: float=0.3):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rewards = np.array([0.0] * n_actions)
        self.rewards_cnt = np.array([0.0] * n_actions)

    def choose_action(self, pull_idx: int) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.n_actions))
        else:
            return np.argmax(self.rewards)

    def record_reward(self, reward: float, reward_idx: int) -> None:
        n = self.rewards_cnt[reward_idx]
        self.rewards[reward_idx] = (n * self.rewards[reward_idx] + reward) / (n + 1)
        self.rewards_cnt[reward_idx] += 1


class UCB(EpsilonGreedy):
    def __init__(self, n_actions: int, c: float):
        super().__init__(n_actions, epsilon=0)
        self.c = c

    def choose_action(self, pull_idx: int):
        # if all of the arms have been pulled at least once
        if np.count_nonzero(self.rewards_cnt) == self.n_actions:
            augmented = self.rewards + self.c * np.sqrt(np.log(pull_idx) / self.rewards_cnt)
            return np.argmax(augmented)
        else:
            return np.argmax(np.array(self.rewards) == 0)
