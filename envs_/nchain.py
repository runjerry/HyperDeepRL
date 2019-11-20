import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from torch.distributions import Normal

class NChainEnv(gym.Env):
    """n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=50, stochastic=False):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.state = 1  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        self.actions = [np.random.choice(2, 1)[0] for i in range(self.n)]
        self.steps = 0
        self.state1_reward = Normal(torch.tensor([0.]), torch.tensor([1.]))
        self.stateN_reward = Normal(torch.tensor([1.]), torch.tensor([1.]))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        if self.actions[self.state]:
            action = not action  # agent slipped, reverse action taken
        # handle actions
        
        if action == 1:
            if self.state >= self.n-1:
                self.state += 0
            else:
                self.state += 1
                reward -= 1./self.n  # negative reward for moving right 
        else:
            if self.state <= 0:
                self.state -= 0
            else:
                self.state -= 1

        # handle reward
        if self.state == 0:
            reward += self.state1_reward.sample().item()  # small reward for 1st state
        elif self.state == self.n - 1:
            reward += self.stateN_reward.sample().item()  # large reward for end state
        else:
            reward = 0.

        self.steps += 1

        if self.steps >= self.n + 9:
            done = True
        else:
            done = False
        
        return self.state, reward, done, {}

    def reset(self, state=1):
        self.state = state
        self.steps = 0
        return self.state