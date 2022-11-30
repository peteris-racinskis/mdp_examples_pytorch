from typing import Tuple
import pickle
import numpy as np
import math
from utils import visualize_model

EPS=1e-9
ENV="CartPole-v1"
TRAIN=False
# TRAIN=True
OUTNAME=f"qmatrix_models/QMatrix_new"
INNAME=f"qmatrix_models/QMatrix"

class QMatrix():

    def __init__(self, bounds: Tuple[Tuple], discrete_steps: Tuple[int], action_size=2, preload: dict = None):
        if preload is None:
            self.steps = np.array([s for s in discrete_steps], dtype=int)
            self.ranges = np.array([h+EPS-(l-EPS) for l,h in bounds])
            self.lows = np.array([l-EPS for l,_ in bounds], dtype=int)
            self._get_initial_matrix(action_size)
        else:
            for k,v in preload:
                self.__dict__[k] = v
    
    def _get_initial_matrix(self, action_space_size):
        self.q_matrix = np.zeros(tuple(self.steps) + (action_space_size,))
    
    def _state_indices(self, continuous_value: np.ndarray, slice=slice(None)):
        return ( ( (continuous_value[slice] - self.lows[slice]) * self.steps[slice] ) // self.ranges[slice] ).astype(int)

    def __getitem__(self, idx):
        _idx = np.array(idx)
        replace_slice = slice(min(len(self.steps), len(_idx)))
        _idx[replace_slice] = self._state_indices(_idx, replace_slice)
        return self.q_matrix[tuple(_idx.astype(int))]

    def expected_next_reward(self, state, beta=0.98):
        return beta * np.max(self[state])

    def update_reward(self, expected_reward, state, action, lr=0.001):
        idx = tuple(self._state_indices(state)) + (action,)
        self.q_matrix[idx] = (1 - lr) * self.q_matrix[idx] + lr * expected_reward

    def get_action(self, state):
        return np.argmax(self[state])

    def randomized_action(self, action_space, state, exploration_rate=0.001):
        if np.random.rand(1) < exploration_rate:
            return action_space.sample(), True
        return self.get_action(state), False
    
    def save(self, filename=f"{OUTNAME}.pickle"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename=f"{INNAME}.pickle"):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

def rate_schedule(ep, decay, min_rate = 0.01, initial = 1.0):
    return max( min_rate, min( initial, math.exp(-decay*ep) ) )

def train_matrix(model: QMatrix, env, save=True):
    for episode in range(4000):
        lr = rate_schedule(episode, 0.001)
        exploration_rate = rate_schedule(episode, 0.001)
        discount = 0.98
        done = False
        state_new, _ = env.reset()
        timestep = 0
        while not done and timestep < 2000:
            timestep += 1
            state_old = state_new
            action, _ = model.randomized_action(env.action_space, state_old, exploration_rate)
            state_new, reward, done, _, __ = env.step(action)
            expected_reward = reward + model.expected_next_reward(state_new, discount)
            model.update_reward(expected_reward, state_old, action, lr)
        if episode % 50 == 0:
            print(f"Training episode {episode} survived for {timestep} steps")     
    if save:
        model.save()


if __name__ == "__main__":
    if TRAIN:
        import gym
        env = gym.make(ENV)
        QSHAPE=(4,8,4,8)
        LIMITS=(
            (env.observation_space.low[0], env.observation_space.high[0]),
            (-10,10),
            (env.observation_space.low[2], env.observation_space.high[2]),
            (-5,5),
        )
        qmat = QMatrix(LIMITS, QSHAPE, env.action_space.n)
        train_matrix(qmat, env, save=TRAIN)
        env.close()
    else:
        qmat = QMatrix.load(f"{INNAME}.pickle")
        visualize_model(qmat)