#importing modules
from cmath import inf
import copy
import os
import numpy as np
import random
import torch
import gym
import gym_super_mario_bros
from collections import deque
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torch import nn
from torchvision import transforms

#Environment Preprocessing
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__()
        self._skip = skip

    def step(self, action):
        '''
        Args:action
        Purpose: Repeacts action and sum rewards
        Return: 
        '''
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs,total_reward,done,info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = Box(
            low = 0,
            high = 255,
            shape = self.observation_space.shape[:2],
            dtype = np.uint8
        )
    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))



class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low = 0,
            high = 255,
            shape = obs_shape,
            dtype = np.uint8
        )

    def observtion(self, observation):
        transformations = transforms.Compose(
            [transforms.Resize(self.shape), transforms.Normalize(0, 255)]
        )
        return transformations(observation).squeeze(0)
    
