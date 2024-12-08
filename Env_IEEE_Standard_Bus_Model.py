from typing import Tuple
import gym
import numpy as np
from IEEE_Standard_Bus_Model import IEEE_Standard_Bus_Model
class Env_IEEE_Standard_Bus_Model(gym.Env):
    def __init__(self, N=68) -> None:
        self.power_model = IEEE_Standard_Bus_Model(N=N)
        self.action_num = self.power_model.N_bus
        self.obs_num = self.power_model.N_bus*2
        self.N_bus = self.power_model.N_bus
        self.alpha = np.ones([self.power_model.N_bus,1], dtype=np.float32)
        self.beta = np.ones([self.power_model.N_bus,1], dtype=np.float32)
        self.reward_max = 10
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, dict]:
        o = self.power_model.step(action)
        omega = o[self.N_bus:self.N_bus*2]
        loss = self.alpha*omega*omega+self.beta*action*action
        reward = self.reward_max-loss
        return o, reward, 0, 0, {}

    def reset(self) -> Tuple[float, dict]:
        o = self.power_model.reset()
        return o, {}
    @property
    def action_space(self):
        pass
    @property
    def observation_space(self):
        pass

