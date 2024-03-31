import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
from torchvision.transforms import Compose, ToPILImage, ToTensor, Grayscale, Resize
import gym
import cv2
import random
from gym import spaces
from gym.wrappers import FrameStack, ResizeObservation

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=[old_space.shape[0], old_space.shape[1], 1], 
                                            dtype=np.uint8)

    def observation(self, observation):
        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(grayscale_observation, axis=0)
	



class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = ResizeObservation(self.env, shape=84)
		self.env = GrayScaleObservation(self.env)
		self.env = FrameStack(self.env, num_stack=4)

		self.env.metadata['render_fps'] = 60

		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = ResizeObservation(self.test_env, shape=84)
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = FrameStack(self.test_env, num_stack=4)

		self.test_env.metadata['render_fps'] = 60
		


		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		# if random.random() < epsilon:
		# 	action = ???
		# else:
		# 	action = ???

		# return action
			if random.random() < epsilon:
				action = np.random.choice(np.arange(action_space.n))
			else:
				with torch.no_grad():
					observation = np.array(observation, dtype=np.float32)
					observation = torch.tensor(observation, device=self.device)

					observation = observation.squeeze(2)  # Squeeze the unnecessary dimension
					q_values = self.behavior_net(observation.unsqueeze(0))
					action = torch.argmax(q_values).item()
			
			return action

		#return NotImplementedError
	
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		state = state.squeeze(2)
		q_values = self.behavior_net(state)
		q_values = q_values.gather(1, action.long())


		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
			q_target = reward + self.gamma * q_next * (1 - done)
		
		criterion = nn.SmoothL1Loss()
		loss = criterion(q_values, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	