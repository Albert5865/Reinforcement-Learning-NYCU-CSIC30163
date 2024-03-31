import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym 
import cv2 as cv
import random

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode="human")
		self.env = ResizeObservationWrapper(self.env)
		
		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="human")
		self.test_env = ResizeObservationWrapper(self.test_env)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
		self.epsilon = 1.0
		self.eps_min = 0.1
		self.eps_decay = 1000000

	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		if eval:
			with torch.no_grad():
				action, value, logp, _ = self.net(torch.tensor(observation[None,:,:,:]).to(self.device), eval=True)
				
		else:
			action, value, logp, _ = self.net(torch.tensor(observation[None,:,:,:]).to(self.device))
			action = torch.tensor([1 if random.random() < self.epsilon else self.env.action_space.sample() if random.random() < self.epsilon else a.item() for a in action])
				
		return action, value, logp

	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				action, value, logp, entropy = self.net(ob_train_batch)
				
				# calculate policy loss
				ac_train_batch = ac_train_batch.squeeze()
				logp = logp[range(logp.shape[0]), ac_train_batch]
				logp_pi_train_batch = logp_pi_train_batch[range(logp.shape[0]), ac_train_batch]
				ratio = torch.exp(logp) / (torch.exp(logp_pi_train_batch) + 1e-8)
				adv_train_batch = (adv_train_batch - torch.mean(adv_train_batch)) / (torch.std(adv_train_batch) + 1e-8)
				surrogate_loss = torch.min(ratio * adv_train_batch, torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch)

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)

				# calculate total loss
				loss = - surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				loss = loss.mean()

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()
				total_surrogate_loss += surrogate_loss.mean().item()
				total_v_loss += v_loss.mean().item()
				total_entropy += entropy.mean().item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	
class ResizeObservationWrapper(gym.Wrapper):
    def __init__(self, env, shape=(84, 84, 4)):
        super(ResizeObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.frame_buffer = np.zeros(shape, dtype=np.uint8)

    def reset(self):
        self.frame_buffer = np.zeros_like(self.frame_buffer)
        observation, info = self.env.reset()
        return self.process_observation(observation), info

    def step(self, action):
        observation, reward, terminate, truncate, info = self.env.step(action)
        return self.process_observation(observation), reward, terminate, truncate, info

    def process_observation(self, observation):
        observation = self.convert_to_grayscale(observation)
        observation = self.resize_image(observation)
        self.frame_buffer = np.concatenate([self.frame_buffer[:,:,1:], observation], axis=2)
        return np.transpose(self.frame_buffer, (2, 0, 1))

    def convert_to_grayscale(self, rgb):
        gray = np.mean(rgb, axis=2, keepdims=True).astype(np.uint8)
        return gray
    
    def resize_image(self, image):
        return cv.resize(image, (84, 84), interpolation=cv.INTER_AREA)[:,:,None]

