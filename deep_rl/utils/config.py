#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
import datetime
import argparse
import torch
import yaml
import os
import json

class Config:
    DEVICE = torch.device('cuda')
    particles = 64

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.target_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.logger = None
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False
        self.particles = 64
        self.hyper = False

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name
    
    def generate_log_handles(self):
        ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        self.log_handle = './tf_log/%s-%s/log.txt' % (self.tag, ts)
        self.tf_log_handle = './tf_log/%s-%s' % (self.tag, ts)
        if not os.path.exists(self.tf_log_handle):
            os.makedirs(self.tf_log_handle)

    def save_config_to_yaml(self):
        items = vars(self)
        itemstr = {k: str(v) for (k, v) in items.items()}
        save_fn = '/config.json'
        with open(self.tf_log_handle+save_fn, 'w') as f:
            json.dump(itemstr, f, indent=2)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        self.save_config_to_yaml()

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
