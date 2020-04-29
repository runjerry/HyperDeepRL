#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
#
# DQN with a generative dynamics model
#
# * In essence, a value-based posterior sampling RL appoach
# * The generative dynamics model captures the agent's uncertainty of the MDP
# * This dynamics model is explicitly used in Q-learning, for better exploration
# * The DQN part inherits from DQN_dist_svgd_agent


import torch.nn.functional as F
from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import torchvision
import torch.autograd as autograd
import sys
import tqdm


def gen_ensemble_tensor(tensor, n_particles):
    ones_mask = torch.ones(tensor.dim()).long().tolist()
    tensor = tensor.unsqueeze(0).repeat(
        n_particles, *ones_mask)  # z.shape[1] = particles
    if tensor.dim() > 2 and tensor.size(2) == 1:  # DM lab has incompatible sizing with gym
        tensor = tensor.squeeze(2)
    return tensor


class DynamicsDQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.k = np.random.choice(config.particles, 1)[0]
        self.mdp_update = False
        self.episode_steps = 0
        self.update_steps = 0

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            state = config.state_normalizer(self._state)
            q_values = self._network(state)
            particle_max = q_values.argmax(-1)
            abs_max = q_values.max(2)[0].argmax()
            q_max = q_values[abs_max]

        q_max = to_np(q_max).flatten()
        q_var = to_np(q_values.var(0))
        q_mean = to_np(q_values.mean(0))
        q_random = to_np(q_values[self.k])

        # we want a best action to take, as well as an action for each particle
        #model_action_prob = 1.0
        # if self._total_steps < config.exploration_steps:
        #    model_action_prob = np.random.rand() # 0.5 prob of taking a model steps during exploration

        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_max))
            actions_log = np.random.randint(
                0, len(q_max), size=(config.particles, 1))
        else:
            # action = np.argmax(q_max)  # Max Action
            # action = np.argmax(q_mean)  # Mean Action
            action = np.argmax(q_random)  # Random Head Action
            actions_log = to_np(particle_max)

        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self.mdp_update = True
            self.update_steps = self.episode_steps
            self.episode_steps = 0
            self._network.sample_model_seed()
            self.k = np.random.choice(config.particles, 1)[0]
            if self._task.record:
                self._task.record_or_not(info)

        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()

        entry = [self._state[0], action, reward[0],
                 next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        self.episode_steps += 1
        return entry


class Dynamics_DQN_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DynamicsDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.mdp = config.mdp_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.mdp_optimizer = config.optimizer_fn(self.mdp.parameters())
        self.alpha_schedule = BaselinesLinearSchedule(
            config.alpha_anneal, config.alpha_final, config.alpha_init)
        self.mdp_alpha_schedule = BaselinesLinearSchedule(
            config.alpha_anneal, config.alpha_final, 5.0)
        self.actor.set_network(self.network)

        self.total_steps = 0
        self.network.sample_model_seed()
        self.target_network.set_model_seed(self.network.model_seed)
        self.head = np.random.choice(config.particles, 1)[0]
        self.save_net_arch_to_file()
        print(self.network)

    def save_net_arch_to_file(self):
        network_str = str(self.network)
        save_fn = '/model_arch.txt'
        save_dir = self.config.tf_log_handle
        with open(save_dir+save_fn, 'w') as f:
            f.write(network_str+'\n\n')

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        # self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state, read_only=True)
        action = self.network.predict_action(state, pred='mean', to_numpy=True)
        action = np.array([action])
        # self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        if not torch.equal(self.network.model_seed['value_z'],
                           self.target_network.model_seed['value_z']):
            self.target_network.set_model_seed(self.network.model_seed)

        # rollout env
        transitions = self.actor.step()
        # experiences = []
        pos_experiences = []
        neg_experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)

            if reward > 0:
                pos_experiences.append(
                    [state, action, reward, next_state, done])
            else:
                neg_experiences.append(
                    [state, action, reward, next_state, done])

            # experiences.append([state, action, reward, next_state, done])
        # self.replay.feed_batch(experiences)

        if len(pos_experiences) > 0:
            self.replay.feed_pos_batch(pos_experiences)
        if len(neg_experiences) > 0:
            self.replay.feed_neg_batch(neg_experiences)

        if self.total_steps == self.config.exploration_steps+1:
            print('pure exploration finished')
            self.train_mdp(train_steps=1000)

        # models training
        if self.total_steps > self.config.exploration_steps:
            alpha = self.alpha_schedule.value(self.total_steps)

            # mdp training
            self.train_mdp()
            # if self.actor.mdp_update:
            # self.train_mdp(alpha=alpha)
            # self.actor.mdp_update = False

            # sample training exp from the replay buffer
            # experiences = self.replay.sample()
            experiences = self.replay.sample_balanced()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            next_states = tensor(next_states)
            next_states = gen_ensemble_tensor(
                next_states, self.config.particles)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            rewards = gen_ensemble_tensor(rewards, self.config.particles)

            # actions = tensor(actions).long()
            sample_z = self.network.sample_model_seed(return_seed=True)
            self.mdp.set_model_seed(sample_z)

            # repeat states batch to get extended states batch
            ones_mask = np.ones(states.ndim - 1, dtype=np.int32).tolist()
            states_ext = np.tile(states, [2, ] + ones_mask)

            # sample random actions for extended states
            perturb = np.random.randint(
                1, self.config.action_dim, size=actions.shape[0])
            actions_rand = np.remainder(
                actions + perturb, self.config.action_dim)
            actions_ext = np.concatenate([actions, actions_rand], axis=0)

            actions_ext = tensor(actions_ext).long()
            actions_rand = tensor(actions_rand).long()

            # get extended rewards and next_states
            next_states_rand, rewards_rand = self.mdp(
                states, actions_rand)  # [particles, batch, d_output]
            next_states_rand = next_states_rand.detach()
            next_states_ext = torch.cat([next_states, next_states_rand], dim=1)
            rewards_rand = rewards_rand.detach()
            rewards_ext = torch.cat([rewards, rewards_rand.squeeze(-1)], dim=1)

            # get target q values
            q_next = self.target_network(
                next_states_ext, ensemble_input=True, seed=sample_z).detach()
            if self.config.double_q:
                # [particles, batch, actions]
                q = self.network(
                    next_states_ext, ensemble_input=True, seed=sample_z)
                best_actions = torch.argmax(
                    q, dim=-1, keepdim=True)  # [particles, batch, 1]
                # [particles, batch]
                q_next = q_next.gather(-1, best_actions).squeeze(-1)
            else:
                q_next = q_next.max(-1)[0]
            q_target = self.config.discount * \
                q_next * (1 - terminals.repeat(2))
            q_target.add_(rewards_ext)

            # get main Q values
            q = self.network(states_ext, seed=sample_z)
            batch_indices = range_tensor(self.replay.batch_size * 2)
            q = q[:, batch_indices, actions_ext]  # [particles, batch]

            q = q.transpose(0, 1).unsqueeze(-1)  # [batch, particles, 1]
            q_target = q_target.transpose(
                0, 1).unsqueeze(-1)  # [batch, particles, 1]

            # [batch, particles/2, 1]
            q_i, q_j = torch.split(q, self.config.particles//2, dim=1)
            # [batch, particles/2, 1]
            qi_target, qj_target = torch.split(
                q_target, self.config.particles//2, dim=1)

            td_loss = (qj_target.detach() - q_j).pow(2).mul(0.5)
            q_grad = autograd.grad(td_loss.sum(), inputs=q_j)[0]
            q_grad = q_grad.unsqueeze(2)  # [batch, particles//2. 1, 1]

            qi_eps = q_i + torch.rand_like(q_i) * 1e-8
            qj_eps = q_j + torch.rand_like(q_j) * 1e-8

            # kappa, grad_kappa: [batch, particles/2, particles/2, 1]
            kappa, grad_kappa = batch_rbf_xy(qj_eps, qi_eps)
            kappa = kappa.unsqueeze(-1)

            # [batch, particles/2, particles/2, 1]
            kernel_logp = torch.matmul(kappa.detach(), q_grad)  # [n, 1]
            # [batch, particles/2, 1]
            svgd = (kernel_logp + alpha * grad_kappa).mean(1)

            self.optimizer.zero_grad()
            autograd.backward(q_i, grad_tensors=svgd.detach())

            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data *= 1./config.particles

            if self.config.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip)

            with config.lock:
                self.optimizer.step()
            self.logger.add_scalar('td_loss', td_loss.mean(), self.total_steps)
            self.logger.add_scalar(
                'q_grad_kappa', grad_kappa.mean(), self.total_steps)
            self.logger.add_scalar('q_kappa', kappa.mean(), self.total_steps)

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def train_mdp(self, alpha=None, train_steps=None):
        config = self.config
        if alpha is None:
            alpha = self.mdp_alpha_schedule.value(self.total_steps)
        if train_steps is None:
            train_steps = 1  # self.actor.update_steps
        for trian_iter in range(train_steps):
            experiences = self.replay.sample()
            # experiences = self.replay.sample_balanced()
            states, actions, rewards, next_states, _ = experiences
            states = self.config.state_normalizer(states, read_only=True)
            next_states = self.config.state_normalizer(
                next_states, read_only=True)
            states = tensor(states)
            next_states = tensor(next_states)
            if states.dim() == 3 and states.size(1) == 1:  # DM lab has incompatible sizing with gym
                states = states.squeeze(1)
                next_states = next_states.squeeze(1)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            targets = torch.cat([next_states, rewards], dim=-1).unsqueeze(1).repeat(
                1, config.particles, 1).detach()
            sample_z = self.mdp.sample_model_seed(return_seed=True)

            pred_states, pred_rewards = self.mdp(
                states, actions, seed=sample_z)

            preds = torch.cat([pred_states, pred_rewards], dim=-1)
            preds = preds.transpose(0, 1)  # [batch, particles, d_state+1]

            # [batch, particles/2, d_state+1]
            preds_i, preds_j = torch.split(
                preds, config.particles//2, dim=1)
            targets_i, targets_j = torch.split(
                targets, config.particles//2, dim=1)

            ## mdp function svgd 
            logp_loss = F.mse_loss(
                preds_j, targets_j.detach(), reduction='none')
            logp_grad = autograd.grad(logp_loss.sum(), inputs=preds_j)[0]
            logp_grad = logp_grad.unsqueeze(2) # [batch, particles//2, 1, d_state+1]

            # grad_kappa: [batch, particles/2, particles/2, d_state+1]
            kappa, grad_kappa = batch_rbf_xy(preds_j, preds_i)
            kappa = kappa.unsqueeze(-1)  # [batch, particles/2, particles/2, 1]

            # [batch, particles/2, particles/2, d_state+1]
            kernel_grad = torch.matmul(kappa.detach(), logp_grad)
            # [batch, particles/2, d_state+1]
            svgd = (kernel_grad + alpha * grad_kappa).mean(1)

            self.mdp_optimizer.zero_grad()
            autograd.backward(preds_i, grad_tensors=svgd.detach())

            for param in self.mdp.parameters():
                if param.grad is not None:
                    param.grad.data *= 1./config.particles

            if config.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.mdp.parameters(), config.gradient_clip)

            with config.lock:
                self.mdp_optimizer.step()

            self.logger.add_scalar(
                'mdp_logp', logp_loss.mean(), self.total_steps)
            self.logger.add_scalar('mdp_kappa', kappa.mean(), self.total_steps)
            self.logger.add_scalar(
                'mdp_kappa', grad_kappa.mean(), self.total_steps)
