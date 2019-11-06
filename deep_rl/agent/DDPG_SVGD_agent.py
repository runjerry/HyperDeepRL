#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
##############################################################################
# File modified to use HyperGAN as an implicit distribution over every model #
# During learning, a head is chosen at random, and used for prediction.      #
# Action selection is done by taking the mean over the sampled ensemble.     #
# A new ensemble is sampled EVERY time a model is called.                    #
##############################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import torch.autograd as autograd

class DDPG_SVGDAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn() # change to single target, not dist
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.alpha_schedule = BaselinesLinearSchedule(2e5, final_p=.1, initial_p=1.)
        print (self.network)
        self.particles = config.particles
        
    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)
        
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action, _ = self.network.predict_action(state, evaluation=True)
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        if config.hyper == True:
            self.network.sample_model_seed()                             ## sample from isotropic gaussian (new ensemble)
            self.target_network.set_model_seed(self.network.model_seed)
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        """ Predict Action """
        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action, _ = self.network.predict_action(self.state, evaluation=True) ## take mean action over ensemble
            action += self.random_process.sample()
        """ Take Action """
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        """ Store to Buffer """
        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1
        """ Learn """
        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)
            """ Update Critic """
            q_next = self.target_network.predict_action(next_states)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)

            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            """ Update Actor """
            phi = self.network.feature(states)
            actor_theta_list = self.network.sample_model(component='actor')
            actions = self.network.actor(phi, actor_theta_list)
            
            # SVGD Update """
            alpha = self.alpha_schedule.value(self.total_steps)
            q = torch.stack([self.network.critic(phi, a) for a in actions])
            q_grad = autograd.grad(q.sum(), inputs=actions)[0]
            q_grad = q_grad.transpose(0, 1).unsqueeze(2)

            for i in range(len(actor_theta_list)):
                actor_theta_list[i] = actor_theta_list[i].view(actor_theta_list[i].size(0), -1)
            actor_theta = torch.cat(actor_theta_list, 1)
            
            kappa = batch_rbf_nograd(actor_theta) / self.particles # [n, n], [n, theta]
            grad_kappa = autograd.grad(kappa.sum(), actor_theta)[0] / self.particles
            q_grad = q_grad.mean(0).mean(-1)  # [n, 1] 
            kernel_logp = torch.matmul(kappa.detach(), q_grad)  # [n, 1]
            svgd = (kernel_logp + alpha * grad_kappa)  # [n, theta]
            self.network.actor_opt.zero_grad()

            autograd.backward(actor_theta, grad_tensors=svgd)

            if not self.total_steps % 100: 
                print (svgd.mean().item(), kappa.mean().item(), grad_kappa.mean().item(), q.mean().item())
            
            self.network.actor_opt.zero_grad()
            # q.backward(torch.ones_like(q).float().cuda())
            
            #q_vals = q.squeeze(-1).t().mean(0)
            #q_vals.backward(torch.tensor(np.ones(q_vals.size())).float().cuda())

            #self.network.critic_opt.zero_grad()

            #action_grads = torch.stack([-dead_action.grad.detach() for dead_action in dead_actions])
            
            #actions.backward(action_grads)
            self.network.actor_opt.step()
            self.soft_update(self.target_network, self.network)
