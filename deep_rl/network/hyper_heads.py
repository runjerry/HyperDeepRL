#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from .hyper_bodies import * 
from .hypernetwork_ops import *
from ..utils.hypernet_heads_defs import *


class DeterministicActorCriticHyperNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticHyperNet, self).__init__()
        self.mixer = False
        if phi_body is None: phi_body = DummyHyperBody(state_dim)
        if actor_body is None: actor_body = DummyHyperBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyHyperBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.config = DeterministicActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.fc_action = LinearGenerator(self.config['fc_action']).cuda()
        # self.fc_critic = LinearGenerator(self.config['fc_critic']).cuda()
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']

        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.n_gen = self.config['n_gen'] + self.phi_body.config['n_gen'] + \
                     self.actor_body.config['n_gen'] + 1 #self.critic_body.config['n_gen'] + 1
        self.particles = Config.particles
        self.sample_model_seed()
        self.to(Config.DEVICE)

    def sample_model_seed(self):
        self.model_seed = {
                'phi_body_z': torch.rand(self.phi_body.config['n_gen'], self.particles, self.z_dim).to(Config.DEVICE),
                'actor_body_z': torch.rand(self.actor_body.config['n_gen'], self.particles, self.z_dim).to(Config.DEVICE),
                'action_z': torch.rand(self.particles, self.z_dim).to(Config.DEVICE),
        }
   
    def set_model_seed(self, seed):
        self.model_seed = seed

    def predict_action(self, obs, evaluation=False, theta=None):
        phi = self.feature(obs)
        actions = self.actor(phi, theta).detach_()
        q_vals = torch.stack([self.critic(phi, action) for action in actions])
        q_vals = q_vals.squeeze(-1).t()
        actions = actions.transpose(0, 1)
        if evaluation:
            q_max = q_vals.max(1)[1]
            actions = actions[torch.tensor(np.arange(actions.size(0))).long(), q_max, :]
            return actions.detach().cpu().numpy(), q_max.detach().cpu().numpy()
        return q_vals.max(1)[0].unsqueeze(-1)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs, self.model_seed['phi_body_z'])

    def actor(self, phi, theta=None):
        if theta:
            a = self.actor_body(phi, self.model_seed['actor_body_z'], theta[:4])
            res = torch.tanh(self.fc_action(self.model_seed['action_z'], a, theta[-2:]))
        else:
            a = self.actor_body(phi, self.model_seed['actor_body_z'])
            res = torch.tanh(self.fc_action(self.model_seed['action_z'], a))
        return res

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))

    def sample_model(self, component):
        param_sets = []
        if component == 'actor':
            param_sets.extend(self.actor_body(z=self.model_seed['actor_body_z']))
            param_sets.extend(self.fc_action(z=self.model_seed['action_z']))
        elif component == 'critic':
            param_sets.extend(*self.critic_body(z=self.model_seed['critic_body_z']))
            param_sets.extend(*self.fc_critic(z=self.model_seed['critic_z']))
        return param_sets


class TD3HyperNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3HyperNet, self).__init__()
        self.mixer = False
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.config = TD3Net_config(self.actor_body.feature_dim, self.critic_body_1.feature_dim, self.critic_body_2.feature_dim, action_dim)
        self.fc_action = LinearGenerator(self.config['fc_action']).cuda()
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen'] + self.actor_body.config['n_gen'] + 1
        
        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        
        self.sample_model_seed()
        
        self.to(Config.DEVICE)
    
    def sample_model_seed(self):
        self.model_seed = {
                'actor_body_z': torch.rand(self.actor_body.config['n_gen'], particles, self.z_dim).to(Config.DEVICE),
                'action_z': torch.rand(particles, self.z_dim).to(Config.DEVICE),
        }
   
    def set_model_seed(self, seed):
        self.model_seed = seed

    def predict_action(self, obs, evaluation=False, maxq=True, theta=None, numpy=True):
        actions = self.actor(obs, theta).detach_()
        q_vals = [self.q(obs, action) for action in actions]
        q1 = torch.stack([q[0] for q in q_vals]).squeeze(-1).t()  # [particles]
        q2 = torch.stack([q[1] for q in q_vals]).squeeze(-1).t()  # [particles]
        actions = actions.transpose(0, 1)  # [1, particles, d_action]
        ## get inddx of the best action
        q1_max, q2_max = q1.max(1), q2.max(1)  # both [[1], [1]]
        ## get best action betweeen them
        if maxq:
            q_max = torch.max(q1_max[0], q2_max[0])
        else:
            q_max = torch.min(q1_max[0], q2_max[0])

        if evaluation:
            if torch.equal(q_max, q1_max[0]):
                q_max_idx = q1_max[1]  # [1]
            else:
                q_max_idx = q2_max[1]  # [1]
            
            actions = actions[torch.tensor(np.arange(actions.size(0))).long(), q_max_idx, :]

            if numpy == True:
                actions = actions.detach().cpu().numpy()
                q_max_idx = q_max_idx.detach().cpu().numpy()
            return actions, q_max_idx
        
        return q_max.unsqueeze(-1)


    def forward(self, obs):
        action = self.actor(obs)
        return action

    def actor(self, obs, theta=None):
        obs = tensor(obs)
        if theta:
            a = self.actor_body(obs, self.model_seed['actor_body_z'], theta[:4])
            action = torch.tanh(self.fc_action(self.model_seed['action_z'], a, theta[-2:]))
        else:
            a = self.actor_body(obs, self.model_seed['actor_body_z'])
            action = torch.tanh(self.fc_action(self.model_seed['action_z'], a))
        return action

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2
