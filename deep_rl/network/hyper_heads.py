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

particles = 32

class VanillaHyperNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaHyperNet, self).__init__()
        self.mixer = False
        self.config = VanillaNet_config(body.feature_dim, output_dim)
        self.fc_head = LinearGenerator(self.config['fc_head'])
        self.body = body
        self.to(Config.DEVICE)

    def sample_model_seed(self):
        if not self.mixer:
            self.model_seed = {
                    'fc_head_z': torch.rand(self.fc_head.config['n_gen'], particles, self.z_dim).to(Config.DEVICE)
            }
        else:
            self.model_seed = torch.rand(particles, self.s_dim)
    
    def forward(self, x, z=None):
        phi = self.body(tensor(x, z))
        y = self.fc_head(z[0], phi)
        return y


class DuelingHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingHyperNet, self).__init__()
        self.mixer = False
        self.config = DuelingNet_config(body.feature_dim, action_dim)
        self.fc_value = LinearGenerator(self.config['fc_value'])
        self.fc_advantage = LinearGenerator(self.config['fc_advantage'])
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(z[0], phi)
        advantange = self.fc_advantage(z[1], phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalHyperNet, self).__init__()
        self.mixer = False
        self.config = CategoricalNet_config(body.feature_dim, action_dim, num_atoms)
        self.fc_categorical = LinearGenerator(self.config['fc_categorical'])
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(z[0], phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileHyperNet, self).__init__()
        self.mixer = False
        self.config = QuantileNet_config(body.feature_dim, action_dim, num_quantiles)
        self.fc_quantiles = LinearGenerator(self.config['fc_quantiles'])
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(z[0], phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticHyperNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticHyperNet, self).__init__()
        self.mixer = False
        self.config = OptionCriticNet(body.features_dim, action_dim, num_options)
        self.fc_q = LinearGenerator(self.config['fc_q'])
        self.fc_pi = LinearGenerator(self.config['fc_pi_'])
        self.fc_beta = LinearGenerator(self.config['fc_beta'])
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(z[0], phi)
        pi = self.fc_pi(z[1], phi)
        beta = F.sigmoid(self.fc_beta(z[2], phi))
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


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
        self.sample_model_seed()
        self.to(Config.DEVICE)

    def sample_model_seed(self):
        if not self.mixer:
            self.model_seed = {
                    'phi_body_z': torch.rand(self.phi_body.config['n_gen'], particles, self.z_dim).to(Config.DEVICE),
                    'actor_body_z': torch.rand(self.actor_body.config['n_gen'], particles, self.z_dim).to(Config.DEVICE),
                    'action_z': torch.rand(particles, self.z_dim).to(Config.DEVICE),
            }
        else:
            self.model_seed = torch.rand(particles, self.s_dim)
   
    def set_model_seed(self, seed):
        self.model_seed = seed

    def predict_action(self, obs, evaluation=False):
        phi = self.feature(obs)
        actions = self.actor(phi).detach_()
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

    def actor(self, phi):
        a = self.actor_body(phi, self.model_seed['actor_body_z'])
        return torch.tanh(self.fc_action(self.model_seed['action_z'], a))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))

    def sample_model(self):
        phi_body = self.phi_body(z=self.model_seed['phi_body_z'])
        actor_body = self.actor_body(z=self.model_seed['actor_body_z'])
        critic_body = self.critic_body(z=self.model_seed['critic_body_z'])
        fc_action = self.fc_action(z=self.model_seed['action_z'])
        fc_critic = self.fc_critic(z=self.model_seed['critic_z'])
        return [*actor_body, *critic_body, *fc_action, *fc_critic]


class GaussianActorCriticHyperNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticHyperNet, self).__init__()
        self.mixer = False
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.config = GaussianActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.fc_action = LinearGenerator(self.config['fc_action'])
        self.fc_critic = LinearGenerator(self.config['fc_critic'])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(z[0], phi_a))
        v = self.fc_critic(z[1], phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}


class CategoricalActorCriticHyperNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticHyperNet, self).__init__()
        self.mixer = False
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.config = CategoricalActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.fc_action = LinearGenerator(self.config['fc_action'])
        self.fc_critic = LinearGenerator(self.config['fc_critic'])
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(z[0], phi_a)
        v = self.fc_critic(z[1], phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}


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

        self.config = TD3Net_config(actor_body.feature_dim, critic_body_1.feature_dim, critic_body_2.feature_dim, action_dim)
        self.fc_action = LinearGenerator(self.config['fc_action'])
        self.fc_critic_1 = LinearGenerator(self.config['fc_critic1'])
        self.fc_critic_2 = LinearGenerator(self.config['fc_critic2'])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(z[0], self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(z[0], self.critic_body_1(x))
        q_2 = self.fc_critic_2(z[1], self.critic_body_2(x))
        return q_1, q_2
