#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from .hypernetwork_ops import *
from ..utils.hypernet_heads_defs import *

particles = 2

class VanillaHyperNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaHyperNet, self).__init__()
        self.mixer = False
        conf = VanillaNet_config(body.feature_dim, output_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_head = LinearGenerator(conf['fc_head'])

        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        print (x.shape)
        phi = self.body(tensor(x))
        y = self.fc_head(z[0], phi)
        return y


class DuelingHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingHyperNet, self).__init__()
        self.mixer = False
        conf = DuelingNet_config(body.feature_dim, action_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_value = LinearGenerator(conf['fc_value'])
        self.fc_advantage = LinearGenerator(conf['fc_advantage'])
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        phi = self.body(tensor(x))
        value = self.fc_value(z[0], phi)
        advantange = self.fc_advantage(z[1], phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalHyperNet, self).__init__()
        self.mixer = False
        conf = CategoricalNet_config(body.feature_dim, action_dim, num_atoms)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_categorical = LinearGenerator(conf['fc_categorical'])
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(z[0], phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileHyperNet, self).__init__()
        self.mixer = False
        conf = QuantileNet_config(body.feature_dim, action_dim, num_quantiles)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_quantiles = LinearGenerator(conf['fc_quantiles'])
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(z[0], phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticHyperNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticHyperNet, self).__init__()
        self.mixer = False
        conf = OptionCriticNet(body.features_dim, action_dim, num_options)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_q = LinearGenerator(conf['fc_q'])
        self.fc_pi = LinearGenerator(conf['fc_pi_'])
        self.fc_beta = LinearGenerator(conf['fc_beta'])
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
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
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        conf = DeterministicActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_action = LinearGenerator(conf['fc_action'])
        self.fc_critic = LinearGenerator(conf['fc_critic'])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        if not self.mixer:
            z = torch.rand(1, particles, self.z_dim)
        return torch.tanh(self.fc_action(z[0], self.actor_body(phi)))

    def critic(self, phi, a):
        if not self.mixer:
            z = torch.rand(1, particles, self.z_dim)
        return self.fc_critic(z[0], self.critic_body(phi, a))


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
        conf = GaussianActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_action = LinearGenerator(conf['fc_action'])
        self.fc_critic = LinearGenerator(conf['fc_critic'])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
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
        conf = CategoricalActorCriticNet_config(actor_body.feature_dim, critic_body.feature_dim, action_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_action = LinearGenerator(conf['fc_action'])
        self.fc_critic = LinearGenerator(conf['fc_critic'])
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
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

        conf = TD3Net_config(actor_body.feature_dim, critic_body_1.feature_dim, critic_body_2.feature_dim, action_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_action = LinearGenerator(conf['fc_action'])
        self.fc_critic_1 = LinearGenerator(conf['fc_critic1'])
        self.fc_critic_2 = LinearGenerator(conf['fc_critic2'])

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        if not self.mixer:
            z = torch.rand(1, particles, self.z_dim)
        obs = tensor(obs)
        return torch.tanh(self.fc_action(z[0], self.actor_body(obs)))

    def q(self, obs, a):
        if not self.mixer:
            z = torch.rand(2, particles, self.z_dim)
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(z[0], self.critic_body_1(x))
        q_2 = self.fc_critic_2(z[1], self.critic_body_2(x))
        return q_1, q_2
