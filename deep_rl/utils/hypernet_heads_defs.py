from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (100, 3, False, 64, 64, 'relu', None, 1) # python 3.6

def VanillaNet_config(feature_dim, output_dim, mixer=False):
    gen_config = namedtuple('VanillaHyperNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_head': gen_config(d_input=feature_dim, d_output=output_dim),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

def DuelingNet_config(feature_dim, action_dim, mixer=False):
    gen_config = namedtuple('DuelingNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_value': gen_config(d_input=feature_dim, d_output=1),
            'fc_advantage': gen_config(d_input=feature_dim, d_output=action_dim),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

def CategoricalNet_config(feature_dim, action_dim, num_atoms, mixer=False):
    gen_config = namedtuple('CategoricalNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_categorical': gen_config(d_input=feature_dim, d_output=action_dim * num_atoms),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=1),
    return net_config 

def QuantileNet_config(feature_dim, action_dim, num_quantiles, mixer=False):
    gen_config = namedtuple('QuantileNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_quantiles': gen_config(d_input=feature_dim, d_output=action_dim * num_quantiles),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

def OptionCriticNet_config(feature_dim, action_dim, num_options, mixer=False):
    gen_config = namedtuple('OptionCriticNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_q': gen_config(d_input=feature_dim, d_output=num_options),
            'fc_pi': gen_config(d_input=feature_dim, d_output=action_dim * num_options),
            'fc_beta': gen_config(d_input=feature_dim, d_output=num_options),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

def DeterministicActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    gen_config = namedtuple('DeterministicActorCriticNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_action': gen_config(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': gen_config(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

def GaussianActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    gen_config = namedtuple('GaussianActorCriticNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_action': gen_config(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': gen_config(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

def CategoricalActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    gen_config = namedtuple('CategoricalActorCriticNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_actor': gen_config(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': gen_config(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

def TD3Net_config(feature_dim_a, feature_dim_c1, feature_dim_c2, action_dim, mixer=False):
    gen_config = namedtuple('TD3CriticNet', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_action': gen_config(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic1': gen_config(d_input=feature_dim_c1, d_output=1),
            'fc_critic2': gen_config(d_input=feature_dim_c2, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 3
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=3)
    return net_config 

