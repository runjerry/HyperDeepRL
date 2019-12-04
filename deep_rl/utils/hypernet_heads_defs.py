from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (64, 3, False, 64, 64, 'relu', None, 1) # python 3.6

vanilla_net = namedtuple('vanilla_net', ' '.join(fields))
vanilla_net.__new__.__defaults__ = defaults # python 3.6
def VanillaNet_config(feature_dim, output_dim, mixer=False):
    net_config = {
            'fc_head': vanilla_net(d_input=feature_dim, d_output=output_dim),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = vanilla_net(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

toy_dueling_net = namedtuple('toy_dueling_net', ' '.join(fields))
toy_dueling_net.__new__.__defaults__ = defaults # python 3.6
def ToyDuelingNet_config(feature_dim, action_dim, mixer=False):
    net_config = {
            'fc_value': toy_dueling_net(d_input=feature_dim, d_output=1, z_dim=24),
            'fc_advantage': toy_dueling_net(d_input=feature_dim, d_output=action_dim, z_dim=24),
            'z_dim': 24,
            's_dim': 24,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = toy_dueling_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config 


dueling_net = namedtuple('dueling_net', ' '.join(fields))
dueling_net.__new__.__defaults__ = defaults # python 3.6
def DuelingNet_config(feature_dim, action_dim, mixer=False):
    net_config = {
            'fc_value': dueling_net(d_input=feature_dim, d_output=1, z_dim=24),
            'fc_advantage': dueling_net(d_input=feature_dim, d_output=action_dim, z_dim=24),
            'z_dim': 24,
            's_dim': 24,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = dueling_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

categorical_net = namedtuple('categorical_net', ' '.join(fields))
categorical_net.__new__.__defaults__ = defaults # python 3.6
def CategoricalNet_config(feature_dim, action_dim, num_atoms, mixer=False):
    net_config = {
            'fc_categorical': categorical_net(d_input=feature_dim, d_output=action_dim * num_atoms),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = categorical_net(d_hidden=64, d_output=None, n_gen=1),
    return net_config 

quantile_net = namedtuple('quantile_net', ' '.join(fields))
quantile_net.__new__.__defaults__ = defaults # python 3.6
def QuantileNet_config(feature_dim, action_dim, num_quantiles, mixer=False):
    net_config = {
            'fc_quantiles': quantile_net(d_input=feature_dim, d_output=action_dim * num_quantiles),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = quantile_net(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

option_net = namedtuple('option_net', ' '.join(fields))
option_net.__new__.__defaults__ = defaults # python 3.6
def OptionCriticNet_config(feature_dim, action_dim, num_options, mixer=False):
    net_config = {
            'fc_q': option_net(d_input=feature_dim, d_output=num_options),
            'fc_pi': option_net(d_input=feature_dim, d_output=action_dim * num_options),
            'fc_beta': option_net(d_input=feature_dim, d_output=num_options),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 1
            }
    if mixer:
        net_config['mixer'] = option_net(d_hidden=64, d_output=None, n_gen=1)
    return net_config 

deterministic_net = namedtuple('deterministic_net', ' '.join(fields))
deterministic_net.__new__.__defaults__ = defaults # python 3.6
def DeterministicActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    net_config = {
            'fc_action': deterministic_net(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': deterministic_net(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = deterministic_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

gaussian_net = namedtuple('gaussian_net', ' '.join(fields))
gaussian_net.__new__.__defaults__ = defaults # python 3.6
def GaussianActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    net_config = {
            'fc_action': gaussian_net(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': gaussian_net(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gaussian_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

categorical_ac_net = namedtuple('categorical_ac_net', ' '.join(fields))
categorical_ac_net.__new__.__defaults__ = defaults # python 3.6
def CategoricalActorCriticNet_config(feature_dim_a, feature_dim_c, action_dim, mixer=False):
    net_config = {
            'fc_actor': categorical_ac_net(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic': categorical_ac_net(d_input=feature_dim_c, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = categorical_ac_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config 

td3_net = namedtuple('td3_net', ' '.join(fields))
td3_net.__new__.__defaults__ = defaults # python 3.6
def TD3Net_config(feature_dim_a, feature_dim_c1, feature_dim_c2, action_dim, mixer=False):
    net_config = {
            'fc_action': td3_net(d_input=feature_dim_a, d_output=action_dim),
            'fc_critic1': td3_net(d_input=feature_dim_c1, d_output=1),
            'fc_critic2': td3_net(d_input=feature_dim_c2, d_output=1),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 3
            }
    if mixer:
        net_config['mixer'] = td3_net(d_hidden=64, d_output=None, n_gen=3)
    return net_config 

