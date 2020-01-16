from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (100, 3, False, 64, 64, 'relu', None, 1) # python 3.6

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

