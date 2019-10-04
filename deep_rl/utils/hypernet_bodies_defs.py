from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (100, 3, False, 64, 64, 'relu', None, 4) # python 3.6

def NatureConvBody_config(in_channels, feature_dim, mixer=False):
    gen_config = namedtuple('NatureConvBody', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'conv1': gen_config(d_input=in_channels, d_output=32, kernel=4),
            'conv2': gen_config(d_input=32, d_output=64, kernel=2),
            'conv3': gen_config(d_input=64, d_output=64, kernel=1),
            'fc4': gen_config(d_input=7*7*64, d_output=feature_dim),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 4
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=4),

    return net_config 

def DDPGConvBody_config(in_channels, feature_dim, mixer=False):
    gen_config = namedtuple('DDPGConvBody', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'conv1': gen_config(d_input=in_channels, d_output=32, kernel=3),
            'conv2': gen_config(d_input=32, d_output=32, kernel=3),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2),

    return net_config 

def FCBody_config(d_state, d_hidden, gate, mixer=False):
    gen_config = namedtuple('FCBody', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {}
    net_config['fc1'] = gen_config(d_input=d_state, d_output=d_hidden[0])
    for i in range(1, len(d_hidden)):
        net_config['fc{}'.format(i+1)] = gen_config(d_input=d_hidden[i-1], d_output=d_hidden[i])
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=len(d_hidden))
    net_config['z_dim'] = 64
    net_config['s_dim'] = 64
    net_config['n_gen'] = len(d_hidden)

    return net_config 

def TwoLayerFCBodyWithAction_config(d_state, d_action, d_hidden, gate, mixer=False):
    gen_config = namedtuple('TwoLayerFCBodyWithAction', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc1': gen_config(d_input=d_state, d_output=d_hidden[0]),
            'fc2': gen_config(d_input=d_hidden[0]+d_action, d_output=d_hidden[1]),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2),

    return net_config 

def OneLayerFCBodyWithAction_config(d_state, d_action, d_hidden, gate, mixer=False):
    gen_config = namedtuple('OneLayerBodyWithAction', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_s': gen_config(d_input=d_state, d_output=d_hidden),
            'fc_a': gen_config(d_input=d_action, d_output=d_hidden),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = gen_config(d_hidden=64, d_output=None, n_gen=2),

    return net_config 

def Dummy_config():
    gen_config = namedtuple('OneLayerBodyWithAction', ' '.join(fields))
    gen_config.__new__.__defaults__ = defaults # python 3.6
    net_config = {
            'fc_s': gen_config(d_input=0, d_output=0),
            'fc_a': gen_config(d_input=0, d_output=0),
            'z_dim': 0,
            's_dim': 0,
            'n_gen': 0
            }
    return net_config 


