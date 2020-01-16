from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (100, 3, False, 64, 64, 'relu', None, 4) # python 3.6


toy_fc_body = namedtuple('toy_fc_body', ' '.join(fields))
toy_fc_body.__new__.__defaults__ = defaults # python 3.6
def ToyFCBody_config(d_state, d_hidden, gate, mixer=False):
    net_config = {}
    net_config['fc1'] = toy_fc_body(d_input=d_state, d_output=d_hidden[0], z_dim=24)
    for i in range(1, len(d_hidden)):
        net_config['fc{}'.format(i+1)] = toy_fc_body(d_input=d_hidden[i-1], d_output=d_hidden[i], z_dim=24)
    if mixer:
        net_config['mixer'] = toy_fc_body(d_hidden=64, d_output=None, n_gen=len(d_hidden))
    net_config['z_dim'] = 24
    net_config['s_dim'] = 24
    net_config['n_gen'] = len(d_hidden)

    return net_config 

fc_body = namedtuple('fc_body', ' '.join(fields))
fc_body.__new__.__defaults__ = defaults # python 3.6
def FCBody_config(d_state, d_hidden, gate, mixer=False):
    net_config = {}
    net_config['fc1'] = fc_body(d_input=d_state, d_output=d_hidden[0], z_dim=24)
    for i in range(1, len(d_hidden)):
        net_config['fc{}'.format(i+1)] = fc_body(d_input=d_hidden[i-1], d_output=d_hidden[i], z_dim=24)
    if mixer:
        net_config['mixer'] = fc_body(d_hidden=d_hidden, d_output=None, n_gen=len(d_hidden))
    net_config['z_dim'] = 24
    net_config['s_dim'] = 24
    net_config['n_gen'] = len(d_hidden)

    return net_config


dummy_body = namedtuple('dummy_body', ' '.join(fields))
dummy_body.__new__.__defaults__ = defaults # python 3.6
def Dummy_config():
    net_config = {
            'fc_s': dummy_body(d_input=0, d_output=0),
            'fc_a': dummy_body(d_input=0, d_output=0),
            'z_dim': 0,
            's_dim': 0,
            'n_gen': 0
            }
    return net_config 


