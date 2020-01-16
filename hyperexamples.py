######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools
import envs

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
   
def sweep(game, tag, model_fn, chain_len=5, trials=50, manual=True):
    hyperparams = {
        'alpha_i': [0.01, 0.1, 1, 10],
        'alpha_f': [0.01, 0.1, 1, 10],
        'anneal': [500e3],
        'lr': [1e-2, 1e-3],
        'freq' : [10, 25],
        'grad_clip': [None, 1],
        'hidden': [256, 128],
        'replay_size': [int(1e3), int(1e4)],
        'replay_bs': [32, 64, 128],
        'dist': ['categorical', 'multinomial', 'normal', 'uniform']
    }
    # manually define
    if manual:
        print ('=========================================================')
        print ('Running Manually Defined Single Trial, [1/1]')
        setting = {
            'game': game,
            'tb_tag': tag,
            'alpha_i': 0.1,
            'alpha_f': 0.1,
            'anneal': (chain_len+9)*2000,
            'lr': 0.01,
            'freq': 10,
            'grad_clip': None,
            'hidden': 256,
            'replay_size': int(1e3),
            'replay_bs': 32,
            'dist': 'categorical'
        }
        print ('Running Config: ')
        for (k, v) in setting.items():
            if setting['alpha_f'] > setting['alpha_i']:
                continue
            print ('{} : {}'.format(k, v))
        setting['chain_len'] = chain_len
        setting['game'] = game
        setting['tb_tag'] = tag
        model_fn(**setting)
        return

    search_space = list(product_dict(hyperparams))
    ordering = list(range(len(search_space)))
    np.random.shuffle(ordering)
    for i, idx in enumerate(ordering):
        setting = search_space[idx]
        setting['game'] = game
        setting['tb_tag'] = tag
        setting['chain_len'] = chain_len
        print ('=========================================================')
        print ('Search Space Contains {} Trials, Running [{}/{}] ---- ({}%)'.format(
            len(search_space), i+1, trials, int(float(i+1)/trials*100.)))
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        model_fn(**setting)
    
# DQN Toy Example
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True
    config.tag = config.tb_tag
    config.generate_log_handles()
    config.task_fn = lambda: Task(config.game, log_dir=config.tf_log_handle, special_args=('NChain', config.chain_len))
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=config.lr)


    body = ThompsonHyperBody(config.state_dim, hidden=config.hidden)
    config.network_fn = lambda: ThompsonHyperNet(config.action_dim, body, hidden=config.hidden, dist=config.dist, particles=config.particles)
    config.replay_fn = lambda: Replay(memory_size=int(config.replay_size), batch_size=int(config.replay_bs))

    config.render = False
    config.random_action_prob = LinearSchedule(0.01, 0.001, 1e4)
    config.discount = 0.8
    config.target_network_update_freq = config.freq
    config.exploration_steps = 0
    config.double_q = True
    config.thompson = True
    config.sgd_update_frequency = 4
    config.gradient_clip = 1
    config.eval_interval = int(5e3)
    config.max_steps = 2000 * (config.chain_len+9)
    config.async_actor = False
    config.particles = 24
    config.alpha_anneal = config.anneal
    config.alpha_init = config.alpha_i
    config.alpha_final = config.alpha_f
    run_steps(DQNThompsonAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)


    tag = 'nchain-thompson-1'
    game = 'NChain-v3'
    for c in list(range(4, 50)):
        sweep(game, tag, dqn_feature, c, manual=True)
    
