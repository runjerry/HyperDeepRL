######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sweep(game, tag, model_fn, trials=50, manual=True):
    hyperparams = {
        'alpha_i': [1, 10, 100],
        'alpha_f': [.1, 0.01],
        'anneal': [500e3],
        'lr_a': [1e-4],
        'lr_c': [1e-3],
        'freq' : [100, 150],
        'hidden': [256, 128, 512],
        'replay_size': [int(1e6), int(1e7)],
        'replay_bs': [512],
        'dist': ['categorical', 'multinomial', 'normal', 'uniform']
        # 'dist': ['multinomial']
    }
    # manually define
    if manual:
        print ('=========================================================')
        print ('Running Manually Defined Single Trial, [1/1]')
        setting = {
            'game': game,
            'tb_tag': tag,
            'alpha_i': 10,
            'alpha_f': .1,
            'anneal': 500e3,
            'lr': 1e-4,
            'freq': 100,
            'hidden': 256,
            'replay_size': int(1e6),
            'replay_bs': 512,
            'dist': 'softmax'
        }
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        model_fn(**setting)
        return

    search_space = list(product_dict(hyperparams))
    ordering = list(range(len(search_space)))
    np.random.shuffle(ordering)
    for i, idx in enumerate(ordering):
        setting = search_space[idx]
        setting['game'] = game
        setting['tb_tag'] = tag
        print ('=========================================================')
        print ('Search Space Contains {} Trials, Running [{}/{}] ---- ({}%)'.format(
            len(search_space), i+1, trials, int(float(i+1)/trials*100.)))
        print ('Running Config: ')
        for (k, v) in setting.items():
            print ('{} : {}'.format(k, v))
        dqn_feature(**setting)
    

# DDPG
def ddpg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.tag = 'WalkerFinalHyper'
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.particles = 24
    print (config.state_dim, config.action_dim)

    config.network_fn = lambda: DeterministicActorCriticHyperNet(
        config.state_dim, config.action_dim,
        actor_body=FCHyperBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.AdamW(params, lr=config.lr_a),
        critic_opt_fn=lambda params: torch.optim.AdamW(params, lr=config.lr_c))

    config.replay_fn = lambda: Replay(memory_size=config.replay_size, config.replay_bs)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3
    config.alpha_init = config.alpha_i
    config.alpha_final = config.alpha_f
    config.alpha_anneal = config.anneal
    run_steps(DDPGAgent(config))
    # run_steps(DDPG_SVGDAgent(config))


# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.tag = config.tb_tag
    config.particles = 24

    config.network_fn = lambda: TD3HyperNet(
        config.action_dim,
        actor_body_fn=lambda: FCHyperBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=config.lr_a),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=config.lr_c))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    config.alpha_init = config.alpha_i
    config.alpha_final = config.alpha_f
    config.alpha_anneal = config.anneal
    run_steps(TD3Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)
 
    # game = 'HalfCheetah-v2'
    # game = 'Ant-v2'
    # game = 'Reacher-v2'
    # game = 'InvertedPendulum-v2'
    # game = 'InvertedDoublePendulum-v2'
    # game = 'Humanoid-v2'
    # game = 'Hopper-v2'
    # game = 'Swimmer-v2'
    # game = 'Walker2d-v2'

    game = 'HalfCheetah-v2'
    tag = 'ddpg_sweep'
    algorithm = ddpg_continuous
    # algorithm = td3_continuous
    sweep(game, tag, algorithm, trials=50, manual=True

