######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import itertools
import pprint

def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sweep(game, tag, model_fn, trials=50, manual=False):
    hyperparams = {
        'alpha_i': [10, 100],
        'alpha_f': [.1],
        'anneal': [1e5, 1e6, 1e7, 2e7],
        'lr': [2e-4, 1e-4],
        'freq' : [100, 1000, 10000],
        'grad_clip': [None, 5],
        'hidden': [256],
        'replay_size': [int(1e6), int(1e7)],
        'replay_bs': [32, 64, 128],
        # 'dist': ['categorical', 'multinomial', 'multivariate_normal']
        'dist': ['categorical']
    }
    # manually define
    if manual:
        print ('=========================================================')
        print ('Running Manually Defined Single Trial, [1/1]')
        model_fn(game=game,
                    tb_tag=tag,
                    alpha_i=10,
                    alpha_f=.1,
                    anneal=500e3,
                    lr=1e-4,
                    freq=10000,
                    grad_clip=5,
                    hidden=256,
                    replay_size=int(1e6),
                    replay_bs=128,
                    dist='categorical')
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
        model_fn(**setting)
    
   
def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True
    config.tag = config.tb_tag
    config.generate_log_handles()
    config.task_fn = lambda: Task(config.game, video=False, gif=False, log_dir=config.tf_log_handle)
    config.eval_env = config.task_fn()
    config.render = False
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, NatureConvBody(in_channels=config.history_length),
                                                hidden=config.hidden, dist=config.dist, particles=config.particles)
    config.random_action_prob = LinearSchedule(0.1, 0.001, 1e6)
    # config.replay_fn = lambda: AsyncReplay(memory_size=config.replay_size, batch_size=config.replay_bs)
    config.replay_fn = lambda: Replay(memory_size=config.replay_size, batch_size=config.replay_bs)
    config.batch_size = config.replay_bs
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = config.freq
    config.exploration_steps = 0
    config.sgd_update_frequency = 4
    config.gradient_clip = config.grad_clip
    config.history_length = 4
    config.double_q = True
    config.async_actor = False
    config.max_steps = int(2e7)
    config.alpha_anneal = config.anneal
    config.alpha_init = config.alpha_i
    config.alpha_final = config.alpha_f
    run_steps(DQN_Dist_SVGD_Agent(config))


def n_step_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, NatureConvBody(), hidden=config.hidden,
                                                dist=config.dist, particles=config.particles)
    config.random_action_prob = LinearSchedule(0.1, 0.01, 1e6)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = config.freq
    config.rollout_length = 5
    config.gradient_clip = config.grad_clip
    config.max_steps = int(2e7)
    config.alpha_anneal = config.anneal
    config.alpha_init = config.alpha_i
    config.alpha_final = config.alpha_f
    run_steps(NStepDQN_Dist_SVGD_Agent(config))



if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    tag = 'pong_trials_cat'
    game = 'FreewayNoFrameskip-v4'
    sweep(game, tag, dqn_pixel, trials=50, manual=True)

