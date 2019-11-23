######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import envs
import itertools

def load_args():
    parser = argparse.ArgumentParser(description='main args')
    parser.add_argument('--tb_tag', default='', type=str)
    args = parser.parse_args()
    return args

# DQN Toy Example
def dqn_toy_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True
    config.tag = 'chain_' + config.tb_tag
    config.task_fn = lambda: Task(config.game, special_args=('NChain', config.chain_len))
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=config.lr)
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, ToyFCHyperBody(config.state_dim), toy=True)
    config.replay_fn = lambda: Replay(memory_size=int(config.replay_memory), batch_size=int(config.replay_bs))

    config.random_action_prob = LinearSchedule(0.01, 0.001, 1e4)
    config.discount = 0.8
    config.target_network_update_freq = 10
    config.exploration_steps = 0
    config.double_q = True
    config.sgd_update_frequency = 4
    config.gradient_clip = 1
    config.eval_interval = int(5e7)
    config.max_steps = 2000 * (config.chain_len+9)
    config.async_actor = False
    config.particles = 24
    config.alpha_anneal = config.max_steps
    config.alpha_init = config.alpha
    config.alpha_final = config.alpha
    # run_steps(DQN_ToyDist_Agent(config))
    run_steps(DQNDistToySVGD_Agent(config))


# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True
    config.tag = config.tb_tag
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, config.lr)
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, CartFCHyperBody(config.state_dim, hidden=config.hidden), toy=True, hidden=config.hidden, dist=config.dist)
    config.replay_fn = lambda: Replay(memory_size=int(1e5), batch_size=128)
    # config.replay_fn = lambda: AsyncReplay(memory_size=int(config.replay_memory_size), batch_size=int(config.replay_bs))

    config.render = False  # Render environment at every train step
    config.random_action_prob = LinearSchedule(0.1, 0.001, 1e4)  # eps greedy params
    config.discount = 0.99  # horizon
    config.target_network_update_freq = config.freq  # hard update to target network
    config.exploration_steps = 0  # random actions taken at the beginning to fill the replay buffer
    config.double_q = True  # use double q update
    config.sgd_update_frequency = 1  # how often to do learning
    config.gradient_clip = config.clip  # max gradient norm
    config.eval_interval = int(5e3) 
    config.max_steps = 250e3
    config.async_actor = False
    config.alpha_anneal = config.anneal  # how long to anneal SVGD alpha from init to final
    config.alpha_init = config.alpha_i  # SVGD alpha strating value
    config.alpha_final = config.alpha_f  # SVGD alpha end value
    items = vars(config)
    print (', '.join("%s: %s" % item for item in items.items()))

    # run_steps(DQN_SVGD_Agent(config))
    run_steps(DQN_Dist_SVGD_Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)
    args = load_args()

    # Grid search for HP
    alpha_init = [.1, 1, 10, 100]   
    alpha_final = [.1, 1, 10, 100] 
    anneal = [10e3, 50e3, 100e3, 250e3]
    learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    freq = [10, 50, 100, 250, 1000]
    clip = [None, 1, 2, 5]
    hidden = [64, 128, 256, 512]
    dist = ['normal', 'categorical', 'uniform']
    
    def chunks(l, n):
        n = max(1, n)
        return list((l[i:i+n] for i in range(0, len(l), n)))

    hyperparams = list(itertools.product(*[alpha_init, alpha_final, anneal, learning_rates, freq, clip, hidden, dist]))
    print (len(hyperparams))
    
    x = chunks(hyperparams, 4)
    c = 76800//4
    hp1 = hyperparams[:c]
    hp2 = hyperparams[c:2*c]
    hp3 = hyperparams[2*c:3*c]
    hp4 = hyperparams[3*c:]
    # game = 'CartPole-v0'  # MUST WORK
    # game = 'bsuite-cartpole/0'
    game = 'bsuite-cartpole_swingup/0'
    for (alpha_i, alpha_f, anneal, lr, freq, clip, hidden, dist) in hp1:
        if alpha_f > alpha_i:
            continue
        tag='alpha{}-{}_lr{}_anneal-{}_freq={}_clip-{}_hidden-{}_dist-{}'.format(alpha_i, alpha_f, lr, anneal, freq, clip, hidden, dist),
        print ('recording a={}-{}, lr={} '.format(alpha_i, alpha_f, lr))
        dqn_feature(game=game, tb_tag=tag, alpha_i=alpha_i, alpha_f=alpha_f, anneal=anneal, lr=lr, freq=freq, clip=clip, hidden=hidden, dist=dist)


    game = 'NChain-v3'
    for (alpha, lr, mem, bs) in hyperparams:
        for i in np.linspace(20, 100, 81)[::2]:
            i = int(i)
            print ('recording N={}, a={}, lr={}, mem={}, bs={}'.format(i, alpha, lr, mem, bs))
            tag='alpha{}-lr{}-rm{}-bs{}N{}'.format(alpha,lr,replay_memory,replay_bs,i),
            dqn_toy_feature(game=game,chain_len=i,tb_tag=tag,alpha=alpha,lr=lr,replay_memory=mem,replay_bs=bs)
