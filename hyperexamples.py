######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import envs

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
    run_steps(DQNDistToySVGD_Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)
    args = load_args()

    switch = 4

    if switch == 1:
        a = [1e2]   
        learning_rate = [1e-2, 1e-3, 5e-4, 2e-4, 1e-4]
        replay_memory_size = [1e3, 1e5]
        replay_batch_size = [32, 128]

    if switch == 2:
        a = [1e1]   
        learning_rate = [1e-2, 1e-3, 5e-4, 2e-4, 1e-4]
        replay_memory_size = [1e3, 1e5]
        replay_batch_size = [32, 128]

    if switch == 3:
        a = [1e-1]   
        learning_rate = [1e-2, 1e-3, 5e-4, 2e-4, 1e-4]
        replay_memory_size = [1e3, 1e5]
        replay_batch_size = [32, 128]

    if switch == 4:
        a = [1e-3, 1]   
        learning_rate = [1e-2, 1e-3, 5e-4, 2e-4, 1e-4]
        replay_memory_size = [1e3, 1e5]
        replay_batch_size = [32, 128]
    

    game = 'NChain-v3'
    for alpha in a:
        for lr in learning_rate:
            for replay_memory in replay_memory_size:
                for replay_bs in replay_batch_size:
                    for i in np.linspace(20, 100, 81)[::2]:
                        i = int(i)
                        dqn_toy_feature(game=game,
                                        tb_tag='alpha{}-lr{}-rm{}-bs{}N{}'.format(alpha,lr,replay_memory,replay_bs,i),
                                        chain_len=i,
                                        alpha=alpha,
                                        lr=lr,
                                        replay_memory=replay_memory,
                                        replay_bs=replay_bs)
