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
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, FCHyperBody(config.state_dim))
    # config.replay_fn = lambda: Replay(memory_size=int(1e5), batch_size=100)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(config.replay_memory_size), batch_size=int(config.replay_bs))

    config.random_action_prob = LinearSchedule(0.1, 0.001, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 0
    config.double_q = True
    # config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 500e3
    config.async_actor = False
    config.alpha_anneal = config.max_steps
    config.alpha_init = config.alpha
    config.alpha_final = config.alpha
    # run_steps(DQNAgent(config))
    # run_steps(DQN_SVGD_Agent(config))
    run_steps(DQN_Dist_SVGD_Agent(config))


def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game, num_envs=1, single_process=True)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    # config.network_fn = lambda: VanillaHyperNet(config.action_dim, NatureConvHyperBody(in_channels=config.history_length))
    config.network_fn = lambda: DuelingHyperNet(config.action_dim, NatureConvHyperBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=128)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    # config.double_q = True
    config.double_q = True
    config.max_steps = int(200e7)
    run_steps(DQN_Dist_Agent(config))


# QR DQN
def quantile_regression_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: QuantileHyperNet(config.action_dim, config.num_quantiles, FCHyperBody(config.state_dim))

    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e4), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.num_quantiles = 20
    config.gradient_clip = 5
    config.sgd_update_frequency = 4
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    run_steps(QuantileRegressionDQNAgent(config))


def quantile_regression_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileHyperNet(config.action_dim, config.num_quantiles, NatureConvHyperBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.num_quantiles = 200
    config.max_steps = int(2e7)
    run_steps(QuantileRegressionDQNAgent(config))


# C51
def categorical_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalHyperNet(config.action_dim, config.categorical_n_atoms, FCHyperBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)

    # config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=10000, batch_size=10)

    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    config.gradient_clip = 5
    config.sgd_update_frequency = 4

    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    run_steps(CategoricalDQNAgent(config))


def categorical_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalHyperNet(config.action_dim, config.categorical_n_atoms, NatureConvHyperBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(2e7)
    run_steps(CategoricalDQNAgent(config))


# A2C
def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticHyperNet(
        config.state_dim, config.action_dim, FCHyperBody(config.state_dim, gate=F.tanh))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    run_steps(A2CAgent(config))


def a2c_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticHyperNet(config.state_dim, config.action_dim, NatureConvHyperBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(A2CAgent(config))


def a2c_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda: GaussianActorCriticHyperNet(
        config.state_dim, config.action_dim,
        actor_body=FCHyperBody(config.state_dim), critic_body=FCHyperBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(A2CAgent(config))


# N-Step DQN
def n_step_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 5
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaHyperNet(config.action_dim, FCHyperBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.gradient_clip = 5
    run_steps(NStepDQNAgent(config))


def n_step_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: VanillaHyperNet(config.action_dim, NatureConvHyperBody())
    config.random_action_prob = LinearSchedule(1.0, 0.05, 1e6)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(NStepDQNAgent(config))


# Option-Critic
def option_critic_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticHyperNet(FCHyperBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    run_steps(OptionCriticAgent(config))


def option_critic_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticHyperNet(NatureConvHyperBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(OptionCriticAgent(config))


# PPO
def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticHyperNet(config.state_dim, config.action_dim, FCHyperBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * 5 * 10
    run_steps(PPOAgent(config))


def ppo_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticHyperNet(config.state_dim, config.action_dim, NatureConvHyperBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = int(2e7)
    run_steps(PPOAgent(config))


def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.hyper = True

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticHyperNet(
        config.state_dim, config.action_dim, actor_body=FCHyperBody(config.state_dim, gate=torch.tanh),
        critic_body=FCHyperBody(config.state_dim, gate=torch.tanh))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 1e6
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


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
    print (config.state_dim, config.action_dim)

    config.network_fn = lambda: DeterministicActorCriticHyperNet(
        config.state_dim, config.action_dim,
        actor_body=FCHyperBody(config.state_dim, (400, 300), gate=F.relu),
        #critic_body=TwoLayerFCHyperBodyWithAction(config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.AdamW(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.AdamW(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3
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

    config.network_fn = lambda: TD3HyperNet(
        config.action_dim,
        actor_body_fn=lambda: FCHyperBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=512)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)
    args = load_args()

    # Grid search for HP
    alphas = [1e-2, 1e-1, 1, 1e1, 1e2]   
    #learning_rates = [1e-2, 1e-3, 5e-4, 2e-4, 1e-4]
    #replay_memory_sizes = [1e3, 1e5]
    #replay_batch_sizes = [32, 64, 128]
    learning_rates = [1e-3, 5e-4, 2e-4, 1e-4]
    replay_memory_sizes = [1e6]
    replay_batch_sizes = [32, 64, 128]
    hyperparams = itertools.product(*[alphas, learning_rates, replay_memory_sizes, replay_batch_sizes])
    # game = 'CartPole-v0'  # MUST WORK
    # game = 'bsuite-cartpole/0'
    game = 'bsuite-cartpole_swingup/0'
    for (alpha, lr, mem, bs) in hyperparams:
        tag='alpha{}-lr{}-rm{}-bs{}'.format(alpha, lr, mem, bs),
        print ('recording a={}, lr={}, mem={}, bs={}'.format(alpha, lr, mem, bs))
        dqn_feature(game=game, tb_tag=tag, alpha=alpha, lr=lr, replay_memory_size=mem, replay_bs=bs)

    game = 'NChain-v3'
    for (alpha, lr, mem, bs) in hyperparams:
        for i in np.linspace(20, 100, 81)[::2]:
            i = int(i)
            print ('recording N={}, a={}, lr={}, mem={}, bs={}'.format(i, alpha, lr, mem, bs))
            tag='alpha{}-lr{}-rm{}-bs{}N{}'.format(alpha,lr,replay_memory,replay_bs,i),
            dqn_toy_feature(game=game,chain_len=i,tb_tag=tag,alpha=alpha,lr=lr,replay_memory=mem,replay_bs=bs)

    # quantile_regression_dqn_feature(game=game)
    # categorical_dqn_feature(game=game)
    # a2c_feature(game=game)
    # n_step_dqn_feature(game=game)
    # option_critic_feature(game=game)
    # ppo_feature(game=game)
    
    # game = 'HalfCheetah-v2'
    # game = 'Ant-v2'
    # game = 'Reacher-v2'
    # game = 'InvertedPendulum-v2'
    # game = 'InvertedDoublePendulum-v2'
    # game = 'Humanoid-v2'
    # game = 'Hopper-v2'
    # game = 'Swimmer-v2'
    # game = 'Walker2d-v2'
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    # td3_continuous(game=game)
    # games = ['HalfCheetah-v2',
    #        'Humanoid-v2',
    #        'Ant-v2',
    #        'Reacher-v2',
    #        'InvertedDoublePendulum-v2',
    #        'Hopper-v2',
    #        'Swimmer-v2',
    #        'Walker2d-v2'
    # ]
    # for i in range(3):
    #     for game in games:
    #         td3_continuous(game=game, tb_tag='td3_{}-{}'.format(game, i))

    # game = 'BreakoutNoFrameskip-v4'
    # dqn_pixel(game=game)
    # quantile_regression_dqn_pixel(game=game)
    # categorical_dqn_pixel(game=game)
    # a2c_pixel(game=game)
    # n_step_dqn_pixel(game=game)
    # option_critic_pixel(game=game)
    # ppo_pixel(game=game)
