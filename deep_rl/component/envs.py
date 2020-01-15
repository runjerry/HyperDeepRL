######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from gym.wrappers import Monitor

import bsuite
from bsuite.utils import gym_wrapper

from ..utils import *
from ..component.wrappers import *
try:
    import roboschool
except ImportError:
    pass

import imageio
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def make_env(env_id, seed, rank, episode_life=True, special_args=None):
    def _thunk():
        random_seed(seed)
        random_seed(seed)
        if env_id.startswith('bsuite'):
            id = env_id.split('bsuite-')[1]
            bsuite_env = bsuite.load_from_id(id)
            env = gym_wrapper.GymFromDMEnv(bsuite_env)
        
        elif env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        
        else:
            if special_args is not None:
                if 'NChain' in special_args[0]:
                    print ('starting chain N = ', special_args[1])
                    env = gym.make(env_id, n=special_args[1])
            else:
                env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)
        
        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_rewards = 0
        self.total_rewards = 0
        self.ep = 0
        self.ep_steps = 0
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_rewards += reward
        self.total_rewards += reward
        self.ep_steps += 1
        if done:
            self.ep += 1
            info['episodic_return'] = self.episode_rewards
            info['episode'] = self.ep
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
            self.episode_rewards = 0
            self.ep_steps = 0
        else:
            info['episodic_return'] = None
            info['episode'] = self.ep
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, name):
        self.name = name
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info
    
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def get_images(self):
        if 'cartpole' in self.name:
            mode = 'cartpole'
        else:
            mode = 'rgb_array'
        return [env.render(mode=mode) for env in self.envs]
    
    def render(self, mode='human'):
        if 'cartpole' in self.name:
            mode='cartpole'
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def close(self):
        return


class GIFlogger(object):
    def __init__(self, log_dir, record_freq=10, max_len=None):
        self.log_dir = log_dir
        self.record_freq = record_freq
        self.max_gif_len = max_len
        self.stored_frames = []
        self.base_name = 'Episode_'
        self.videos_logged = 1
        self.create_gif_directory()

    def reset_frames(self):
        self.stored_frames = []
    
    def add_frame(self, frame):
        self.stored_frames.append(frame)

    def __len__(self):
        return len(self.stored_frames)
    
    def create_gif_directory(self):
        self.save_dir = self.log_dir+'/episode_gif_recordings/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create_and_save(self):
        assert len(self) > 0, "Trying to record GIF with empty buffer"
        fp = self.save_dir+self.base_name+str(self.record_freq*self.videos_logged)+'.gif'
        np_frames = np.array(self.stored_frames)
        imageio.mimsave(fp, np_frames, fps=60)
        self.videos_logged += 1
    

class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5)),
                 video=False,
                 gif=False,
                 video_logging_freq=10,
                 special_args=None):

        self.log_dir = log_dir
        self.video_freq = video_logging_freq
        self.video_enabled = video
        self.gif_enabled = gif
        self.videos_logged = 0
        if self.gif_enabled:
            self.gif_logger = GIFlogger(log_dir, video_logging_freq)
        if video == False and gif == False:
            self.record = False
        else:
            self.record = True

        self.record_now = False
    
        self.name = name
        envs = [self.make_env(name, seed, i, episode_life, special_args) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs, self.name)
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        if 'Chain' in self.name:
            self.state_dim = self.env.envs[0].n
        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    # adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
    def make_env(self, env_id, seed, rank, episode_life=True, special_args=None):
        def _thunk():
            random_seed(seed)
            if env_id.startswith('bsuite'):
                id = env_id.split('bsuite-')[1]
                self.video_enabled = False
                bsuite_env = bsuite.load_from_id(id)
                env = gym_wrapper.GymFromDMEnv(bsuite_env)
            
            elif env_id.startswith("dm"):
                import dm_control2gym
                _, domain, task = env_id.split('-')
                env = dm_control2gym.make(domain_name=domain, task_name=task)
            
            else:
                if special_args is not None:
                    if 'NChain' in special_args[0]:
                        print ('starting chain N = ', special_args[1])
                        env = gym.make(env_id, n=special_args[1])
                else:
                    env = gym.make(env_id)

            if self.video_enabled:
                env = Monitor(env, self.log_dir, video_callable=self.video_callable)

            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(env_id)
            env.seed(seed + rank)
            env = OriginalReturnWrapper(env)
            if is_atari:
                env = wrap_deepmind(env,
                                    episode_life=episode_life,
                                    clip_rewards=False,
                                    frame_stack=False,
                                    scale=False)
                obs_shape = env.observation_space.shape
                if len(obs_shape) == 3:
                    env = TransposeImage(env)
                env = FrameStack(env, 4)
            return env

        return _thunk


    def reset(self):
        return self.env.reset()
    
    def record_or_not(self, info):
        if info[0]['episode'] % self.video_freq == 0:
            self.record_now = True
            self.start_record()
        else:
            self.record_now = False
            self.end_record()

    def start_record(self):
        self.gif_logger.reset_frames()

    def end_record(self):
        if len(self.gif_logger) > 0:
            self.gif_logger.create_and_save()
            self.gif_logger.reset_frames()

    def render(self):
        if self.record_now and self.gif_enabled:
            frame = self.env.render()
            self.gif_logger.add_frame(frame)
        return frame

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
