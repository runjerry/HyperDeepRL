# DeepRL

<div style="text-align:center"><img src="/images/RLpaper_hypergan.png" /></div>


# HyperGAN structure differences

In `deep_rl/utils` there are config definitions for specifying the hypergan structure

`deep_rl/networks/hypergan_ops.py` there is the actual implementation of the Generator factory methods for Linear and Convolutional layers

`deep_rl/networks/hyper_heads.py` has the actual implementation of DDPG and TD3 architectures which actually instantiate these layers

`hyperexamples.py` has the top level calls to these methods. 

==================

Modularized implementation of popular deep RL algorithms by PyTorch. Easy switch between toy tasks and challenging games.

Implemented algorithms:
* Hyper - (Double/Dueling) Deep Q-Learning (DQN)
* Hyper - Deep Deterministic Policy Gradient (DDPG, low-dim-state)
* Hyper - Twined Delayed DDPG (TD3)

# Dependency
* MacOS 10.12 or Ubuntu 16.04
* PyTorch v1.1.0
* Python 3.6, 3.5
* OpenAI Baselines (commit ```8e56dd```)
* Core dependencies: `pip install -e .`

# Remarks
* There is a super fast DQN implementation with an async actor for data generation and an async replay buffer to transfer data to GPU. Enable this implementation by setting `config.async_actor = True` and using `AsyncReplay`. However, with atari games this fast implementation may not work in macOS. Use Ubuntu or Docker instead.
* TensorFlow is used only for logging. Open AI baselines is used very slightly. If you carefully read the code, you should be able to remove/replace them.

# Usage

```examples.py``` contains examples for all the implemented algorithms

```Dockerfile``` contains the environment for generating the curves below. 

Please use this bibtex if you want to cite this repo
```
@misc{deeprl,
  author = {Shangtong, Zhang},
  title = {Modularized Implementation of Deep RL Algorithms in PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ShangtongZhang/DeepRL}},
}
```
