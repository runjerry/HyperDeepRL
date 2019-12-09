# DeepRL

<p align="center">
  <img width="460" height="300" src="/images/RLpaper_hypergan.png">
</p>

# HyperGAN structure differences

In `deep_rl/utils` there are config definitions for specifying the hypergan structure

`deep_rl/networks/hypergan_ops.py` there is the actual implementation of the Generator factory methods for Linear and Convolutional layers

`deep_rl/networks/hyper_heads.py` has the actual implementation of DDPG and TD3 architectures which actually instantiate these layers

`hyperexamples.py` has the top level calls to these methods. 

==================

Modularized implementation of popular deep RL algorithms by PyTorch. Easy switch between toy tasks and challenging games. Framework implemented by ShongTang, modified by me for use in implicit ensmeble experiments. 

Implemented algorithms:

* Hyper - (Double/Dueling) Deep Q-Learning (DQN)
* Hyper - Deep Deterministic Policy Gradient (DDPG, low-dim-state)
* Hyper - Twined Delayed DDPG (TD3)

Implemeted experiments:

* Atari (branch: master)
* Linear NChain toy experiments (branch: chain)
* Sparse Cartpole Swingup (branch: sparse-cartpole)
* Control Suite (branch: actor-critic)


# Dependency
* MacOS 10.12 or Ubuntu 16.04
* PyTorch v1.1.0
* Python 3.6, 3.5
* OpenAI Baselines (commit ```8e56dd```)
* Core dependencies: `pip install -e .`

# Usage

```hyperexamples.py``` contains examples for implemeneted algorithms in each branch

