import matplotlib
matplotlib.use('agg')
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_logp(episode_logp, runs):
    plt.figure(4)
    plt.clf()
    loss_t = torch.FloatTensor(episode_logp)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss_t.numpy())
    plt.savefig('runs/{}/TD_loss.png'.format(runs))
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_svgd():
    plt.figure(3)
    plt.clf()
    loss_t = torch.FloatTensor(episode_svgd)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss_t.numpy())
    plt.savefig('runs/{}/svgd_loss.png'.format(runs))
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_rewards(episode_rewards, runs):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.FloatTensor(episode_rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    plt.savefig('runs/{}/reward.png'.format(runs))
    plt.pause(0.001)  # pause a bit so that plots are updated

def orthogonal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)

def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.zero_()
            nn.init.kaiming_normal_(m.weight)

def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.zero_()
            nn.init.xavier_uniform_(m.weight)


