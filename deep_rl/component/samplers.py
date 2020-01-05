import torch

# returns a sampler which we can use to sample from a given prior dsitribution
class NoiseSampler(object):
    def __init__(self, dist_type, z_dim, particles=None, p1=None, p2=None):
        self.dist_type = dist_type
        self.z_dim = z_dim
        self.particles = particles
        self.p1 = p1
        self.p2 = p2
        self.aux_dist = None
        self.base_dist = None
        self.set_base_sampler()

    def set_base_sampler(self):
        if self.dist_type == 'uniform':
            high = torch.ones(self.z_dim)
            low = -1 * high
            self.base_dist = torch.distributions.Uniform(low, high)

        elif self.dist_type == 'normal':
            loc = torch.zeros(self.z_dim)
            scale = torch.ones(self.z_dim)
            self.base_dist = torch.distributions.Normal(loc, scale)

        elif self.dist_type == 'bernoulli':
            k_classes = torch.ones(self.z_dim)
            probs = k_classes * .5
            self.base_dist = torch.distributions.Bernoulli(probs=probs)

        elif self.dist_type == 'categorical':
            k_classes = self.z_dim
            probs = torch.ones(k_classes)/float(k_classes)
            self.base_dist = torch.distributions.OneHotCategorical(probs=probs)

        elif self.dist_type == 'softmax':
            k_classes = self.z_dim
            probs = torch.ones(k_classes-1)/float(k_classes-1)
            self.base_dist = torch.distributions.OneHotCategorical(probs=probs)
            high = torch.ones(self.particles, 1) * 0.001
            low = torch.ones(self.particles, 1) * 0.0
            self.aux = torch.linspace(0, 1, steps=self.particles)
            mean = torch.ones(self.particles, 1)
            std = torch.ones(self.particles, 1) * 0.3
            self.aux_dist = torch.distributions.Normal(mean, std)
            self.aux = self.aux_dist.sample()
            self.aux = self.aux.view(self.particles, 1)
            self.aux_dist = torch.distributions.Uniform(low, high)

        elif self.dist_type == 'multisoftmax':
            k_classes = self.z_dim
            probs = torch.ones(k_classes-1)/float(k_classes-1)
            self.base_dist = torch.distributions.OneHotCategorical(probs=probs)
            high = torch.ones(3, 1) * 1
            low = torch.ones(3, 1) * 0
            #high = torch.ones(self.particles, self.z_dim) * 1.0
            #low = torch.ones(self.particles, self.z_dim) * 0.9
            self.aux_dist = torch.distributions.Uniform(low, high)

        elif self.dist_type == 'multinomial':
            total_count = self.z_dim
            probs = torch.ones(self.z_dim)
            self.base_dist = torch.distributions.Multinomial(total_count, probs)

        elif self.dist_type == 'multivariate_normal':
            loc = torch.zeros(self.z_dim)
            rng_mat = torch.rand(self.z_dim, self.z_dim)
            psd_mat = torch.mm(rng_mat, rng_mat.t())
            cov = psd_mat
            self.base_dist = torch.distributions.MultivariateNormal(loc, cov)

    def sample(self):
        sample = self.base_dist.sample()
        sample = sample.unsqueeze(0).repeat(self.particles, 1)
        if self.aux_dist is not None:
            # sample = self.base_dist.sample([self.particles])
            #sample = sample.unsqueeze(0).repeat(3, 1, 1)
            # sample_aux = self.aux_dist.sample()
            sample_aux = self.aux
            #sample_aux = sample_aux.repeat(1, self.particles).unsqueeze(-1)
            sample = torch.cat((sample, sample_aux), dim=-1)
            # print (sample)
            #sample *= sample_aux
            # return sample, aux_sample
        return sample

