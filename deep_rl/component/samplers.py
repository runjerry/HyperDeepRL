import torch

# returns a sampler which we can use to sample from a given prior dsitribution
class NoiseSampler(object):
    def __init__(self, dist_type, z_dim, particles=None, p1=None, p2=None):
        self.dist_type = dist_type
        self.z_dim = z_dim
        self.particles = particles
        self.p1 = p1  # optional first moment
        self.p2 = p2  # optional second moment
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
        elif self.dist_type == 'softmax':  # Aux dist affects across particles
            k_classes = self.z_dim
            probs = torch.ones(k_classes)/float(k_classes)
            self.base_dist = torch.distributions.OneHotCategorical(probs=probs)
            high = torch.ones(self.particles, self.z_dim) * .0001
            low = torch.zeros(self.particles, self.z_dim)
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
    
    def sample(self, batch=None):
        sample = self.base_dist.sample()
        if self.aux_dist is not None:
            sample = self.combine_aux(sample)
        return sample
    
    def combine_aux(self, sample):
        sample_aux = self.aux_dist.sample()
        sample = sample.unsqueeze(0).repeat(self.particles, 1)
        sample += sample_aux * 0
        return sample

