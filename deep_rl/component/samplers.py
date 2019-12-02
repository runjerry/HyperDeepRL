import torch

# returns a sampler which we can use to sample from a given prior dsitribution
class NoiseSampler(object):
    def __init__(self, dist_type, shape, p1=None, p2=None):
        self.dist_type = dist_type
        self.shape = shape
        self.particles = shape[0]
        self.z_dim = shape[1]
        self.p1 = p1
        self.p2 = p2
        self.set_base_sampler()

    def set_base_sampler(self):
        if self.dist_type == 'uniform':
            high = torch.ones(*self.shape)
            low = -1 * high
            self.base_dist = torch.distributions.Uniform(low, high)
        elif self.dist_type == 'normal':
            loc = torch.zeros(*self.shape)
            scale = torch.ones(*self.shape)
            self.base_dist = torch.distributions.Normal(loc, scale)
        elif self.dist_type == 'bernoulli':
            classes = torch.ones(self.z_dim)
            probs = classes/len(classes)
            self.base_dist = torch.distributions.Bernoulli(probs)
        elif self.dist_type == 'categorical':
            k_classes = self.z_dim
            probs = torch.ones(k_classes)/float(k_classes)
            self.base_dist = torch.distributions.OneHotCategorical(k_classes, probs)
        elif self.dist_type == 'multinomial':
            total_count = self.z_dim
            probs = torch.ones(self.z_dim)
            self.base_dist = torch.distributions.Multinomial(total_count, probs)
        elif self.dist_type == 'multivariate_normal':
            loc = torch.zeros(*self.shape)
            rng_mat = torch.rand(self.z_dim, self.z_dim)
            psd_mat = torch.mm(rng_mat, rng_mat.t())
            cov = psd_mat
            self.base_dist = torch.distributions.MultivariateNormal(loc, cov)

    def sample(self, shape=None):
        if shape is not None:
            sample = self.base_dist.sample(sample_shape=shape)
        else:
            sample = self.base_dist.sample()
        print (sample.shape)
        return sample

