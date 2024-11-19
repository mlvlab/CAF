from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy.stats import norm
import torch.distributed as dist
import scipy.stats as stats

def create_named_schedule_sampler(name, step):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(step)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def sample(self):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """

class UniformSampler(ScheduleSampler):
    def __init__(self, step):
        self.step = step

    def sample(self, batch_size, device):
        indices = th.randint(0, self.step+1, (batch_size, )).to(device)
        return indices / self.step


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device):
        if self.even:
            # buckets = [1/G]
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (th.arange(start_i, end_i) + th.rand(bs)) / global_batch_size
            log_sigmas = th.tensor(self.inv_cdf(locs), dtype=th.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * th.randn(bs, device=device)
        sigmas = th.exp(log_sigmas)
        weights = th.ones_like(sigmas)
        return sigmas, weights


def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = th.from_numpy(t).float()
    t = th.cat([t, 1 - t], dim=0)
    t = t[th.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5


    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    return t