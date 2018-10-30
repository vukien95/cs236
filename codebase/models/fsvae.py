import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.x_v = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)

    def negative_elbo_bound(self, x, y):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m, v = self.enc.encode(x, y)
        z = ut.sample_gaussian(m, v)
        x_m = self.dec.decode(z, y)

        rec = ut.log_normal(x, x_m, self.x_v.expand(x_m.size())).mean()
        kl_z = ut.kl_normal(m, v,
                self.z_prior_m.expand(m.size()),
                self.z_prior_v.expand(v.size())).mean()

        nelbo = kl_z - rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec.decode(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_clipped_x(self, batch):
        z = self.sample_z(batch)
        z = ut.duplicate(z, 10)
        y = np.arange(10)
        y = z.new(np.repeat(np.eye(10)[y], batch, 0))
        x_m = self.compute_mean_given(z, y)
        return x_m
