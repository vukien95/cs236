import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m, v = self.enc.encode(x)
        eps = ut.sample_gaussian(self.z_prior_m.expand(m.size()), self.z_prior_v.expand(v.size()))
        z_eps = m + eps.mul(v.pow(0.5))
        x_hat = self.dec.decode(z_eps)

        rec = ut.log_bernoulli_with_logits(x, x_hat).mean()
        #kl = ut.kl_normal(m, v, self.z_prior_m.expand(m.size()),
        #                    self.z_prior_v.expand(v.size())).mean()
        hq = ut.log_normal(z_eps, m, v).mean()
        #hz = ut.log_normal(z_eps, self.z_prior_m.expand(m.size()), self.z_prior_v.expand(v.size())).mean()
        kl = hq
        nelbo = kl - rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        niwae = 0
        for i in range(x.size()[0]):
            x_i = x[i][:].view(1, x.size()[1])
            x_i = ut.duplicate(x_i, iw)
            m, v = self.enc.encode(x_i)
            z = ut.sample_gaussian(m, v)
            x_hat = self.dec.decode(z)

            exponent = ut.log_bernoulli_with_logits(x_i, x_hat) + \
                    ut.log_normal(z, self.z_prior_m.expand(m.size()), self.z_prior_v.expand(v.size())) \
                    - ut.log_normal(z, m, v)
            niwae += -ut.log_mean_exp(exponent, 0).squeeze()
        #print(np.std(exponent.data.cpu().numpy()))
        #print(exponent.data.cpu().numpy().shape)
        niwae = niwae / x.size()[0]
        kl = rec = torch.tensor(0)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
