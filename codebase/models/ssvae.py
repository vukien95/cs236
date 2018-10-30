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

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        self.y_prior = torch.nn.Parameter(torch.tensor([1 / self.y_dim]), requires_grad=False)

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
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)

        m, v = self.enc.encode(x, y)
        z = ut.sample_gaussian(m, v)
        x_hat = self.dec.decode(z, y)

        kl_y = ut.kl_cat(y_prob, y_logprob, self.y_prior.expand(y_prob.size()).log()).mean()

        elementwise_kl_z = ut.kl_normal(m, v, self.z_prior_m.expand(m.size()),
                self.z_prior_v.expand(v.size())).view(self.y_dim, -1).t()
        elementwise_rec = -ut.log_bernoulli_with_logits(x, x_hat).view(self.y_dim, -1).t()

        kl_z = (y_prob * elementwise_kl_z).sum(dim=1).mean()
        rec = (y_prob * elementwise_rec).sum(dim=1).mean()

        #kl_z = rec = 0
        #for i in range(elementwise_kl_z.size()[1]):
        #    kl_z += torch.dot(elementwise_kl_z[:, i].squeeze(), y_prob[i, :].squeeze())
        #    rec += torch.dot(elementwise_rec[:, i].squeeze(), y_prob[i, :].squeeze())
        #kl_z /= elementwise_kl_z.size()[1]
        #rec /= elementwise_kl_z.size()[1]

        nelbo = kl_y + kl_z + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
