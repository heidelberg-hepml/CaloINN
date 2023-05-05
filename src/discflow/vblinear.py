import math
import sys
import torch
import torch.nn as nn
import numpy as np

class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=1.0, _map=False):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = _map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        self.random = None

    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576; k2 = 1.87320; k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * torch.log(self.mu_w.abs() + 1e-8)
            kl = -th.sum(k1 * torch.sigmoid(k2 + k3 * log_alpha)
                         - 0.5 * F.softplus(-log_alpha) - k1)
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                        - logsig2_w - 1 - np.log(self.prior_prec)).sum()
        return kl

    def forward(self, input):
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return nn.functional.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(-11, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias) + 1e-8

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"
