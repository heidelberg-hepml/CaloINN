import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from FrEIA.modules import InvertibleModule
import numpy as np
import math

def random_perm_matrix(dim):
    w = torch.zeros((dim, dim))
    for i, j in enumerate(np.random.permutation(dim)):
        w[i, dim-i-1] = 1.
    return w

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)

class HierarchicalCouplingBlock(InvertibleModule):
    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_EPS = 1e-5
    DEFAULT_QUADRATIC_THRESHOLD = 1e-3

    def __init__(self, dims_in, dims_c, input_split, subnet_constructor, permute_soft,
                 num_bins, bounds, combined_mode):
        assert sum(input_split) == dims_in[0][0]
        assert len(input_split) == dims_c[0][0]
        super().__init__(dims_in, dims_c)

        self.input_dims = input_split
        self.num_bins = num_bins
        self.bounds = bounds
        self.combined_mode = combined_mode

        if permute_soft:
            ws = [special_ortho_group.rvs(dim) for dim in self.input_dims]
        else:
            ws = [random_perm_matrix(dim) for dim in self.input_dims]
        self.w_perms = [nn.Parameter(torch.FloatTensor(w), requires_grad=False)
                        for w in ws]
        self.w_perms_inv = [nn.Parameter(torch.FloatTensor(w.T), requires_grad=False)
                            for w in ws]

        self.n_dims = len(self.input_dims)
        self.splits = [(dim - dim//2, dim//2) for dim in self.input_dims]
        self.cond_dims = list(range(self.n_dims, 1, -1)) + [0]
        cubic_idim = 2*num_bins + 2
        if combined_mode:
            self.sc_subnets = nn.ModuleList([
                    subnet_constructor(s0 + cdim + sum(self.input_dims[:i]), cubic_idim * s1)
                    for i, ((s0, s1), cdim) in enumerate(zip(self.splits, self.cond_dims))])
        else:
            self.sc_subnets = nn.ModuleList([
                    subnet_constructor(s0 + cdim, cubic_idim * s1)
                    for (s0, s1), cdim in zip(self.splits, self.cond_dims)])
            self.c_subnets = nn.ModuleList([
                    subnet_constructor(sum(self.input_dims[:i]) + self.cond_dims[i],
                                       cubic_idim * self.input_dims[i])
                    for i in range(1, self.n_dims)])

    def forward(self, x, c, rev=False, jac=True):
        x, c = x[0], c[0]
        x_split = list(torch.split(x, self.input_dims, dim=1))
        masks = [torch.any(c[:,i:].bool(), dim=1) for i in range(self.n_dims)]
        jacs = torch.zeros(x.shape[0], 2*self.n_dims-1, dtype=torch.float, device=x.device)
        cond_max = c.shape[1]

        if rev:
            if not self.combined_mode:
                for i, c_subnet, cond_dim, mask in zip(range(1, self.n_dims), self.c_subnets,
                                                       self.cond_dims[1:], masks[1:]):
                    masked_split = [s[mask,:] for s in x_split[:i]]
                    num_cond = c[mask,cond_max-cond_dim:]
                    sub_out = c_subnet(torch.cat([*masked_split, num_cond], dim=1))
                    sub_out = sub_out.reshape(sub_out.shape[0], -1, 2*self.num_bins + 2)
                    coup_out, bj = self.coupling_func(x_split[i][mask,:], sub_out, rev)
                    x_split[i] = x_split[i].masked_scatter(mask[:,None], coup_out)
                    jacs[mask,i+self.n_dims-1] = bj

            for i, (perm, csplit, sc_subnet, cond_dim, mask) in reversed(list(enumerate(zip(
                    self.w_perms_inv, self.splits, self.sc_subnets, self.cond_dims, masks)))):
                cond, data = torch.split(x_split[i][mask,:], csplit, dim=1)
                num_cond = c[mask,cond_max-cond_dim:]
                if self.combined_mode:
                    masked_split = [s[mask,:] for s in x_split[:i]]
                    all_cond = (cond, *masked_split, num_cond)
                else:
                    all_cond = (cond, num_cond)
                sub_out = sc_subnet(torch.cat(all_cond, dim=1))
                sub_out = sub_out.reshape(sub_out.shape[0], -1, 2*self.num_bins + 2)
                xs, bj = self.coupling_func(data, sub_out, rev)
                jacs[mask,i] = bj
                xs_rot = F.linear(torch.cat((cond, xs), dim=1), perm.to(x.device))
                x_split[i] = x_split[i].masked_scatter(mask[:,None], xs_rot)

        else:
            for i, (perm, csplit, sc_subnet, cond_dim, mask) in enumerate(zip(self.w_perms,
                    self.splits, self.sc_subnets, self.cond_dims, masks)):
                xs = F.linear(x_split[i][mask,:], perm.to(x.device))
                cond, data = torch.split(xs, csplit, dim=1)
                num_cond = c[mask,cond_max-cond_dim:]
                if self.combined_mode:
                    masked_split = [s[mask,:] for s in x_split[:i]]
                    all_cond = (cond, *masked_split, num_cond)
                else:
                    all_cond = (cond, num_cond)
                sub_out = sc_subnet(torch.cat(all_cond, dim=1))
                sub_out = sub_out.reshape(sub_out.shape[0], -1, 2*self.num_bins + 2)
                coup_out, fj = self.coupling_func(data, sub_out, rev)
                xs = torch.cat((cond, coup_out), dim=1)
                x_split[i] = x_split[i].masked_scatter(mask[:,None], xs)
                jacs[mask,i] = fj

            if not self.combined_mode:
                for i, c_subnet, cond_dim, mask in reversed(list(zip(range(1, self.n_dims),
                        self.c_subnets, self.cond_dims[1:], masks[1:]))):
                    masked_split = [s[mask,:] for s in x_split[:i]]
                    num_cond = c[mask,cond_max-cond_dim:]
                    sub_out = c_subnet(torch.cat([*masked_split, num_cond], dim=1))
                    sub_out = sub_out.reshape(sub_out.shape[0], -1, 2*self.num_bins + 2)
                    coup_out, fj = self.coupling_func(x_split[i][mask,:], sub_out, rev)
                    x_split[i] = x_split[i].masked_scatter(mask[:,None], coup_out)
                    jacs[mask,i+self.n_dims-1] = fj

        return (torch.cat(x_split, dim=1),), torch.sum(jacs, dim=1)

    def coupling_func(self, inputs, theta, rev):
        inside_interval_mask = torch.all((inputs >= -self.bounds) & (inputs <= self.bounds),
                                         dim = -1)
        outside_interval_mask = ~inside_interval_mask

        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], device=inputs.device)
        masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
        masked_logabsdet[outside_interval_mask] = 0

        inputs = inputs[inside_interval_mask]
        theta = theta[inside_interval_mask, :]

        min_bin_width=self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height=self.DEFAULT_MIN_BIN_HEIGHT
        eps=self.DEFAULT_EPS
        quadratic_threshold=self.DEFAULT_QUADRATIC_THRESHOLD

        if not rev and (torch.min(inputs).item() < -self.bounds or
                        torch.max(inputs).item() > self.bounds):
            raise ValueError("Spline Block inputs are not within boundaries")
        elif rev and (torch.min(inputs).item() < -self.bounds or
                      torch.max(inputs).item() > self.bounds):
            raise ValueError("Spline Block inputs are not within boundaries")

        unnormalized_widths = theta[...,:self.num_bins]
        unnormalized_heights = theta[...,self.num_bins:self.num_bins*2]
        unnorm_derivatives_left = theta[...,-2].reshape(theta.shape[0], -1, 1)
        unnorm_derivatives_right = theta[...,-1].reshape(theta.shape[0], -1, 1)

        inputs = (inputs + self.bounds) / (2*self.bounds)

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths

        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths[..., -1] = 1
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights

        cumheights = torch.cumsum(heights, dim=-1)
        cumheights[..., -1] = 1
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

        slopes = heights / widths
        min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                    torch.abs(slopes[..., 1:]))
        min_something_2 = (
                0.5 * (widths[..., 1:]*slopes[..., :-1] + widths[..., :-1]*slopes[..., 1:])
                / (widths[..., :-1] + widths[..., 1:])
        )
        min_something = torch.min(min_something_1, min_something_2)

        derivatives_left = (torch.sigmoid(unnorm_derivatives_left) * 3
                            * slopes[..., 0][..., None])
        derivatives_right = (torch.sigmoid(unnorm_derivatives_right) * 3
                             * slopes[..., -1][..., None])

        derivatives = min_something * (torch.sign(slopes[..., :-1]) +
                                       torch.sign(slopes[..., 1:]))
        derivatives = torch.cat([derivatives_left,
                                 derivatives,
                                 derivatives_right], dim=-1)
        a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths.pow(2)
        b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
        c = derivatives[..., :-1]
        d = cumheights[..., :-1]

        if rev:
            bin_idx = searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = searchsorted(cumwidths, inputs)[..., None]

        inputs_a = a.gather(-1, bin_idx)[..., 0]
        inputs_b = b.gather(-1, bin_idx)[..., 0]
        inputs_c = c.gather(-1, bin_idx)[..., 0]
        inputs_d = d.gather(-1, bin_idx)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

        if rev:
            # Modified coefficients for solving the cubic.
            inputs_b_ = (inputs_b / inputs_a) / 3.
            inputs_c_ = (inputs_c / inputs_a) / 3.
            inputs_d_ = (inputs_d - inputs) / inputs_a

            delta_1 = -inputs_b_.pow(2) + inputs_c_
            delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
            delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

            discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

            depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
            depressed_2 = delta_1

            three_roots_mask = discriminant >= 0
            one_root_mask = discriminant < 0

            outputs = torch.zeros_like(inputs)

            # Deal with one root cases.

            p = cbrt((-depressed_1[one_root_mask] +
                      torch.sqrt(-discriminant[one_root_mask])) / 2.)
            q = cbrt((-depressed_1[one_root_mask] -
                      torch.sqrt(-discriminant[one_root_mask])) / 2.)

            outputs[one_root_mask] = ((p + q)
                                      - inputs_b_[one_root_mask]
                                      + input_left_cumwidths[one_root_mask])

            # Deal with three root cases.

            theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]),
                                -depressed_1[three_roots_mask])
            theta /= 3.

            cubic_root_1 = torch.cos(theta)
            cubic_root_2 = torch.sin(theta)

            root_1 = cubic_root_1
            root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
            root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

            root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
            root_shift = (- inputs_b_[three_roots_mask]
                          + input_left_cumwidths[three_roots_mask])

            root_1 = root_1 * root_scale + root_shift
            root_2 = root_2 * root_scale + root_shift
            root_3 = root_3 * root_scale + root_shift

            root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
            root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
            root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
            root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            roots = torch.stack([root_1, root_2, root_3], dim=-1)
            masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
            mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
            outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=mask_index).view(-1)

            # Deal with a -> 0 (almost quadratic) cases.

            quadratic_mask = inputs_a.abs() < quadratic_threshold
            a = inputs_b[quadratic_mask]
            b = inputs_c[quadratic_mask]
            c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
            alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c)) / (2*a)
            outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

            shifted_outputs = (outputs - input_left_cumwidths)
            logabsdet = -torch.log((3 * inputs_a * shifted_outputs.pow(2) +
                                    2 * inputs_b * shifted_outputs +
                                    inputs_c))
        else:
            shifted_inputs = (inputs - input_left_cumwidths)
            outputs = (inputs_a * shifted_inputs.pow(3) +
                       inputs_b * shifted_inputs.pow(2) +
                       inputs_c * shifted_inputs +
                       inputs_d)

            logabsdet = torch.log((3 * inputs_a * shifted_inputs.pow(2) +
                                   2 * inputs_b * shifted_inputs +
                                   inputs_c))

        logabsdet = torch.sum(logabsdet, dim=1)
        outputs = outputs * 2 * self.bounds - self.bounds
        masked_outputs[inside_interval_mask] = outputs
        masked_logabsdet[inside_interval_mask] = logabsdet

        return masked_outputs, masked_logabsdet

    def output_dims(self, input_dims):
        return input_dims
