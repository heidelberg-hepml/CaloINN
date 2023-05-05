#import numpy as np
import numpy as np
import torch

torch.pi = np.pi
if not hasattr(torch, "arctanh"):
    torch.arctanh = lambda x: 0.5 * torch.log((x+1)/(x-1))


from observables import ObsPT, ObsPhi, ObsEta, ObsM, ObsDeltaPhi
from util import FunctionRegistry


class Preprocessing:
    register = FunctionRegistry()

    def __init__(self, params, data_store):
        self.params = params
        self.data_store = data_store
        self.device = data_store["device"]

    def apply(self, data, forward, init_trafo, disc_steps=False, as_numpy=False):

        if forward and type(data[list(data.keys())[0]]) == np.ndarray:
            data = self.dict_to_torch(data, forward)

        if disc_steps:
            steps = self.params["disc_preprocessing"]
        else:
            steps = self.params["preprocessing_steps"]

        if (not forward) and len(steps):
            steps = steps[::-1]
        for func_name in steps:
            func = Preprocessing.register.functions[func_name]
            data = func(self, data, forward, init_trafo)
        if (not forward) and as_numpy:
            data = self.dict_to_torch(data, forward)
        return data

    def init_pt_obs(self):
        excluded = self.params.get("exclude_pt_obs", [])
        self.pt_obs = [ObsPT(i) for i in range(self.data_store["n_jets"])
                       if i not in excluded]

    def dict_to_torch(self, data, forward):
        if forward:
            for obs in data.keys():
                data[obs] = torch.from_numpy(data[obs]).to(self.device).float()
            return data
        else:
            for obs in data.keys():
                data[obs] = data[obs].detach().cpu().numpy()
            return data

    @register
    def variable_jet_number(self, data, forward, init_trafo):
        if init_trafo:
            self.init_pt_obs()
            pt_data = [data[obs] for obs in self.pt_obs]
            self.pt_min_jets = [torch.min(pt[pt != 0.]) for pt in pt_data]

        if forward:
            return data
        else:
            ret_data = data.copy()
            for i, (obs, pt_min) in enumerate(zip(self.pt_obs, self.pt_min_jets)):
                factor = data[obs] >= pt_min
                ret_data[obs] = factor * data[obs]
                for other_obs in [ObsPhi(i), ObsEta(i), ObsM(i)]:
                    if other_obs in data:
                        ret_data[other_obs] = factor * data[other_obs]
            return ret_data

    @register
    def zero_jet_noise(self, data, forward, init_trafo):
        if init_trafo:
            self.init_pt_obs()
            self.zero_pt_mean = self.params["zero_pt_mean"]
            self.zero_pt_std = self.params["zero_pt_std"]
            etas = [data[ObsEta(i)] for i,_ in enumerate(self.pt_obs)]
            self.zero_eta_means = list(map(torch.mean, etas))
            self.zero_eta_stds = list(map(torch.std, etas))

        if forward:
            ret_data = data.copy()
            n_events = len(data[self.pt_obs[0]])
            for i, pt_obs in enumerate(self.pt_obs):
                pt_dist = torch.randn(n_events) * self.zero_pt_std + self.zero_pt_mean
                eta_dist = torch.randn(n_events) * self.zero_eta_stds[i] \
                                                + self.zero_eta_means[i]
                phi_dist = torch.rand(n_events) * 2 * torch.pi - torch.pi
                zero_mask = data[pt_obs] == 0.
                ret_data[pt_obs] = torch.where(zero_mask, pt_dist, data[pt_obs])
                ret_data[ObsEta(i)] = torch.where(zero_mask, eta_dist, data[ObsEta(i)])
                ret_data[ObsPhi(i)] = torch.where(zero_mask, phi_dist, data[ObsPhi(i)])
            return ret_data
        else:
            return data

    @register
    def pt_logs(self, data, forward, init_trafo):
        """This function calculates pT' = log(pT - pT_min) for all jets
        Can only be used BEFORE make_tensor.
        """
        eps = 5e-5
        if init_trafo:
            self.init_pt_obs()
            pt_data = [data[obs] for obs in self.pt_obs]
            self.pt_min = min(torch.min(pt[pt != 0.]) for pt in pt_data)

        ret_data = data.copy()
        if forward:
            for obs in self.pt_obs:
                ret_data[obs] = torch.log(torch.clip(data[obs] - self.pt_min, eps, None))
        else:
            for obs in self.pt_obs:
                ret_data[obs] = torch.exp(data[obs]) + self.pt_min
        return ret_data

    @register
    def pt_logs_individual(self, data, forward, init_trafo):
        """This function calculates pT' = log(pT - pT_min) for all jets
        with pT_min calculated separately for each jet.
        Can only be used BEFORE make_tensor.
        """
        eps = 5e-5
        if init_trafo:
            self.init_pt_obs()
            self.pt_mins = [torch.min(data[obs][data[obs] != 0.]) for obs in self.pt_obs]

        ret_data = data.copy()
        if forward:
            for obs, pt_min in zip(self.pt_obs, self.pt_mins):
                ret_data[obs] = torch.log(torch.clip(data[obs] - pt_min, min=eps))
        else:
            for obs, pt_min in zip(self.pt_obs, self.pt_mins):
                ret_data[obs] = torch.exp(data[obs]) + pt_min
        return ret_data

    @register
    def pt_logs_diff(self, data, forward, init_trafo):
        """This function calculates pT_n' = log(pT_n - min pT_n) for the last jet
        and pT_i' = log(pT_i - pT_(i+1)) for the other jets
        Can only be used BEFORE make_tensor.
        """
        eps = 5e-5
        if init_trafo:
            self.pt_obs = [ObsPT(i) for i in range(self.data_store["n_jets"])]
            self.pt_min = torch.min(data[self.pt_obs[-1]])

        ret_data = data.copy()
        if forward:
            ret_data[self.pt_obs[-1]] = torch.log(torch.clip(data[self.pt_obs[-1]] -
                                                          self.pt_min, eps, None))
            for obs, prev_obs in zip(self.pt_obs[-2::-1], self.pt_obs[:0:-1]):
                ret_data[obs] = torch.log(torch.clip(data[obs] - data[prev_obs], eps, None))
        else:
            ret_data[self.pt_obs[-1]] = torch.exp(data[self.pt_obs[-1]]) + self.pt_min
            for obs, prev_obs in zip(self.pt_obs[-2::-1], self.pt_obs[:0:-1]):
                ret_data[obs] = ret_data[prev_obs] + torch.exp(data[obs])
        return ret_data

    @register
    def phi_arctanh(self, data, forward, init_trafo):
        """This function applies the function x' = arctanh(x / pi) to all observables
        phi and delta_phi.
        Can only be used BEFORE make_tensor.
        """
        if init_trafo:
            self.phi_obs = [obs for obs in data if type(obs) in [ObsPhi, ObsDeltaPhi]]

        ret_data = data.copy()
        if forward:
            for obs in self.phi_obs:
                 arctanh = torch.arctanh(data[obs] / torch.pi)
                 ret_data[obs] = torch.where(arctanh.isnan(), torch.zeros_like(arctanh), arctanh)
        else:
            for obs in self.phi_obs:
                ret_data[obs] = torch.tanh(data[obs]) * torch.pi
        return ret_data

    @register
    def make_tensor(self, data, forward, init_trafo):
        """This function creates a 2d tensor from the dictionary of 1d observable tensors
        """
        if init_trafo:
            self.observables_order = list(data.keys())

        if forward:
            return torch.stack([data[obs] for obs in self.observables_order], axis=1)
        else:
            return {obs: data[:,i] for i, obs in enumerate(self.observables_order)}

    @register
    def normalize(self, data, forward, init_trafo):
        """This function centers and normalizes the input data.
        Can only be used AFTER make_tensor.
        """
        if init_trafo:
            torch_nan = torch.tensor(np.nan, device=self.device, dtype=torch.float32)
            exclude_cols = self.params.get("norm_exclude_cols", [])
            norm_data = (torch.where(data == 0., torch_nan, data).to(data.device)
                         if self.params.get("norm_exclude_zeros", False) else data)
            self.norm_cols = [i for i in range(norm_data.shape[1]) if i not in exclude_cols]
            norm_data = norm_data[:,self.norm_cols]
            no_nan_counts = (~norm_data.isnan()).sum(dim=0)
            self.norm_means = torch.nansum(norm_data, dim=0) / no_nan_counts
            self.norm_stds = torch.sqrt(torch.nansum(
                        (norm_data - self.norm_means)**2, dim=0) / no_nan_counts)

        ret_data = data.clone()

        if forward:
            ret_data[:,self.norm_cols] = (
                    (data[:,self.norm_cols] - self.norm_means) / self.norm_stds)
        else:
            ret_data[:,self.norm_cols] = (
                    data[:,self.norm_cols] * self.norm_stds + self.norm_means)
        return ret_data

    def get_whitening_matrix(self, data):
        eps = 1e-7
        m = data - torch.mean(data, axis=0)
        sigma = m.T @ m / (data.shape[0] - 1)
        try:
            eigenvalues, eigenvectors = torch.linalg.eig(sigma)
            eigenvalues = torch.abs(eigenvalues) + eps
            eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        except AttributeError:
            eigenvalues, eigenvectors = torch.eig(sigma, eigenvectors=True)
            eigenvalues = torch.sqrt(torch.sum(eigenvalues ** 2, dim=-1)) + eps
        mode = self.params["whitening_mode"]
        if mode == "PCA":
            whitening_matrix = torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
        elif mode == "ZCA":
            whitening_matrix = eigenvectors @ (torch.diag(1/torch.sqrt(eigenvalues)) @
                                               eigenvectors.T)
        else:
            raise RuntimeError(f"Unknown whitening mode {self.mode}")
        try:
            return whitening_matrix, torch.linalg.inv(whitening_matrix)
        except AttributeError:
            return whitening_matrix, torch.inverse(whitening_matrix)



    @register
    def whitening(self, data, forward, init_trafo):
        """This function performs whitening on the input data
        Can only be used AFTER make_tensor.
        Run parameters:
            whitening_mode:
                "PCA": Principle Component Analysis -> W = S^(-1/2) * U^T
                "ZCA": Zero Component Analysis -> W = U * S^(-1/2) * U^T
        """
        if init_trafo:
            exclude_cols = self.params.get("whitening_exclude_cols", [])
            self.whitening_cols = [i for i in range(data.shape[1]) if i not in exclude_cols]
            self.whitening_matrix, self.inverse_whitening_matrix = self.get_whitening_matrix(
                    data[:,self.whitening_cols])



        ret_data = data.clone()
        if forward:
            ret_data[:,self.whitening_cols] = (
                    (self.whitening_matrix.to(ret_data.device) \
                    @ data[:,self.whitening_cols].T).T)
        else:
            ret_data[:,self.whitening_cols] = (
                    (self.inverse_whitening_matrix.to(ret_data.device) \
                    @ data[:,self.whitening_cols].T).T)
        return ret_data

    @register
    def whitening_by_count(self, data, forward, init_trafo):
        n_jets = data[:,0]
        if init_trafo:
            self.network_dims = self.params["network_dims"]
            if self.params.get("whitening_add_dims", False):
                self.network_dims = np.cumsum(self.network_dims)
            self.jet_counts = list(range(self.params["min_jets"], self.params["max_jets"]+1))
            self.whitening_mats = []
            self.inv_whitening_mats = []
            for dim, count in zip(self.network_dims, self.jet_counts):
                wm, iwm = self.get_whitening_matrix(data[n_jets == count,1:dim+1])
                self.whitening_mats.append(wm)
                self.inv_whitening_mats.append(iwm)

        ret_data = data.clone()
        if forward:
            for dim, count, matrix in zip(self.network_dims, self.jet_counts,
                                          self.whitening_mats):
                mask = n_jets == count
                ret_data[mask,1:dim+1] = (matrix @ data[mask,1:dim+1].T).T
        else:
            for dim, count, matrix in zip(self.network_dims, self.jet_counts,
                                          self.inv_whitening_mats):
                mask = n_jets == count
                ret_data[mask,1:dim+1] = (matrix @ data[mask,1:dim+1].T).T
        return ret_data

    @register
    def whitening_by_group(self, data, forward, init_trafo):
        n_jets = data[:,0]
        if init_trafo:
            self.network_dims = self.params["network_dims"]
            self.jet_counts = list(range(self.params["min_jets"], self.params["max_jets"]+1))
            self.whitening_mats = []
            self.inv_whitening_mats = []
            idx = 1
            for dim, count in zip(self.network_dims, self.jet_counts):
                wm, iwm = self.get_whitening_matrix(data[n_jets >= count,idx:idx+dim])
                self.whitening_mats.append(wm)
                self.inv_whitening_mats.append(iwm)
                idx += dim

        ret_data = data.clone()
        idx = 1
        if forward:
            for dim, count, matrix in zip(self.network_dims, self.jet_counts,
                                          self.whitening_mats):
                mask = n_jets >= count
                ret_data[mask,idx:idx+dim] = (matrix @ data[mask,idx:idx+dim].T).T
                idx += dim
        else:
            for dim, count, matrix in zip(self.network_dims, self.jet_counts,
                                          self.inv_whitening_mats):
                mask = n_jets == count
                ret_data[mask,idx:idx+dim] = (matrix @ data[mask,idx:idx+dim].T).T
                idx += dim
        return ret_data

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        # Maybe safer this way than just overwriting dict
        for key in state_dict.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = state_dict[key]
