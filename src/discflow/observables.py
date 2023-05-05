import numpy as np
import torch
import math

torch.pi = np.pi
torch.clip = torch.clamp


class Observable:
    calculate_call_stack = []

    def __init__(self, *indices):
        self.indices = indices

    def from_data(self, data):
        if self not in data:
            if self in Observable.calculate_call_stack:
                self.not_found()
            else:
                Observable.calculate_call_stack.append(self)
                try:
                    data[self] = self.calculate(data)
                finally:
                    Observable.calculate_call_stack.pop()
        return data[self]

    def calculate(self, data):
        self.not_found()

    def flat_indices(self):
        ret = []
        for idx in self.indices:
            if isinstance(idx, JetSum) and not isinstance(idx, InclJetSum):
                ret.extend(idx.indices)
            else:
                ret.append(idx)
        return ret

    def not_found(self):
        raise KeyError(f"Observable {self} not found in data")

    def tex_name(self, params):
        all_jet_names = {JetSum(*key) if isinstance(key,list) else key: name
                         for key, name in params.get("jet_names", [])}
        jet_names = tuple(all_jet_names[i] if i in all_jet_names else
                          (" ".join(str(j+1) for j in i.indices) if isinstance(i, JetSum)
                           else str(i+1)) for i in self.indices)
        return self.tex_template % jet_names

    def __eq__(self, other):
        return type(self) == type(other) and self.indices == other.indices

    def __hash__(self):
        return hash((type(self), *self.indices))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__ + str(self.indices)

class JetSum:
    def __init__(self, *indices):
        self.indices = indices

    def sum_obs(self, obs_class, data):
        ret = 0.
        for i in self.indices:
            ret += obs_class(i).from_data(data)
        return ret

    def __eq__(self, other):
        return type(self) == type(other) and self.indices == other.indices

    def __hash__(self):
        return hash((*self.indices, ))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "+".join(str(i) for i in self.indices)
 
class InclJetSum(JetSum):
    def sum_obs(self, obs_class, data):
        ret = 0.
        count = ObsCount().from_data(data)
        for i in self.indices:
            ret += np.where(count > i, obs_class(i).from_data(data), 0.)
        return ret

class ObsE(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "E_{%s}"
        self.unit = "GeV"

    def calculate(self, data):
        index = self.indices[0]
        if isinstance(index, JetSum):
            return index.sum_obs(ObsE, data)
        else:
            m = ObsM(index).from_data(data)
            pt = ObsPT(index).from_data(data)
            eta = ObsEta(index).from_data(data)
            module = checktype(pt)
            return module.sqrt(m**2 + (pt * module.cosh(eta))**2)

class ObsPx(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "p_{x,%s}"
        self.unit = "GeV"

    def calculate(self, data):
        index = self.indices[0]
        if isinstance(index, JetSum):
            return index.sum_obs(ObsPx, data)
        else:
            pt = ObsPT(index).from_data(data)
            phi = ObsPhi(index).from_data(data)
            module = checktype(pt)
            return pt * module.cos(phi)

class ObsPy(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "p_{y,%s}"
        self.unit = "GeV"

    def calculate(self, data):
        index = self.indices[0]
        if isinstance(index, JetSum):
            return index.sum_obs(ObsPy, data)
        else:
            pt = ObsPT(index).from_data(data)
            phi = ObsPhi(index).from_data(data)
            module = checktype(pt)
            return pt * module.sin(phi)

class ObsPz(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "p_{z,%s}"
        self.unit = "GeV"

    def calculate(self, data):
        index = self.indices[0]
        if isinstance(index, JetSum):
            return index.sum_obs(ObsPz, data)
        else:
            pt = ObsPT(index).from_data(data)
            eta = ObsEta(index).from_data(data)
            module = checktype(pt)
            return pt * module.sinh(eta)

class ObsCount(Observable):
    def __init__(self):
        super().__init__()
        self.tex_template = "n_{\\mathrm{jets}}"
        self.unit = None

    def bins(self, data):
        count = self.from_data(data)
        module = checktype(count)
        return module.arange(module.min(count) - 0.5, module.max(count) + 1)

    def calculate(self, data):
        pt_nonzero = [obs.from_data(data) != 0. for obs in data if type(obs) == ObsPT]
        if len(pt_nonzero) == 0:
            pt_nonzero = [obs.from_data(data) != 0. for obs in data if type(obs) == ObsE]
        module = checktype(pt_nonzero[0])
        return module.sum(module.stack(pt_nonzero, axis=1), axis=1)

class ObsPT(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "p_{T,%s}"
        self.unit = "GeV"

    def bins(self, data):
        pt = self.from_data(data)
        module = checktype(pt)
        cutoff = module.sort(pt)[math.ceil(len(pt)*0.99)]
        return module.linspace(0, cutoff, 60)

    def calculate(self, data):
        try:
            px = ObsPx(self.indices[0]).from_data(data)
            py = ObsPy(self.indices[0]).from_data(data)
            module = checktype(px)
            return module.sqrt(px**2 + py**2)
        except KeyError as error:
            if isinstance(self.indices[0], JetSum):
                pt_sq = 0
                for index in self.indices[0].indices:
                    pt_sq += ObsPT(index).from_data(data) ** 2
                module = checktype(pt_sq)
                #Probably an error here if we try to get PT of more than 2 jets
                for index1 in self.indices[0].indices:
                    for index2 in self.indices[0].indices:
                        if index1 > index2:
                            pt_prod = ObsPT(index1).from_data(data)*ObsPT(index2).from_data(data)
                            pt_sq += 2*module.cos(ObsDeltaPhi(index1, index2).from_data(data))*pt_prod
                return module.sqrt(pt_sq)
            else:
                raise(KeyError(error))

class ObsPhi(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "\phi_{%s}"
        self.unit = None

    def bins(self, data):
        return np.linspace(-3.5, 3.5, 60)

    def calculate(self, data):
        try:
            px = ObsPx(self.indices[0]).from_data(data)
            py = ObsPy(self.indices[0]).from_data(data)
            module = checktype(px)
            return module.arctan2(py, px)
        except KeyError:
            phi_indices = [obs.indices[0] for obs in data if type(obs) == ObsPhi]
            graph = [obs.indices for obs in data if type(obs) == ObsDeltaPhi]
            for phi_index in phi_indices:
                route = find_route(graph, self.indices[0], phi_index)
                if route is not None:
                    break
            else:
                self.not_found()
            phi = 0
            phi += ObsPhi(phi_index).from_data(data)
            for a,b,sign in route:
                phi += sign * data[ObsDeltaPhi(a,b)]
            module = checktype(phi)
            return (phi + module.pi) % (2*module.pi) - module.pi

class ObsEta(Observable):
    def __init__(self, jet_index):
        super().__init__(jet_index)
        self.tex_template = "\eta_{%s}"
        self.unit = None

    def bins(self, data):
        return np.linspace(-6, 6, 60)

    def calculate(self, data):
        try:
            px = ObsPx(self.indices[0]).from_data(data)
            py = ObsPy(self.indices[0]).from_data(data)
            pz = ObsPz(self.indices[0]).from_data(data)
            module = checktype(px)
            p = module.sqrt(px**2 + py**2 + pz**2)
            eps = 1e-15
            return 0.5 * (module.log(module.clip(module.abs(p + pz), eps, None)) -
                          module.log(module.clip(module.abs(p - pz), eps, None)))
        except KeyError:
            eta_indices = [obs.indices[0] for obs in data if type(obs) == ObsEta]
            graph = [obs.indices for obs in data if type(obs) == ObsDeltaEta]
            for eta_index in eta_indices:
                route = find_route(graph, self.indices[0], eta_index)
                if route is not None:
                    break
            else:
                self.not_found()
            return (ObsEta(eta_index).from_data(data) +
                    sum(sign * data[ObsDeltaEta(a,b)] for a,b,sign in route))

class ObsM(Observable):
    def __init__(self, jet_index, n_bins=60, low_cutoff=None, high_cutoff=None):
        super().__init__(jet_index)
        self.tex_template = "M_{%s}"
        self.unit = "GeV"
        self.n_bins = n_bins
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def bins(self, data):
        masses = self.from_data(data)
        module = checktype(masses)
        sorted_mass = module.sort(masses)
        low_cutoff = (sorted_mass[math.ceil(len(sorted_mass)*0.002)]
                      if self.low_cutoff is None else self.low_cutoff)
        high_cutoff = (sorted_mass[math.ceil(len(sorted_mass)*0.998)]
                       if self.high_cutoff is None else self.high_cutoff)
        return module.linspace(low_cutoff, high_cutoff, self.n_bins)

    def calculate(self, data):
        e  = ObsE(self.indices[0]).from_data(data)
        try:
            px = ObsPx(self.indices[0]).from_data(data)
            py = ObsPy(self.indices[0]).from_data(data)
            pt_sq = px**2 + py**2
        except KeyError:
            pt_sq = ObsPT(self.indices[0]).from_data(data)**2
        pz = ObsPz(self.indices[0]).from_data(data)
        module = checktype(pt_sq)
        return module.sqrt(module.clip(e**2 - pt_sq - pz**2, 1e-6, None))

class ObsDeltaPhi(Observable):
    def __init__(self, jet_index1, jet_index2):
        super().__init__(jet_index1, jet_index2)
        self.tex_template = "\Delta \phi_{%s %s}"
        self.unit = None

    def calculate(self, data):
        try:
            phi1 = ObsPhi(self.indices[0]).from_data(data)
            phi2 = ObsPhi(self.indices[1]).from_data(data)
            module = checktype(phi1)
            return (phi1 - phi2 + module.pi) % (2*module.pi) - module.pi
        except KeyError:
            graph = [obs.indices for obs in data if type(obs) == ObsDeltaPhi]
            route = find_route(graph, *self.indices)
            if route is None:
                self.not_found()
            delta_phi = 0
            for a,b,sign in route:
                delta_phi += sign * data[ObsDeltaPhi(a,b)]
            module = checktype(delta_phi)
            return (delta_phi + module.pi) % (2*module.pi) - module.pi

    def bins(self, data):
        return np.linspace(-np.pi, np.pi, 60)

class ObsDeltaEta(Observable):
    def __init__(self, jet_index1, jet_index2):
        super().__init__(jet_index1, jet_index2)
        self.tex_template = "\Delta \eta_{%s %s}"
        self.unit = None

    def calculate(self, data):
        try:
            eta1 = ObsEta(self.indices[0]).from_data(data)
            eta2 = ObsEta(self.indices[1]).from_data(data)
            return eta1 - eta2
        except KeyError:
            graph = [obs.indices for obs in data if type(obs) == ObsDeltaEta]
            route = find_route(graph, *self.indices)
            if route is None:
                self.not_found()
            return sum(sign * data[ObsDeltaEta(a,b)] for a,b,sign in route)

    def bins(self, data):
        return np.linspace(-12, 12, 60)

class ObsDeltaR(Observable):
    def __init__(self, jet_index1, jet_index2):
        super().__init__(jet_index1, jet_index2)
        self.tex_template = "\Delta R_{%s %s}"
        self.unit = None

    def calculate(self, data):
        delta_eta = ObsDeltaEta(*self.indices).from_data(data)
        delta_phi = ObsDeltaPhi(*self.indices).from_data(data)
        module = checktype(delta_phi)
        return module.sqrt(delta_eta**2 + delta_phi**2)

    def bins(self, data):
        return np.linspace(0, np.sqrt(12**2 + np.pi**2), 60)

class ObsScalarPTSum(Observable):
    def __init__(self, *indices, binning_cutoff=0.99, n_bins=60):
        super().__init__(*indices)
        self.tex_template = "p_{T," + "%s"*len(indices) + "}"
        self.unit = "GeV"
        self.binning_cutoff = binning_cutoff
        self.n_bins = n_bins

    def bins(self, data):
        pt = self.from_data(data)
        cutoff = np.sort(pt)[:math.ceil(len(pt)*self.binning_cutoff)+1][-1]
        return np.linspace(0, cutoff, self.n_bins)

    def calculate(self, data):
        ret = 0.
        count = ObsCount().from_data(data)
        for i in self.indices:
            ret += np.where(count > i, ObsPT(i).from_data(data), 0.)
        return ret

def find_route(graph, from_node, to_node, route=[]):
    if from_node == to_node:
        return route
    for i,(a,b) in enumerate(graph):
        result = None
        if a == from_node:
            new_graph = graph[:]
            new_graph.pop(i)
            result = find_route(new_graph, b, to_node, [*route, (a,b,+1)])
        elif b == from_node:
            new_graph = graph[:]
            new_graph.pop(i)
            result = find_route(new_graph, a, to_node, [*route, (a,b,-1)])
        if result is not None:
            return result
    return None

def checktype(data):
    if type(data) == np.ndarray or type(data) == np.array:
        return np
    elif type(data) == torch.Tensor:
        return torch
    else:
        raise(ValueError("Data of type {} does not fit format".format(type(data))))
