import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import pickle

from observables import ObsPT, ObsCount
from util import FunctionRegistry, eval_observables_list

class Plots:
    register = FunctionRegistry()

    def __init__(self, params, doc):
        self.params = params
        self.doc = doc

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    def create_plots(self, data_store):
        for func_name in self.params["plots"]:
            print(f"  Running {func_name}", flush=True)
            Plots.register.functions[func_name](self, data_store)

    def observable_histogram(self, pp, obs_train, obs_test, obs_predict,
                             bins, tex_name, unit, log_scale=False, weight_samples=None,
                             weights=None, save_dict=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)


            y_t,  _ = np.histogram(obs_test, bins=bins)
            y_tr, _ = np.histogram(obs_train, bins=bins)

            if not (weight_samples is None):
                obs_predict = obs_predict.reshape(weight_samples,
                        len(obs_predict)//weight_samples)
                hists_g = np.array([np.histogram(obs_predict[i,:], bins=bins)[0]
                                    for i in range(weight_samples)])
                hists = [y_t, np.mean(hists_g, axis=0), y_tr]
                hist_errors = [np.sqrt(y_t), np.std(hists_g, axis=0), np.sqrt(y_tr)]
            elif not (weights is None):
                y_r, _ = np.histogram(obs_predict, bins=bins, weights=weights)
                y_g, _ = np.histogram(obs_predict, bins=bins)
                hists = [y_t, y_g, y_tr, y_r]
                hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr), np.sqrt(y_r)]
            else:
                y_g,  _ = np.histogram(obs_predict, bins=bins)
                hists = [y_t, y_g, y_tr]
                hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]

            integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
            scales = [1 / integral if integral != 0. else 1. for integral in integrals]

            if save_dict is not None:
                sd = save_dict
                sd["true"], sd["inn"], sd["train"] = hists[:3]
                sd["true_err"], sd["inn_err"], sd["train_err"] = hist_errors[:3]
                sd["bins"], sd["tex_name"], sd["unit"] = bins, tex_name, unit
                if weights is not None:
                    sd["reweighted"], sd["reweighted_err"] = hists[-1], hist_errors[-1]

            FONTSIZE = 14
            labels = ["True", "INN", "Train"]
            colors = ["#e41a1c", "#3b528b", "#1a8507"]
            if not (weights is None):
                labels.append("Reweighted")
                colors.append("#8b3b78")
            dup_last = lambda a: np.append(a, a[-1])
            fig1, axs = plt.subplots(3 + int(not (weights is None)), 1, sharex=True,
                    gridspec_kw={"height_ratios" : [4, 1, 1] + [1 for i in range(int(not (weights is None)))], "hspace" : 0.00})

            for y, y_err, scale, label, color in zip(hists, hist_errors, scales,
                                                     labels, colors):
                axs[0].step(bins, dup_last(y) * scale, label=label, color=color,
                        linewidth=1.0, where="post")
                axs[0].step(bins, dup_last(y + y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
                axs[0].step(bins, dup_last(y - y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
                axs[0].fill_between(bins, dup_last(y - y_err) * scale,
                        dup_last(y + y_err) * scale, facecolor=color,
                        alpha=0.3, step="post")

                if label == "True": continue

                ratio = (y * scale) / (hists[0] * scales[0])
                ratio_err = np.sqrt((y_err / y)**2 + (hist_errors[0] / hists[0])**2)
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
                axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
                axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
                axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                        dup_last(ratio + ratio_err), facecolor=color, alpha=0.3, step="post")

                delta = np.fabs(ratio - 1) * 100
                delta_err = ratio_err * 100

                markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:])/2, delta,
                        yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                        linewidth=0, fmt=".", capsize=2)
                [cap.set_alpha(0.5) for cap in caps]
                [bar.set_alpha(0.5) for bar in bars]

            axs[0].legend(loc="lower right", frameon=False)
            axs[0].set_ylabel("normalized", fontsize = FONTSIZE)
            if log_scale:
                axs[0].set_yscale("log")

            axs[1].set_ylabel(r"$\frac{\mathrm{True}}{\mathrm{Model}}$",
                    fontsize = FONTSIZE)
            axs[1].set_yticks([0.95,1,1.05])
            axs[1].set_ylim([0.9,1.1])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
            plt.xlabel(r"${%s}$ %s" % (tex_name, ("" if unit is None else f"[{unit}]")),
                    fontsize = FONTSIZE)

            axs[2].set_ylim((0.05,20))
            axs[2].set_yscale("log")
            axs[2].set_yticks([0.1, 1.0, 10.0])
            axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

            axs[2].axhline(y=1.0,linewidth=0.5, linestyle="--", color="grey")
            axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)

            if not (weights is None):
                weight = y_r/y_g
                disc = weight/(1+weight)
                axs[3].step(bins[:-1], disc, color = colors[-1], linewidth=1.0)
                axs[3].set_ylabel("D(x)",fontsize = FONTSIZE)
                axs[3].set_yticks([0.0,0.25,0.5,0.75,1.0])
                axs[3].axhline(y=0.5,linewidth=0.5, linestyle="--", color="black")


            plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
            plt.close()

    @register
    def histograms_1d(self, data_store):
        observables = eval_observables_list(self.params["observables_1d"])
        save_dict_all = {}

        with PdfPages(self.doc.add_file("observables.pdf")) as pp:
            for obs in observables:
                obs_train = obs.from_data(data_store["train"])
                obs_test = obs.from_data(data_store["test"])
                obs_predict = obs.from_data(data_store["predict"])
                bins = obs.bins(data_store["test"])
                tex_name = obs.tex_name(self.params)
                weight_samples = self.params.get("weight_samples")
                weights = data_store.get("events_weights")
                save_dict = {}
                save_dict_all[str(obs)] = save_dict
                self.observable_histogram(pp, obs_train, obs_test, obs_predict,
                                          bins, tex_name, obs.unit,
                                          log_scale=isinstance(obs, ObsPT),
                                          weight_samples=weight_samples,
                                          weights=weights, save_dict=save_dict)

        if self.params.get("save_hist_data", True):
            with open(self.doc.add_file("hist_data.pkl"), "wb") as f:
                pickle.dump(save_dict_all, f)
    @register
    def histograms_2d(self, data_store):
        titles              = ["Train", "True", "INN", "INN reweighted", "Average weight in bin", "Change in density"]
        keys                = ["train", "test", "predict", "reweighted", "weights", "change"]
        observable_pairs    = eval_observables_list(self.params["observable_pairs"])
        weights             = data_store.get("events_weights", None)

        with PdfPages(self.doc.add_file("2d_hists.pdf")) as pp:
            for obs1, obs2 in observable_pairs:
                bins1 = obs1.bins(data_store["test"])
                bins2 = obs2.bins(data_store["test"])

                plt.figure(figsize=(25,3))
                for i, (key, title) in enumerate(zip(keys, titles)):
                    if key == "reweighted" and not weights is None:
                        y1 = obs1.from_data(data_store["predict"])
                        y2 = obs2.from_data(data_store["predict"])

                        plt.subplot(1, 6, i+1)
                        hist, xedges, yedges = np.histogram2d(y1, y2, bins=[bins1, bins2], density=True, weights=weights)
                        plt.pcolormesh(xedges, yedges, hist.T, rasterized=True)
                        plt.xlabel(f"${obs1.tex_name(self.params)}$")
                        plt.ylabel(f"${obs2.tex_name(self.params)}$")
                        cb = plt.colorbar()
                        cb.set_label("density of events")
                    elif key == "weights" and not weights is None:
                        y1 = obs1.from_data(data_store["predict"])
                        y2 = obs2.from_data(data_store["predict"])

                        plt.subplot(1, 6, i+1)
                        hist_rw, xedges, yedges = np.histogram2d(y1, y2, bins=[bins1, bins2], density=True, weights=weights)
                        hist, xedges, yedges    = np.histogram2d(y1, y2, bins=[bins1, bins2], density=True)
                        hist_weights            = hist_rw/(hist+1.e-12)
                        hist_weights            = np.clip(hist_weights, 0, 2)
                        plt.pcolormesh(xedges, yedges, hist_weights.T, rasterized=True, vmin=0, vmax=2)
                        plt.xlabel(f"${obs1.tex_name(self.params)}$")
                        plt.ylabel(f"${obs2.tex_name(self.params)}$")
                        cb = plt.colorbar(ticks=np.linspace(0,2,6))
                        cb.set_label("weight (clamped to <2)")
                    elif key == "change" and not weights is None:
                        y1 = obs1.from_data(data_store["predict"])
                        y2 = obs2.from_data(data_store["predict"])

                        plt.subplot(1, 6, i+1)
                        hist_rw, xedges, yedges = np.histogram2d(y1, y2, bins=[bins1, bins2], density=True, weights=weights)
                        hist, xedges, yedges    = np.histogram2d(y1, y2, bins=[bins1, bins2], density=True)
                        hist_change             = hist_rw - hist
                        plt.pcolormesh(xedges, yedges, hist_change.T, rasterized=True)
                        plt.xlabel(f"${obs1.tex_name(self.params)}$")
                        plt.ylabel(f"${obs2.tex_name(self.params)}$")
                        cb = plt.colorbar(ticks=np.linspace(-0.1,0.1,51))
                        cb.set_label("change in density")
                    elif key in ["train", "test", "predict"]:
                        y1 = obs1.from_data(data_store[key])
                        y2 = obs2.from_data(data_store[key])
                        plt.subplot(1, 6, i+1)
                        hist, xedges, yedges = np.histogram2d(y1, y2, bins=[bins1, bins2],
                                                            density=True)
                        plt.pcolormesh(xedges, yedges, hist.T, rasterized=True)
                        plt.xlabel(f"${obs1.tex_name(self.params)}$")
                        plt.ylabel(f"${obs2.tex_name(self.params)}$")
                        cb = plt.colorbar()
                        cb.set_label("density of events")

                plt.tight_layout()
                plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                plt.close()

    def min_max_count(self, *data):
        return (min(np.min(d) for d in data),
                max(np.max(d) for d in data))

    @register
    def plot_jet_counts(self, data_store):
        njet_obs = ObsCount()
        njet_train = njet_obs.from_data(data_store["train"])
        njet_test = njet_obs.from_data(data_store["test"])
        njet_predict = njet_obs.from_data(data_store["predict"])
        min_n, max_n = self.min_max_count(njet_train, njet_test, njet_predict)
        bins = np.arange(min_n-0.5, max_n+1)
        tex_name = njet_obs.tex_name(self.params)

        with PdfPages(self.doc.add_file("jet_count.pdf")) as pp:
            self.observable_histogram(pp, njet_train, njet_test, njet_predict, bins, tex_name,
                                      njet_obs.unit, log_scale = True)

    @register
    def histograms_1d_bycount(self, data_store):
        observables = eval_observables_list(self.params["observables_1d"])
        incl_observables = eval_observables_list(self.params.get("inclusive_observables_1d", []))

        njet_obs = ObsCount()
        njet_train = njet_obs.from_data(data_store["train"])
        njet_test = njet_obs.from_data(data_store["test"])
        njet_predict = njet_obs.from_data(data_store["predict"])
        min_n, max_n = self.min_max_count(njet_train, njet_test, njet_predict)
        if min_n < 1: min_n = 1
        save_dict_all = {}
        weights = data_store.get("events_weights")
        if weights is not None:
            weights_adj = weights.copy()

        for i, jet_count in enumerate(range(min_n, max_n+1)):
            train_mask = njet_train == jet_count
            test_mask = njet_test == jet_count
            predict_mask = njet_predict == jet_count
            save_dict_count = {}
            save_dict_all[jet_count] = save_dict_count
            with PdfPages(self.doc.add_file(f"{jet_count}jet_observables.pdf")) as pp:
                for obs in observables:
                    if any(i >= jet_count for i in obs.flat_indices()): continue
                    obs_train = obs.from_data(data_store["train"])[train_mask]
                    obs_test = obs.from_data(data_store["test"])[test_mask]
                    obs_predict = obs.from_data(data_store["predict"])[predict_mask]
                    bins = obs.bins(data_store["test"])
                    tex_name = obs.tex_name(self.params)
                    if weights is not None:
                        nweights = weights[predict_mask]
                        save_dict_all[f"all_weights{i}"] = nweights
                        weights_adj[predict_mask] = nweights / np.mean(nweights)
                    else:
                        nweights = None
                    save_dict = {}
                    save_dict_count[str(obs)] = save_dict
                    self.observable_histogram(pp, obs_train, obs_test, obs_predict,
                                              bins, tex_name, obs.unit,
                                              weights=nweights,
                                              log_scale=isinstance(obs, ObsPT),
                                              save_dict=save_dict)

        if len(incl_observables) > 0:
            save_dict_count = {}
            save_dict_all["incl"] = save_dict_count
            with PdfPages(self.doc.add_file(f"incl_jet_observables.pdf")) as pp:
                for obs in incl_observables:
                    obs_train = obs.from_data(data_store["train"])
                    obs_test = obs.from_data(data_store["test"])
                    obs_predict = obs.from_data(data_store["predict"])
                    bins = obs.bins(data_store["test"])
                    tex_name = obs.tex_name(self.params)
                    save_dict = {}
                    save_dict_count[str(obs)] = save_dict
                    self.observable_histogram(pp, obs_train, obs_test, obs_predict,
                                              bins, tex_name, obs.unit,
                                              weights=weights_adj,
                                              log_scale=isinstance(obs, ObsPT),
                                              save_dict=save_dict)

        if self.params.get("save_hist_data", True):
            with open(self.doc.add_file("hist_data.pkl"), "wb") as f:
                pickle.dump(save_dict_all, f)

    @register
    def histograms_2d_bycount(self, data_store):
        titles = ["Train", "True", "INN"]
        keys = ["train", "test", "predict"]
        observable_pairs = eval_observables_list(self.params["observable_pairs"])

        njet_obs = ObsCount()
        njet_train = njet_obs.from_data(data_store["train"])
        njet_test = njet_obs.from_data(data_store["test"])
        njet_predict = njet_obs.from_data(data_store["predict"])
        min_n, max_n = self.min_max_count(njet_train, njet_test, njet_predict)
        if min_n < 2: min_n = 2

        for jet_count in range(min_n, max_n+1):
            masks = [njet_train == jet_count,
                     njet_test == jet_count,
                     njet_predict == jet_count]
            with PdfPages(self.doc.add_file(f"{jet_count}jet_2d_hists.pdf")) as pp:
                for obs1, obs2 in observable_pairs:
                    if any(i >= jet_count for i in obs1.flat_indices() + obs2.flat_indices()):
                        continue
                    bins1 = obs1.bins(data_store["test"])
                    bins2 = obs2.bins(data_store["test"])

                    plt.figure(figsize=(10,3))
                    for i, (key, title, mask) in enumerate(zip(keys, titles, masks)):
                        y1 = obs1.from_data(data_store[key])[mask]
                        y2 = obs2.from_data(data_store[key])[mask]
                        plt.subplot(1, 3, i+1)
                        hist, xedges, yedges = np.histogram2d(y1, y2, bins=[bins1, bins2],
                                                              density=True)
                        plt.pcolormesh(xedges, yedges, hist.T, rasterized=True)
                        plt.xlabel(f"${obs1.tex_name(self.params)}$")
                        plt.ylabel(f"${obs2.tex_name(self.params)}$")
                        cb = plt.colorbar()
                        cb.set_label("number of events")

                    plt.tight_layout()
                    plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                    plt.close()

    @register
    def latent_plots(self, data_store):
        data = data_store["latent_test"]
        latent_dim = data.shape[1]
        figsize = latent_dim*2.5
        plt.figure(figsize=(figsize,figsize))

        bins_1d = np.linspace(-4,4,20)
        bins_2d = np.linspace(-3,3,30)
        x = np.linspace(-4,4,100)
        gauss = np.exp(-0.5*x*x) / np.sqrt(2*np.pi)
        circles = np.arange(0.5, 2.4, 0.5)

        for i in range(latent_dim):
            for j in range(latent_dim):
                if i < j: continue
                plt.subplot(latent_dim,latent_dim,i*latent_dim+j+1)
                if i == j:
                    plt.hist(data[:,i], bins=bins_1d, color="blue", density=True)
                    plt.plot(x, gauss, color="red")
                else:
                    plt.hist2d(data[:,i], data[:,j], bins=[bins_2d, bins_2d], rasterized=True)
                    for r in circles:
                        plt.gca().add_patch(plt.Circle((0, 0), r, color="red", fill=False))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.doc.add_file("latent.pdf"), bbox_inches="tight", format="pdf",
                pad_inches=0.05)
        plt.close()

    @register
    def loss_plots(self, data_store):
        loss_keys = ["inn_losses", "kl_losses", "inn_loss_gan", \
        "inn_loss_gauss", "disc_losses", "total_losses"]
        labels = ["INN loss", "KL loss", "INN GAN loss", "INN gauss loss", \
        "Discriminator loss", "total loss"]

        if all(key not in data_store for key in loss_keys):
            print("Skipping loss plots")
            return

        with PdfPages(self.doc.add_file("losses.pdf")) as pp:
            plt.figure(figsize = [15.,8.])
            for key, label in zip(loss_keys, labels):
                if key not in data_store:
                    continue

                loss_data = data_store[key]
                if len(loss_data) > 0 and isinstance(loss_data[0], list):
                    all_losses = [np.array(l) for l in loss_data]
                    all_labels = [f"{label} Model {k}" for k in range(len(loss_data))]
                else:
                    all_losses = [np.array(loss_data)]
                    all_labels = [label]

                for loss, lbl in zip(all_losses, all_labels):
                    #if key == "inn_losses" or key == "inn_loss_gauss":
                    #    loss -= np.clip(np.min(loss - 1.e-7, axis=0), -1.e30, None)
                    plt.plot(loss, label=lbl)

            plt.legend(fontsize=16)
            plt.xlabel("Epochs",fontsize=16)
            plt.ylabel("Loss (clipped to 0)", fontsize=16)
            plt.title("Logarithmic Loss Curve", fontsize=16)
            plt.yscale("log")
            plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
            plt.close()

            plt.figure(figsize = [15.,8.])
            for key, label in zip(loss_keys, labels):
                if key not in data_store:
                    continue

                loss_data = data_store[key]
                if len(loss_data) > 0 and isinstance(loss_data[0], list):
                    all_losses = [np.array(l) for l in loss_data]
                    all_labels = [f"{label} Model {k}" for k in range(len(loss_data))]
                else:
                    all_losses = [np.array(loss_data)]
                    all_labels = [label]

                for loss, lbl in zip(all_losses, all_labels):
                    plt.plot(loss, label=lbl)

            plt.legend(fontsize=16)
            plt.xlabel("Epochs",fontsize=16)
            plt.ylabel("Loss", fontsize=16)
            plt.title("Non-logarithmic Loss", fontsize=16)
            plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
            plt.close()

    @register
    def weight_plots(self, data_store):
        weights_true = data_store["weights_true"]
        weights_fake = data_store["weights_fake"]

        with PdfPages(self.doc.add_file("weights.pdf")) as pp:
            means_true  = np.mean(weights_true, axis = 1)
            means_fake  = np.mean(weights_fake, axis = 1)
            std_true    = np.std(weights_true, axis = 1)
            std_fake    = np.std(weights_true, axis = 1)
            x_true = np.arange(len(means_true))
            x_fake = np.arange(len(means_fake))

            plt.figure(figsize = [15.,8.])

            plt.plot(x_true, means_true, label="Mean Weight Truth", color="blue", linewidth=1.0)
            plt.plot(x_true, means_true+std_true, color="blue", linewidth=0.5)
            plt.plot(x_true, means_true-std_true, color="blue", linewidth=0.5)
            plt.fill_between(x_true, means_true+std_true, means_true-std_true, alpha=0.3, color="blue")
            plt.plot(x_fake, means_fake, label="Mean Weight Fake", color="red", linewidth=1.0)
            plt.plot(x_fake, means_fake+std_fake, color="red", linewidth=0.5)
            plt.plot(x_fake, means_fake-std_fake, color="red", linewidth=0.5)
            plt.fill_between(x_fake, means_fake+std_fake, means_fake-std_fake, alpha=0.3, color="red")

            plt.legend(fontsize=16)
            plt.yscale("log")
            plt.xlabel(r"$\frac{\text{Epoch}}{10}$",fontsize=16)
            plt.ylabel(r"$\overline{w_i}$", fontsize=16)
            plt.title("Mean Discriminator Weight", fontsize=16)
            plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
            plt.close()


            for i in range(len(weights_true)):
                epoch_weights_true = weights_true[i]
                epoch_weights_fake = weights_fake[i]

                plt.figure(figsize = [15.,8.])

                bins = np.logspace(np.log10(1.e-2), np.log10(1.e2), 60)

                y_t, x_t = np.histogram(weights_true, bins=bins, range=[0, 10], density=True)
                y_f, x_f = np.histogram(weights_fake, bins=bins, range=[0, 10], density=True)

                plt.step(x_t[:-1], y_t, label="Weight Distribution Truth", color="blue", linewidth=1.0, where='mid')
                plt.step(x_f[:-1], y_f, label="Weight Distribution Fake", color="red", linewidth=1.0, where='mid')
                plt.legend(fontsize=16)
                plt.xlabel('Discriminator weights',fontsize=16)
                plt.ylabel("normalized", fontsize=16)
                plt.title("Weight distribution in Epoch {}".format(i*10), fontsize=16)
                plt.xscale("log")
                plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
                plt.close()


    @register
    def plot_roc(self, data_store, modelname=None):
        if modelname is None:
            modelname = "Adversarial"
        n_points = 100
        events_logit_true = data_store["events_logit_true"]
        events_logit_fake = data_store["events_logit_fake"]
        cut_off = np.linspace(0, 1, n_points)
        false_pos = np.mean(np.expand_dims(events_logit_true, axis=-1) < np.expand_dims(cut_off, axis=0), axis=0)
        true_pos = np.mean(np.expand_dims(events_logit_fake, axis=-1) < np.expand_dims(cut_off, axis=0), axis=0)
        np.savez(
            self.doc.add_file(f"disc_data_{modelname.replace(' ', '_')}.npz"),
            events_true = data_store["events_logit_true"],
            events_fake = data_store["events_logit_fake"]
        )

        with PdfPages(self.doc.add_file(f"ROC_{modelname.replace(' ', '_')}.pdf")) as pp:
            plt.figure(figsize=(4.5, 4))
            plt.plot(false_pos, true_pos, label="Discriminator")
            #plt.legend(fontsize=16)
            plt.xlabel('False Positives',fontsize=16)
            plt.ylabel("True Positives",fontsize=16)
            #plt.title(f'RoC curve of model {modelname}')
            plt.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
            plt.close()

            plt.figure(figsize=(4.5,4))
            y_t, x_t = np.histogram(events_logit_true, bins=60, range=[0, 1], density=True)
            y_f, x_f = np.histogram(events_logit_fake, bins=60, range=[0, 1], density=True)

            plt.step(x_t[:60], y_t, label="Truth", color="blue", linewidth=1.0, where='mid')
            plt.step(x_f[:60], y_f, label="Fake", color="red", linewidth=1.0, where='mid')
            plt.legend(fontsize=16)
            plt.xlabel('Discriminator output',fontsize=16)
            plt.ylabel("Density",fontsize=16)
            #plt.title(f'Label Distribution of model {modelname}')
            plt.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
            plt.close()


            plt.figure(figsize=(4.5,4))
            weights_true = (1 - events_logit_true)/events_logit_true
            weights_fake = events_logit_fake/(1 - events_logit_fake)

            bins = np.logspace(np.log10(1.e-1), np.log10(1.e1), 50)
            widths = (bins[1:] - bins[:-1])

            y_t, x_t = np.histogram(weights_true, bins=bins)
            y_f, x_f = np.histogram(weights_fake, bins=bins)
            plt.xscale("log")
            # y_t, y_f = y_t/widths, y_f/widths
            y_t = y_t/len(weights_true)
            y_f = y_f/len(weights_fake)


            #plt.step(x_t[:-1], y_t, label="Truth", color="blue", linewidth=1.0, where='mid')
            plt.step(x_f[:-1], y_f, label="Fake", color="red", linewidth=1.0, where='mid')
            #plt.legend(fontsize=16)
            plt.xlabel('Discriminator weights',fontsize=16)
            plt.ylabel("Density",fontsize=16)
            #plt.title(f'Weight Distribution of model {modelname}')
            plt.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
            plt.close()


    @register
    def plot_roc_bycount(self, data_store):
        njet_obs = ObsCount()
        njet_train = njet_obs.from_data(data_store["train"])
        njet_test = njet_obs.from_data(data_store["test"])
        njet_predict = njet_obs.from_data(data_store["predict"])
        min_n, max_n = self.min_max_count(njet_train, njet_test, njet_predict)

        for jet_count in range(min_n, max_n+1):
            masks = [njet_train == jet_count,
                     njet_test == jet_count,
                     njet_predict == jet_count]
            logits_masked = {
                "events_logit_fake": data_store["events_logit_fake"][masks[2]],
                "events_logit_true": data_store["events_logit_true"][masks[1]]
                }

            self.plot_roc(logits_masked, modelname=f"Discriminator {jet_count} jets")
