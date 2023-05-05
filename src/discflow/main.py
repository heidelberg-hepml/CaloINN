import yaml
import sys
import os
import shutil
import torch
import argparse
import numpy as np
import pandas as pd

from documenter import Documenter
import inn, multi_cond_inn, multi_separate_inn, multi_chain_inn, multi_hierarchical_inn
import binn, igan, multi_cond_igan
from preprocessing import Preprocessing
from plots import Plots
from util import eval_observables_list, eval_observables_expr, model_class
from load_data import load_h5, split_data, get_eppp_observables, select_by_jet_count
import observables as ob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile")
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--start_from", type=str)
    parser.add_argument("--plot_model", action="store_true")
    args = parser.parse_args()

    if args.warm_start or args.plot_model:
        doc = Documenter(args.paramfile[16:], existing_run=args.paramfile)
        with open(doc.get_file("params.yaml", False)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(args.paramfile) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        doc = Documenter(params["run_name"])
        shutil.copy(sys.argv[1], doc.add_file("params.yaml", False))
    data_store = {}

    use_cuda = torch.cuda.is_available()
    print("Using device " + ("GPU" if use_cuda else "CPU"))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    data_store["device"] = device

    print("Loading data", flush=True)
    data = load_h5(params["data_path"])
    if "select_by_jet_count" in params:
        data = select_by_jet_count(data, *params["select_by_jet_count"])
    data_train, data_test = split_data(data, params["test_split"])
    data_store["train"] = get_eppp_observables(data_train)
    data_store["test"] = get_eppp_observables(data_test)
    data_store["n_jets"] = data.shape[1] // 4
    print("Preprocessing data")

    preproc = Preprocessing(params, data_store)
    input_obs = eval_observables_list(params["input_observables"])
    train_input = { obs: obs.from_data(data_store["train"]) for obs in input_obs }
    test_input = { obs: obs.from_data(data_store["test"]) for obs in input_obs }
    data_store["train_preproc"] = preproc.apply(train_input, forward=True, init_trafo=True)
    data_store["test_preproc"] = preproc.apply(test_input, forward=True, init_trafo=False)
    data_store["dim_x"] = data_store["train_preproc"].shape[1]

    print("Building model", flush=True)
    model_type = params.get("model_type", "INN")
    if model_type not in model_class.functions:
        raise ValueError(f"Unknown model type '{model_type}'")

    model = model_class.functions[model_type](params, data_store, doc)
    if model_type in ["iGAN", "MultiCondIGAN"]:
        model.disc_preproc = Preprocessing(params, data_store)
        input_copy = train_input.copy()
        model.add_provided_values(input_copy, next(iter(input_copy.values())).shape[0])
        data_store["train_disc_preproc"] = model.disc_preproc.apply(
            { obs: obs.from_data(input_copy) for obs in model.obs_converter},
            forward=True, init_trafo=True, disc_steps=True)

    model.define_model_architecture()
    model.initialize_data_loaders()
    model.set_optimizer()
    model.preproc = preproc

    if args.plot_model or args.warm_start:
        model.load()

    if args.start_from:
        model.load(name=args.start_from)


    print("Running training", flush=True)
    if not args.plot_model:
        model.train()
        model.save()

    print("Generating data", flush=True)
    n_samples = params.get("generate_count", data_store["test_preproc"].shape[0])
    predict_pp = model.predict(n_samples)
    data_store["predict_preproc"] = predict_pp
    data_store["predict"] = preproc.apply(predict_pp, forward=False,
                                        init_trafo=False, as_numpy=True)
    if "latent_plots" in params["plots"]:
        data_store["latent_test"] = model.compute_latent()
    for obs_expr, value in params.get("provided_values", []):
        obs = eval_observables_list(obs_expr)
        data_store["predict"][obs] = np.full(n_samples, value, dtype=np.float32)

    counts = ob.ObsCount().from_data(data_store["predict"])
    output_obs = []
    for i in range(5):
        oe = ob.ObsE(i).from_data(data_store["predict"])
        opx = ob.ObsPx(i).from_data(data_store["predict"])
        opy = ob.ObsPy(i).from_data(data_store["predict"])
        opz = ob.ObsPz(i).from_data(data_store["predict"])
        mask = counts <= i
        oe[mask] = 0.
        opx[mask] = 0.
        opy[mask] = 0.
        opz[mask] = 0.
        output_obs.extend([oe, opx, opy, opz])
    pd.DataFrame(np.stack(output_obs, axis=1)).to_hdf(
        doc.add_file("generated.h5", False),
        "events",
        complib="blosc",
        complevel=5
    )

    if "generate_filter" in params:
        for key in ["test", "train", "predict"]:
            mask = eval_observables_expr(params["generate_filter"], data_store[key])
            for obs, data in data_store[key].items():
                data_store[key][obs] = data[mask]
            drop_count = np.sum(~mask)
            drop_percent = drop_count / len(mask) * 100
            print(f"Removed {drop_count} events from data set '{key}' ({drop_percent:.2f}%)")

    if model_type in ["iGAN", "MultiCondIGAN"]:
        print("Discriminate generated data", flush=True)
        data_store["events_logit_fake"] = model.predict_discriminator(
            data_store["predict_preproc"],
            sig = True
        ).detach().cpu().numpy()
        data_store["events_logit_true"] = model.predict_discriminator(
            data_store["test_preproc"],
            sig = True
        ).detach().cpu().numpy()
        data_store["events_weights"] = data_store["events_logit_fake"] / \
        np.clip((1 - data_store["events_logit_fake"]), 1.e-5, np.inf)
    data_store["predict_preproc"].cpu().detach().numpy()
    data_store["train_preproc"].cpu().detach().numpy()
    data_store["test_preproc"].cpu().detach().numpy()
    print("Creating plots", flush=True)
    plots = Plots(params, doc)
    plots.create_plots(data_store)

    if params.get("save_data", False):
        print("Saving the data tensors")
        os.makedirs(doc.get_file("data", False), exist_ok=True)
        torch.save({"test"      : data_store["test"],
                    "predict"   : data_store["predict"]},doc.get_file("data/data", False))
    print("The end", flush=True)

if __name__ == "__main__":
    main()
