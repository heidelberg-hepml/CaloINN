import torch
import matplotlib.pyplot as plt
import yaml
import os
import sys
from copy import deepcopy
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from model import CINN
from trainer import ECAETrainer
import data_util
from documenter import Documenter
from plotter import Plotter
from plotting import plot_hist
from matplotlib import cm
from myDataLoader import MyDataLoader
from calc_obs import *
from trainer import DNNTrainer
from model import CINN, DNN
import plotting

def load_trainer(directory, use_cuda=True):
    use_cuda = torch.cuda.is_available() and use_cuda
    device = 'cuda:0' if use_cuda else 'cpu' 


    with open(directory + "/params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    doc = Documenter(params['run_name'], existing_run=True, basedir=directory,
                    log_name="log_classifier.txt", read_only=False)
    
    particle = params.get("particle_type", "piplus")

    data_path_train = os.path.join("..","..","..", "Datasets", particle , "train_" + particle + ".hdf5")
    params["data_path_train"] = data_path_train

    data_path_test = os.path.join("..","..","..", "Datasets", particle , "test_" + particle + ".hdf5")
    params["data_path_test"] = data_path_test

    classification_set = os.path.join("..","..","..", "Datasets", particle , "cls_" + particle + ".hdf5")
    params["classification_set"] = classification_set
    
    trainer = ECAETrainer(params, device, doc, vae_dir=params.get("vae_dir", None))
    
    return trainer, params, device, doc

def main():
    directory = "/remote/gpu06/ernst/Master_Thesis/add_VAE/CaloINN/results/2023_02_16_Test_ECAE/piplus_pre_sampling"
    trainer, params, device, doc = load_trainer(directory)
    trainer.load("_best")

    basedir = directory + "/eval_best_model"
    os.makedirs(basedir)
    doc.basedir = basedir

    classifier_trainer = DNNTrainer(trainer, params, device, doc)
    
    print("\n\nstarting classifier test")
    sys.stdout.flush()
    for _ in range(params["classifier_runs"]):
        classifier_trainer.train()
        classifier_trainer.do_classifier_test(do_calibration=True)
        classifier_trainer.reset_model()

    # Return the results
    classifier_trainer.clean_up()
    
if __name__ == "__main__":
    main()