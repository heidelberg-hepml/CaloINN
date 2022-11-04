import os
import torch
import numpy as np
from classifer_data import get_dataloader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
# torch.set_default_dtype(torch.float64)

class DNN(torch.nn.Module):
    """ NN for vanilla classifier """
    def __init__(self, input_dim, num_layer=2, num_hidden=512, dropout_probability=0.,
                 is_classifier=True):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        if is_classifier:
            all_layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class CNN(torch.nn.Module):
    """ CNN for improved classification """
    def __init__(self, is_classifier=True):
        super(CNN, self).__init__()
        self.cnn_layers_0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=(3, 7), padding=(2, 0)),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=(4, 7), padding=(2, 0)),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 14)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        linear_layers = [
            torch.nn.Linear(256+256+256+4, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1)
        ]
        if is_classifier:
            linear_layers.append(torch.nn.Sigmoid())
        self.linear_layers = torch.nn.Sequential(*linear_layers)

    def forward(self, x):
        split_elements = [*torch.split(x, [288, 144, 72, 4], dim=-1)]
        layer_0 = split_elements[0].view(x.size(0), 3, 96)
        layer_1 = split_elements[1].view(x.size(0), 12, 12)
        layer_2 = split_elements[2].view(x.size(0), 12, 6)
        layer_0 = self.cnn_layers_0(layer_0.unsqueeze(1)).view(x.size(0), -1)
        layer_1 = self.cnn_layers_1(layer_1.unsqueeze(1)).view(x.size(0), -1)
        layer_2 = self.cnn_layers_2(layer_2.unsqueeze(1)).view(x.size(0), -1)
        all_together = torch.cat((layer_0, layer_1, layer_2, split_elements[3]), 1)
        ret = self.linear_layers(all_together)
        return ret

def preprocess(data, arguments):
    """ takes dataloader and returns tensor of
        layer0, layer1, layer2, log10 energy
    """

    # Called for batches in order to prevent ram overflow
    ALPHA = 1e-6

    def logit(x):
        return torch.log(x / (1.0 - x))

    def logit_trafo(x):
        local_x = ALPHA + (1. - 2.*ALPHA) * x
        return logit(local_x)

    device = arguments["device"]
    threshold = arguments["threshold"]
    normalize = arguments["normalize"]
    use_logit = arguments["use_logit"]

    layer0 = data['layer_0']
    layer1 = data['layer_1']
    layer2 = data['layer_2']
    energy = torch.log10(data['energy']*10.).to(device)
    E0 = data['layer_0_E']
    E1 = data['layer_1_E']
    E2 = data['layer_2_E']

    if threshold:
        layer0 = torch.where(layer0 < 1e-7, torch.zeros_like(layer0), layer0)
        layer1 = torch.where(layer1 < 1e-7, torch.zeros_like(layer1), layer1)
        layer2 = torch.where(layer2 < 1e-7, torch.zeros_like(layer2), layer2)

    if normalize:
        layer0 /= (E0.reshape(-1, 1, 1) +1e-16)
        layer1 /= (E1.reshape(-1, 1, 1) +1e-16)
        layer2 /= (E2.reshape(-1, 1, 1) +1e-16)

    E0 = (torch.log10(E0.unsqueeze(-1)+1e-8) + 2.).to(device)
    E1 = (torch.log10(E1.unsqueeze(-1)+1e-8) + 2.).to(device)
    E2 = (torch.log10(E2.unsqueeze(-1)+1e-8) + 2.).to(device)

    # ground truth for the training
    target = data['label'].to(device)

    layer0 = layer0.view(layer0.shape[0], -1).to(device)
    layer1 = layer1.view(layer1.shape[0], -1).to(device)
    layer2 = layer2.view(layer2.shape[0], -1).to(device)

    if use_logit:
        layer0 = logit_trafo(layer0)/10.
        layer1 = logit_trafo(layer1)/10.
        layer2 = logit_trafo(layer2)/10.

    return torch.cat((layer0, layer1, layer2, energy, E0, E1, E2), 1), target

def load_classifier(model, save_dir, mode, device, run_number=None):
    """ loads a saved model """

    if run_number is not None:
        filename = f"{run_number}_{mode}.pt"
    else:
        filename = f"{mode}.pt"

    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        raise ValueError(f"Can not open the model {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def save_model(model, save_dir, mode, run_number=None):
    """ saves a model """

    if run_number is not None:
        filename = f"{run_number}_{mode}.pt"
    else:
        filename = f"{mode}.pt"

    path = os.path.join(save_dir, filename)

    torch.save({'model_state_dict':model.state_dict()},
                path)

    return model

def train_epoch(model, dataloader, optimizer, epoch, n_epochs, sigmoid_in_BCE, arguments):
    """ train one epoch """
    model.train()

    for batch, data_batch in enumerate(dataloader):
        input_vector, target_vector = preprocess(data_batch, arguments)
        output_vector = model(input_vector)
        if sigmoid_in_BCE:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.BCELoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % (len(dataloader)//2) == 0:
            print(f'Epoch {epoch+1} / {n_epochs}, step {batch} / {len(dataloader)}; loss {loss.item():.4f}')


        #TODO: What happens here? Accuracy estimate?
        if sigmoid_in_BCE:
            pred = torch.round(torch.sigmoid(output_vector.detach()))
        else:
            pred = torch.round(output_vector.detach())
        target = torch.round(target_vector.detach())
        if batch == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), res_pred.cpu()))

def evaluate(model, dataloader, arguments, sigmoid_in_BCE, return_ROC=False, final=False):
    model.eval()
    for batch, data_batch in enumerate(dataloader):
        input_vector, target_vector = preprocess(data_batch, arguments)
        output_vector = model(input_vector)

        pred = output_vector.reshape(-1)
        target = target_vector.double()

        # TODO BCE Loss written by hand, why?
        # Probably to compute JSD, too
        if batch == 0:
            result_true = target
            result_pred = pred

            # log_A = torch.log(pred)[target == 1]
            # log_B = torch.log(1.-pred)[target == 0]

        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)

            # log_A = torch.cat((log_A, torch.log(pred)[target == 1]), 0)
            # log_B = torch.cat((log_B, torch.log(1.-pred)[target == 0]), 0)

    if sigmoid_in_BCE:
        BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
        result_pred = torch.sigmoid(result_pred).cpu().numpy()
    else:
        BCE = torch.nn.BCELoss()(result_pred, result_true)
        result_pred = result_pred.cpu().numpy()


    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    # BCE = torch.mean(log_A) + torch.mean(log_B)
    # JSD = 0.5* BCE + np.log(2.)
    JSD = - BCE + np.log(2.)
    print(f"BCE loss of test set is {BCE}, JSD of the two dists is {JSD/np.log(2.)}")

    if final:
        return eval_acc, JSD, result_true, result_pred

    if not return_ROC:
        return eval_acc, JSD
    else:
        return roc_curve(result_true, result_pred)

# TODO: not fully understood
def calibrate_classifier(model, calibration_data, sigmoid_in_BCE, arguments):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for batch, data_batch in enumerate(calibration_data):
        input_vector, target_vector = preprocess(data_batch, arguments)
        output_vector = model(input_vector)
        if sigmoid_in_BCE:
            pred = torch.sigmoid(output_vector).reshape(-1)
        else:
            pred = output_vector.reshape(-1)
        target = target_vector.to(torch.float64)
        if batch == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg

# TODO: input_dim, probably 508 (?)
# input_dim = {'0': 288, '1': 144, '2': 72, 'all': 504}[args.which_layer]
# input_dim += 4
def classifier_test(input_dim, device, data_path_train, data_path_val, data_path_test, save_dir, 
                    threshold, normalize, use_logit, sigmoid_in_BCE=True,
                    lr=2e-4, n_epochs=30, batch_size=1000, load=False,
                    num_layer=2, num_hidden=512, dropout_probability=0.,
                    run_number=None, modes=["DNN", "CNN"]):

    """
    Starts the classifier test

    parameters:

    (int)    input_dim:             dimension of the input data
    (str)    data_path_train:       full path to the training hdf5 file
    (str)    data_path_val:         full path to the validation hdf5 file
    (str)    data_path_test:        full path to the test hdf5 file
    (str)    save_dir:              directory in which the data of this test run can be written
    (bool)   threshold:             if threshold of 1e-2MeV is applied
    (bool)   normalize:             If voxels should be normalized per layer
    (bool)   use_logit:             if data is logit transformed
    (bool)   sigmoid_in_BCE=True:   remove sigmoid from NN and uses BCEWithLogitsLoss instead (numerically more stable)
    (float)  lr=2e-4:               learning rate
    (int)    n_epochs=30:           number of epochs
    (int)    batch_size=1000:       batch size
    (bool)   load=False:            Should a matching existing model in save_dir be loaded
    (int)    num_layer=2:           number of hidden layers for the DNN
    (int)    num_layer=2:           neurons per hidden layer in den DNN
    (float)  dropout_probability=0: dropout probability in the DNN
    (int)    run_number=None:       a number used if one needs the average over several results. Affects the naming of the saved models
    (list)   modes=["DNN", "CNN"]:  A array containing the classifier names that should be trained. Currently supported: CNN, DNN
                                    Warning: CNN much slower!  
    """

    arguments = {"device" : device, 
    "threshold": threshold, 
    "normalize" : normalize, 
    "use_logit" : use_logit}

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Make sure to return to the old default value afterwards!
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    # Create the DNN and the CNN classifiers
    new_modes = []
    models = []
    
    if "DNN" in modes:
        dnn = DNN(input_dim, num_layer, num_hidden, dropout_probability, is_classifier=(not sigmoid_in_BCE))
        dnn.to(device)
        new_modes.append("DNN")
        models.append(dnn)
    
    if "CNN" in modes:
        cnn = CNN(not sigmoid_in_BCE)
        cnn.to(device)
        new_modes.append("CNN")
        models.append(cnn)

    for mode, model in zip(new_modes, models):
        dataloader_train, dataloader_val, dataloader_test = get_dataloader(
                                                            data_path_train,
                                                            data_path_test,
                                                            data_path_val=data_path_val,
                                                            apply_logit=False,
                                                            device=device,
                                                            batch_size=batch_size,
                                                            with_noise=False,
                                                            normed=False,
                                                            normed_layer=False,
                                                            return_label=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # load the model if required otherwise create it
        if load:
            model = load_classifier(model, save_dir, mode, device, run_number)
        else:
            # Train the model and save the best one
            best_eval_acc = float('-inf')
            for epoch in range(n_epochs):
                # Train for one epoch
                train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                            epoch=epoch, n_epochs=n_epochs, sigmoid_in_BCE=sigmoid_in_BCE,
                            arguments=arguments)

                # Check, if the model did improve on the validation data
                with torch.no_grad():
                    eval_acc, _ = evaluate(model=model, dataloader=dataloader_val,
                                            sigmoid_in_BCE=sigmoid_in_BCE, arguments=arguments)
                # Save the model if it is better than the best so far.
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_epoch = epoch + 1
                    save_model(model, save_dir, mode, run_number)
                if eval_acc == 1:
                    break

        # Load the classifier version that resulted in the best val loss
        model = load_classifier(model, save_dir, mode, device, run_number)

        # Now run on the test data
        with torch.no_grad():
            eval_acc, JSD, result_true, result_pred = evaluate(model=model, dataloader=dataloader_test,
                                                sigmoid_in_BCE=sigmoid_in_BCE, arguments=arguments,
                                                final=True)
        
            calibration_data = torch.clone(dataloader_val)

            prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
            print("unrescaled calibration curve:", prob_true, prob_pred)
            
            # Isotonic calibration
            calibrator = calibrate_classifier(model, calibration_data, sigmoid_in_BCE, arguments)
            rescaled_pred = calibrator.predict(result_pred)
            eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
            print("Rescaled accuracy is", eval_acc)
            eval_auc = roc_auc_score(result_true, rescaled_pred)
            print("rescaled AUC of dataset is", eval_auc)
            prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
            print("rescaled calibration curve:", prob_true, prob_pred)

            # Compute the losses
            BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
            JSD = - BCE.cpu().numpy() + np.log(2.)
            print(f"rescaled BCE loss of test set is {BCE}, rescaled JSD of the two dists is {JSD/np.log(2.)}" )

            #write to file
            results = np.array([[eval_acc, eval_auc, JSD/np.log(2.), best_epoch]])
            filename = 'summary_'+('loaded_' if load else '')+mode+'.npy'
            if run_number==0:
                    np.save(os.path.join(save_dir, filename), results)
            else:
                prev_res = np.load(os.path.join(save_dir, filename),
                                    allow_pickle=True)
                new_res = np.concatenate([prev_res, results])
                np.save(os.path.join(save_dir, filename), new_res)
                
                
    torch.set_default_dtype(old_default_dtype)
    





