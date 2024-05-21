# caloinn
INN for Calorimeter Shower Generation

Mainly developed by Thorsten Buss,
repo forked from ThorstenBuss/caloinn.

test

## Usage

Running a training:
```
python src/main.py --use_cuda params/example_bayesian.yaml
```
This creates a subfolder in the `results` folder named `yyyymmdd_hhmmss_run_name` where the
prefix is the date and time and `run_name` is specified in the param card.

## Parameters

This is a list of the parameters that can be used in yaml parameter files. Many have default
values, such that not all parameters have to be specified.

### Run parameters


Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
run\_name               | Name for the output folder

### Data and global preprocessing parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
data\_path              | Name of the hdf5 file containing the data set
vae\_dir                | Path to a possibly pretrained VAE/KVAE. Default (none) trains a new VAE/KVAE
dtype                   | float16, float32 or float64; Higher precision makes training and generating slower.
particle\_type          | Used particle (photon, pion or electron)
dataset                 | Index of the given CaloChallenge dataset (1,2 or 3)
vae\_frac               | Fraction of the dataset that is used for validation
eps                     | epsilon, used to ensure numerical stability for the layer normalization
threshold               | Energy threshold applied outside of the VAE and its training (not seen in the loss function).

### VAE/KVAE preprocessing parameters
Parameter                 | Explanation
------------------------- | -------------------------------------------------------------------
alpha                     | Constant value to add on the data before taking the logarithm 
VAE\_internal\_threshold  | Energy threshold, seen by the VAE loss function. Alternative to ``threshold''
VAE\_einc\_preprocessing  | (hot\_one or logit) This describes the way how the extra dimensions of the VAE are preprocessed. (KVAE supports only logit)

### VAE/KVAE training parameters

Parameter                 | Explanation
------------------------- | -------------------------------------------------------------------
VAE\_type                 | Type of VAE model used for compression. CVAE and KVAE are supported.
VAE\_learnable\_norm      | Enables a learnable affine transformation before the VAE. If false, the data is transformed to unit variance and zero mean.
VAE\_lr                   | Base learning rate used for the VAE training
VAE\_lr\_scheduler        | Type of LR scheduling: "none", "reduce\_on\_plateau", "step" or "one\_cycle"  
VAE\_max\_lr              | Only one-cycle scheduler: Maximum learning rate. Defaults to $10 \cdot$ VAE\_lr
VAE\_lr\_decay\_epochs    | Only step scheduler: decay interval in epochs
VAE\_lr\_decay\_factor    | Only step scheduler: decay factor
VAE\_weight\_decay             | L2 weight decay
VAE\_batch\_size          | Batch size
VAE\_n\_epochs            | Number of training epochs
VAE\_save\_interval       | Interval in epochs for saving (plotting and (temporary) model checkpoint)
VAE\_keep\_models         | Interval in epochs for saving (model checkpoint with unique name)
VAE\_zero\_logit          | Sets the logit loss to zero
sparsity\_loss            | The loss weight for a smooth sparsity approximation
VAE\_smearing\_self       | Used for the smearing matrix. Defines the fraction of energy that each voxel keeps.
VAE\_smearing\_share      | Used for the smearing matrix. Defines the fraction of energy that is passed to each of the 8 surrounding voxels.
VAE\_hidden\_sizes        | A list with the number of neurons for the individual VAE layers (for the KVAE it specifies the fully connected part).
VAE\_latent\_dim          | Latent dimension
VAE\_beta                 | Loss weight for the KL loss
VAE\_gamma                | Loss weight for the BCE loss
VAE\_hidden\_sizes\_kernel| Only KVAE: A list with the number of neurons for the individual subnetworks
VAE\_kernel\_size         | Only KVAE: kernel size
VAE\_kernel\_stride       | Only KVAE: stride
VAE\_kernel\_latent       | Only KVAE: laten space for the underlying block modules



### INN training parameters

Parameter               | Explanation
----------------------- | --------------------------------------------------------------------
latent\_type            | The part of the latent space that the INN is trained on. (pre\_sampling, post\_sampling or only\_means)
logit\_transformation   | If true use apply another logit transformation after the VAE
log\_cond               | If true use the logarithm of the patron energy as condition
lr                      | Learning rate
batch\_size             | Batch size
lr\_scheduler           | Type of LR scheduling: "reduce\_on\_plateau", "step" or "one\_cycle"
max\_lr                 | Only one-cycle scheduler: Maximum learning rate. Defaults to $10 \cdot$ lr
lr\_decay\_epochs       | Only step scheduler: decay interval in epochs
lr\_decay\_factor       | Only step scheduler: decay factor
weight\_decay           | L2 weight decay
betas                   | List of the two Adam beta parameters
n\_epochs               | Number of training epochs
save\_interval          | Interval in epochs for saving (plotting and (temporary) model 
keep\_models            | Interval in epochs for saving (model checkpoint with unique name)
bayesian                | True to enable Bayesian training
std\_init               | Only Bayesian: ln of the initial standard deviation of the weight distributions
prior\_prec             | Only Bayesian: Inverse of the prior standard deviation for the Bayesian layers
n\_blocks               | Number of coupling blocks
internal\_size          | Internal size of the coupling block subnetworks
layers\_per\_block      | Number of layers in each coupling block subnetwork
coupling\_type          | Type of coupling block: "affine", "cubic", "rational\_quadratic" or "MADE"
bounds\_init            | Only spline blocks: bounds of the splines
num\_bins               | Only spline blocks: number of bins
permute\_soft           | If True, uses random rotation matrices instead of permutations
dropout                 | Dropout fraction for the subnetworks
