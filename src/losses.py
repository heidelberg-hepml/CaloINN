import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
#from gradient_penalty import Discriminator_Regularizer


class LatentLoss:
    def __init__(self, params):
        '''Initializes the latent loss for the generative model.
            args:
                self                : [object] Loss
                type                : [string] specifies the loss function to use
                model               : [object] iGAN model
         '''
        self.params                 = params
        self.outprint               = True
        self.loss_type              = eval("self."+self.params.get("latent_loss_type", "weighted_latent"))
        if self.loss_type == self.weighted_latent:
            self.weight_sched = 1

    def apply(self, z, jac, sig=None):
        '''Applies the specified loss function loss_type to the training batch x_samps.
            args:
                x_samps             : [tensor] tensor holding the observables for each event
        '''
        loss        = self.loss_type(z, jac, sig)
        return loss

    def latent(self, z, jac, sig):
        '''Computes the usual maximum likelihood latent loss.'''
        return torch.mean(z**2)/2 - torch.mean(jac) / z.shape[1]


    def weighted_latent(self, z, jac, sig):
        '''Computes the maximum likelihood latent loss with reweighted events'''
        if sig is None:
            return self.latent(z, jac, sig)
        if self.outprint:
            print("Starting to reweight the latent loss objective.")
            self.outprint = False

        weights     = sig/torch.clamp((1-sig), min = 1.e-7)
        weight_pot  = self.params.get("weight_pot", 1)

        if self.params.get("sig_pot", False):
            weight_pot = torch.sqrt((self.params.get("sig_scale", 6) * (sig-0.5)))**2 * weight_pot

        weights     = weights ** (self.weight_sched * weight_pot)

        weights     = weights/torch.mean(weights)

        nan_mask    = torch.isnan(weights)
        num_nan     = nan_mask[nan_mask == True].shape[0]
        if num_nan >= 1:
            print("Setting {} nan weights ({}%) to 1.".format(num_nan, num_nan/weights.shape[0]*100))
            weights[nan_mask] = 1
        return torch.mean(weights[:,None] * z**2)/2 - torch.mean(weights * jac) / z.shape[1]


    def weight_pot_scheduler(self, epoch):
        start_adv_training_epoch = self.params.get("start_adv_training_epoch", 0)
        weight_fade_in_epochs    = max([self.params.get("weight_fade_in_epochs", 0), 1])
        end_fade                 = start_adv_training_epoch + weight_fade_in_epochs

        if epoch < start_adv_training_epoch:
            self.weight_sched = 0
        elif epoch >= start_adv_training_epoch and epoch < end_fade:
            self.weight_sched = 1/weight_fade_in_epochs * (epoch - start_adv_training_epoch)
        else:
            self.weight_sched = 1
        
class GanLoss:
    def __init__(self, params, data_store, adversarial):
        '''Initializes the Loss function used for the GAN.
            args:
                self                : [object] Loss
                params              : [dict] run parameters
                data_store          : [dict] data store
                adversarial         : [bool]   use the loss for traing the generator (adversarial)
        '''
        # initialize the loss function
        self.BCE                        = nn.BCEWithLogitsLoss()
        self.adversarial                = adversarial
        self.data_store                 = data_store
        self.device                     = data_store["device"]
        self.params                     = params
        if not adversarial:
            #                             v----- teifiteifiteifi
            self.loss_type              = eval("self."+self.params.get("disc_loss_type", "loss"))
            if self.params.get("disc_loss_type", "loss") == "loss_ns":
                print("Warning: using non-saturating loss for the discriminative model.")
        else:
            self.loss_type              = eval("self."+self.params.get("adv_loss_type", "loss"))

    def apply(self, pos, neg, return_acc=False, requires_grad=True, epoch=0,
              x_samps=None, x_gen=None):
        '''Applies the loss function self.Loss_type onto the discriminator input.
            args:
                self                : [object] Loss
                pos                 : [tensor] discriminator output for true samples
                neg                 : [tensor] discriminator output for fake samples
            kwargs:
                return_acc          : [bool]   return the accuracys of the discriminative model
                requires_grad       : [bool]   attach gradient to the noise, required for gradient penalty
                epoch               : [int] current training epoch
                x_samps, x_gen      : [tensor] needed for gradient penalty
        '''

        if ("weight_plots" in self.params.get("plots",[]) and
                epoch%self.params.get("weight_interval",5) == 0 and
                not self.adversarial):
            weights_true = pos/torch.clamp((1-pos), min=1.e-7)
            weights_fake = neg/torch.clamp((1-neg), min=1.e-7)
            self.data_store["epoch_weights_true"].append(weights_true.detach().cpu().numpy())
            self.data_store["epoch_weights_fake"].append(weights_fake.detach().cpu().numpy())

        # regularization
        #regularization = 0

        #if self.params.get("gradient_penalty", 0.0) > 0 and requires_grad:
        #    regularization = self.params.get("gradient_penalty", 0.0) * \
        #    Discriminator_Regularizer(pos, x_samps, neg, x_gen)

        # compute loss
        loss       = self.loss_type(pos, neg) + regularization

        if return_acc:
            return loss, torch.mean((pos > 0).float()), torch.mean((neg < 0).float())
        else:
            return loss

    def loss(self, pos, neg):
        '''Computes the usual BCE Loss for true and fake data.'''
        return self.BCE(neg, torch.zeros_like(neg)) + \
        self.BCE(pos, torch.ones_like(pos))

    def loss_ns(self, pos, neg):
        '''Computes the non-saturating BCE Loss for true and fake data (-> should only be used if adversarial==True).'''
        return self.BCE(neg, torch.ones_like(neg))


    def loss_rel(self, dist_true, dist_gen):
        '''Computes the relativistic loss log(sig(D(x_true-x_fake))).'''
        shape_true      = dist_true.shape[0]
        shape_gen       = dist_gen.shape[0]
        if shape_true < shape_gen:
            dist_gen    = dist_gen[:shape_true]
        else:
            dist_true   = dist_true[:shape_gen]

        diff            = dist_true - dist_gen
        pseudo_target   = torch.ones_like(diff, device= self.device).squeeze()
        if not self.adversarial:
            return self.LossF(diff, pseudo_target)
        else:
            return self.LossF(-diff, pseudo_target)

    def loss_w(self, dist_true, dist_gen):
        '''Computes the wasserstein loss (-> needs regularization for lipschitz steadiness).'''
        if not self.adversarial:
            return torch.mean(dist_true) - torch.mean(dist_gen)
        else:
            return - torch.mean(dist_gen)

    def regularizer(self, dist_true, x_true, dist_gen, x_gen):
        '''Computes the gradient penalty term (-> https://github.com/rothk/Stabilizing_GANs)'''
        # gradient penalty
        batch_size = x_true.size(0)
        D1 = torch.sigmoid(dist_true)
        D2 = torch.sigmoid(dist_gen)

        grad_dist_true = autograd.grad(outputs=dist_true, inputs=x_true,
                            grad_outputs=torch.ones(dist_true.size(), device=self.device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dist_gen = autograd.grad(outputs=dist_gen, inputs=x_gen,
                            grad_outputs=torch.ones(dist_gen.size(), device=self.device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_dist_true_norm = torch.norm(grad_dist_true.view(batch_size,-1), dim=1, keepdim=False)
        grad_dist_gen_norm = torch.norm(grad_dist_gen.view(batch_size,-1), dim=1, keepdim=False)


        #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
        assert grad_dist_true_norm.size() == D1.size(), f"Shape {grad_dist_true_norm.size()} does not match {D1.size()}"
        assert grad_dist_gen_norm.size() == D2.size(), f"Shape {grad_dist_gen_norm.size()} does not match {D2.size()}"

        reg_D1 = (1.0-D1)**2 * (grad_dist_true_norm)**2
        reg_D2 = (D2)**2 * (grad_dist_gen_norm)**2

        disc_regularizer = torch.mean(reg_D1 + reg_D2)

        return disc_regularizer
