import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg):
    batch_size = D1_arg.size(0)
    D1 = torch.sigmoid(D1_logits)
    D2 = torch.sigmoid(D2_logits)

    grad_D1_logits = autograd.grad(outputs=D1_logits, inputs=D1_arg,
                        grad_outputs=torch.ones(D1_logits.size(), device=D1.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_D2_logits = autograd.grad(outputs=D2_logits, inputs=D2_arg,
                        grad_outputs=torch.ones(D2_logits.size(), device=D2.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_D1_logits_norm = torch.norm(grad_D1_logits.view(batch_size,-1), dim=1, keepdim=True)
    grad_D2_logits_norm = torch.norm(grad_D2_logits.view(batch_size,-1), dim=1, keepdim=True)


    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    assert grad_D1_logits_norm.size() == D1.size()
    assert grad_D2_logits_norm.size() == D2.size()

    reg_D1 = (1.0-D1)**2 * (grad_D1_logits_norm)**2
    reg_D2 = (D2)**2 * (grad_D2_logits_norm)**2

    disc_regularizer = torch.mean(reg_D1 + reg_D2)

    return disc_regularizer
