import torch
from torch import nn as nn
from torch.autograd import Function

class DiceLoss(nn.Module):
    """
    Computes dice loss
    """
    def __init__(self, normalization='sigmoid'):
        super(DiceLoss, self).__init__()
        assert normalization in ['sigmoid', 'softmax']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target, epsilon=1e-6):
        # get probabilities from logits
        input = self.normalization(input)

        # input and target shapes must match
        assert input.size() == target.size()

        input = torch.flatten(input)
        target = torch.flatten(target)
        target = target.float()

        intersect = (input * target).sum(-1)

        # Standard dice
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        dice_score =  2 * (intersect / denominator.clamp(min=epsilon))

        return 1. - torch.mean(dice_score)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):

        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, device):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# Helper class for training SSL
class Contrast(torch.nn.Module):
    def __init__(self, device, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)

# Helper class for training SSL
class ProxyLoss(torch.nn.Module):
    def __init__(self, batch_size, device, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().to(device)
        self.recon_loss = torch.nn.L1Loss().to(device)
        self.contrast_loss = Contrast(device, batch_size).to(device)
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        #rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        rot_loss = 0
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
