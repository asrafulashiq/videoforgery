from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(x, t, gamma=2):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,1, ...].
        y: (tensor) sized [N, 1, ...].
    Return:
        (tensor) focal loss.
    '''

    x = x.view(-1)
    t = t.view(-1)

    wgt = torch.sum(t) / (t.shape[0])

    p = torch.sigmoid(x)
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = (1-wgt)*t + wgt*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach())

def dice_loss(y, labels):
    smooth = 1
    y = torch.sigmoid(y.view(-1))
    lab = labels.view(-1)

    numer = 2 * (y * lab).sum()
    den = y.sum() + lab.sum()

    return 1 - (numer + smooth) / (den + smooth)

def BCE_loss(y, labels):
    eps = 1e-8
    y = y.squeeze().double()
    labels = labels.squeeze().double()

    wgt = torch.sum(labels) / (labels.shape[0] * labels.shape[1] * labels.shape[2])

    bce_loss = -(1 - wgt) * labels * F.logsigmoid(y) - wgt * (1 - labels) * torch.log(
        1 - torch.sigmoid(y) + eps
    )
    bce_loss = torch.mean(bce_loss)

    if torch.isnan(bce_loss) or bce_loss < 0:
        import pdb

        pdb.set_trace()

    return bce_loss.float()


def Index_loss(ysim, ydis, ind_gt, device):
    ysim = torch.pow(torch.sigmoid(ysim.squeeze()), 2)
    ydis = torch.pow(torch.sigmoid(ydis.squeeze()), 2)
    ind_gt = ind_gt.squeeze()

    y_gt = torch.abs(ind_gt[:, 0] - ind_gt[:, 1])
    # l1_loss = torch.mean(torch.max(y_gt - y, torch.FloatTensor([0]).to(device)))
    _loss = torch.mean(torch.max(0.2 + ysim - ydis, torch.FloatTensor([0]).to(device)))

    if torch.isnan(_loss):
        import pdb
        pdb.set_trace()

    return _loss


def train_match_in_the_video(
    dataset, model, optimizer, args, iteration, device, logger=None
):
    model.train()
    X_im, Ind = dataset.load_triplet(num=args.batch_size)
    X_im = X_im.to(device)
    Ind = Ind.to(device)

    y_sim = model(X_im[:, [0, 1]])
    y_dis = model(X_im[:, [0, 2]])

    loss = Index_loss(y_sim, y_dis, Ind, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iteration: {iteration}  Loss : {loss.data.cpu().numpy():.4f}")

    if logger is not None:
        logger.add_scalar("loss/loss_ind", loss, iteration)


def train(inputs, labels, model, optimizer, args, iteration, device, logger=None,
         validate=False):

    if validate:
        model.eval()
    else:
        model.train()

    inputs = inputs.to(device)
    labels = labels.to(device)

    # prediction
    y = model(inputs)

    if args.loss_type == "dice":
        fn_loss = dice_loss
    else:
        fn_loss = BCE_loss

    # loss = focal_loss(y, labels)
    # loss = BCE_loss(y, labels)
    loss = fn_loss(y, labels)
    loss_val = loss.data.cpu().numpy()

    if not validate:
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        print(f"Iteration: {iteration:4d} Loss : {loss_val:.4f}")
    else:
        print(f"Iteration: {iteration:4d} \t\t Loss : {loss_val:.4f}")


    if logger is not None:
        if validate:
            logger.add_scalar("val_loss/total", loss, iteration)
        else:
            logger.add_scalar("train_loss/total", loss, iteration)



def train_with_boundary(
    inputs, labels, model, optimizer, args, iteration, device, logger=None,
    validate=False
):

    if validate:
        model.eval()
    else:
        model.train()

    inputs = inputs.to(device)
    labels = labels.to(device)

    mask, boundary = labels[:, 0], labels[:, 1]
    # prediction
    y_m, y_b = model(inputs)

    loss_mask = BCE_loss(y_m, mask)
    loss_boundary = BCE_loss(y_b, boundary)

    loss = loss_mask + args.gamma_b * loss_boundary

    loss_val = loss.data.cpu().numpy()

    if not validate:
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


        print(
            f"Iteration: {iteration:4d}, loss_m : {loss_mask.data.cpu().numpy():.4f} "
            + f"loss_b : {loss_boundary.data.cpu().numpy():.4f} "
            + f"Loss : {loss_val:.4f}"
        )
    else:
        print(
            f"VALIDATE:::: Iteration: {iteration:4d}, loss_m : {loss_mask.data.cpu().numpy():.4f} "
            + f"loss_b : {loss_boundary.data.cpu().numpy():.4f} "
            + f"Loss : {loss_val:.4f}"
        )

    if logger is not None:
        if validate:
            pref = "val_loss"
        else:
            pref = "loss"

        logger.add_scalar(f"{pref}/total", loss, iteration)
        logger.add_scalar(f"{pref}/mask", loss_mask, iteration)
        logger.add_scalar(f"{pref}/boundary", loss_boundary, iteration)
