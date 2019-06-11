from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from loss import *



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


def train_with_src(inputs, labels, model, optimizer, args, iteration, device, logger=None,
          validate=False):

    if validate:
        model.eval()
    else:
        model.train()

    inputs = inputs.to(device)
    labels = labels.to(device)

    # prediction
    y = model(inputs)

    fn_loss =  Two_loss

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
