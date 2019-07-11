from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from loss import *
from skimage import io


def train_tcn(inputs, labels, model, optimizer, args, iteration, device, logger=None,
              validate=False, with_src=False):

    if validate:
        model.eval()
    else:
        model.train()

    inputs = inputs.to(device)
    labels = labels.to(device)

    y = model(inputs)

    if args.loss_type == "dice":
        fn_loss = dice_loss
    else:
        if with_src:
            fn_loss = BCE_loss_with_src
        else:
            fn_loss = BCE_loss

    loss_ = fn_loss(y, labels, with_weight=True)

    if with_src:
        loss_f = loss_[0].data.cpu().numpy()
        loss_s = loss_[1].data.cpu().numpy()
        loss = loss_[0] + loss_[1]
    else:
        loss = loss_

    loss_val = loss.data.cpu().numpy()

    if not validate:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iteration: {iteration:4d} Loss : {loss_val:.4f}")
    else:
        print(f"Iteration: {iteration:4d} \t\t Loss : {loss_val:.4f}")

    if with_src:
        print(f"\t loss_f: {loss_f:.4f}  loss_s:{loss_s:.4f}")

    if logger is not None:
        if validate:
            logger.add_scalar("val_loss/total", loss, iteration)
        else:
            logger.add_scalar("train_loss/total", loss, iteration)
            if with_src:
                logger.add_scalar("train_loss/total_frg", loss_[0], iteration)
                logger.add_scalar("train_loss/total_src", loss_[1], iteration)


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

    fn_loss = Two_loss

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


def train_GAN(inputs, labels, model, optimizer, args, iteration, device, logger=None,
              validate=False):

    generator, discriminator = model

    valid = torch.tensor(1., dtype=torch.float32).to(device)
    fake = torch.tensor(0., dtype=torch.float32).to(device)

    if validate:
        [m.eval() for m in model]
    else:
        [m.train() for m in model]

    inputs = inputs.to(device)
    labels = labels.to(device)

    # prediction
    y = generator(inputs)

    optimizer_G, optimizer_D = optimizer

    if args.loss_type == "dice":
        fn_loss = dice_loss
    else:
        fn_loss = BCE_loss
    fn_dis_loss = nn.BCEWithLogitsLoss()

    # Discriminator
    optimizer_D.zero_grad()

    pred_real = discriminator(labels, inputs)
    loss_real = fn_dis_loss(pred_real, valid.expand_as(pred_real))

    pred_fake = discriminator(torch.sigmoid(y.detach()), inputs)
    loss_fake = fn_dis_loss(pred_fake, fake.expand_as(pred_real))

    loss_D = 0.5 * (loss_real + loss_fake)

    if not validate:
        loss_D.backward()
        optimizer_D.step()

    # Generator
    optimizer_G.zero_grad()

    pred_fake = discriminator(torch.sigmoid(y), inputs)
    loss_gen = fn_loss(y, labels, with_weight=False)
    loss_dis = fn_dis_loss(pred_fake, valid.expand_as(pred_fake))

    loss_G = 0.5 * (loss_gen + args.gamma * loss_dis)

    if not validate:
        loss_G.backward()
        optimizer_G.step()

    if not validate:
        print(f"Iteration: {iteration:4d} Loss_G : {loss_G.data.cpu().numpy():.4f}" +
              f" Loss_D : {loss_D.data.cpu().numpy():.4f}")
    else:
        print(f"Iteration: {iteration:4d} \t\t Loss_G : {loss_G.data.cpu().numpy():.4f}" +
              f" Loss_D : {loss_D.data.cpu().numpy():.4f}")

    if logger is not None:
        if validate:
            logger.add_scalar("val_loss/loss_G", loss_G, iteration)
            logger.add_scalar("val_loss/loss_D", loss_D, iteration)
        else:
            logger.add_scalar("train_loss/loss_G", loss_G, iteration)
            logger.add_scalar("train_loss/loss_D", loss_D, iteration)


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


def train_template_match(Xs, Xt, Ys, Yt, model, optimizer, args, iteration, device,
                         logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    # if iteration > 500:
    #     module.set_bn_to_eval()

    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    preds, predt = model(Xs, Xt)

    optimizer.zero_grad()

    loss1 = CE_loss_src_target(preds, Ys, mode="source")
    loss2 = CE_loss_src_target(predt, Yt, mode="forge")

    loss = loss1 + loss2
    loss.backward()

    # loss.backward()

    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(f"{iteration:5d}: loss     {loss_val:.4f}")

    # loss_val_t = loss_top.data.cpu().numpy()
    # print(f"{'':5s}: \t\t\tloss top {loss_val_t:.4f}")

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)


def train_template_match_im(Xs, Xt, Ys, Yt, model, optimizer, args, iteration, device,
                         logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    # if iteration > 500:
    #     module.set_bn_to_eval()

    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    preds, predt = model(Xs, Xt)

    optimizer.zero_grad()

    loss1 = CE_loss_src_target(preds, Ys, mode="source")
    loss2 = CE_loss_src_target(predt, Yt, mode="forge")

    loss = loss1 + loss2
    loss.backward()

    # loss.backward()

    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(f"{iteration:5d}: loss     {loss_val:.4f}")

    # loss_val_t = loss_top.data.cpu().numpy()
    # print(f"{'':5s}: \t\t\tloss top {loss_val_t:.4f}")

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)
