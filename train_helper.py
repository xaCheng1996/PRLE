import time
import torch
from tqdm import tqdm
from utils.misc import AverageMeter, set_seed, save_checkpoint, get_all_preds_labels
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


def train(model, optimizer, criterion, train_loader, device=torch.device('cpu'), scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_mask = AverageMeter()
    end = time.time()
    batch_size = 0
    model.train()

    for (inputs, targets, _, _, _, weight) in tqdm(train_loader):
        inputs, targets, weight = inputs.to(device), targets.to(device), weight.to(device)

        data_time.update(time.time() - end)
        # with autocast('cuda'):
        logits = model(inputs)
        # features = model.extract_features(inputs)
        # print(logits)
        # print(targets, weight)
        loss = criterion(logits, targets)
        final_loss = loss

        # loss__weighted = loss * weight
        # final_loss = loss__weighted.mean()

        losses.update(final_loss.item(), inputs.size(0))
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # scaler.scale(loss).backward(retain_graph=True)
        # scaler.step(optimizer)
        # scaler.update()

        final_loss.backward(retain_graph=True)
        optimizer.step()
        losses.update(final_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # if batch_idx % 100 == 0:
        #     logger.info(f"batch idx: {batch_idx}, train loss: {loss}")
    print("batch time: %.4f, data_time: %.4f" % (batch_time.avg, data_time.avg))
    print("video throughout during training: %.4f videos/s" % (train_loader.batch_size / batch_time.avg))
    return losses.avg, losses_mask.avg


def train_effi(config, model, optimizer, criterion, criterion_cos, train_loader, device=torch.device('cpu')):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_mask = AverageMeter()
    losses_contras = AverageMeter()
    end = time.time()
    batch_size = 0
    model.train()

    for (inputs, targets, _, _, inputs_contras, weight) in tqdm(train_loader):
        inputs, targets, weight = inputs.to(device), targets.to(device), weight.to(device)

        data_time.update(time.time() - end)

        # with autocast('cuda'):
        logits = model(inputs)
        # features = model.extract_features(inputs)
        # print(logits)
        # print(targets, weight)
        loss = criterion(logits, targets)
        final_loss = loss
        # loss__weighted = loss * weight
        # final_loss = loss__weighted.mean()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # scaler.scale(loss).backward(retain_graph=True)
        # scaler.step(optimizer)
        # scaler.update()
        final_loss.backward(retain_graph=True)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(final_loss.item(), inputs.size(0))
        losses.update(final_loss.item())
        # if batch_idx % 100 == 0:
        #     logger.info(f"batch idx: {batch_idx}, train loss: {loss}")
    print("batch time: %.4f, data_time: %.4f" % (batch_time.avg, data_time.avg))
    print("video throughout during training: %.4f videos/s" % (train_loader.batch_size / batch_time.avg))
    return losses.avg, losses_mask.avg


def train_remove(model, optimizer, criterion, train_loader, device=torch.device('cpu')):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_contras = AverageMeter()
    end = time.time()
    batch_size = 0
    model.train()
    criterion_cos = torch.nn.CosineEmbeddingLoss()

    for (inputs, targets, _, _, inputs_contras, weight) in tqdm(train_loader):
        inputs, targets, inputs_contras, weight = inputs.to(device), targets.to(device), inputs_contras.to(device), weight.to(device)
        data_time.update(time.time() - end)
        # with autocast('cuda'):
        # with autocast('cuda'):
        logits, _ = model(inputs)
        # features = model.extract_features(inputs)
        # print(logits)
        # print(targets, weight)
        loss = criterion(logits, targets)
        final_loss = loss
        # loss__weighted = loss * weight
        # final_loss = loss__weighted.mean()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # scaler.scale(loss).backward(retain_graph=True)
        # scaler.step(optimizer)
        # scaler.update()
        final_loss.backward(retain_graph=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.detach().cpu().numpy(), inputs.size(0))
        # if batch_idx % 100 == 0:
        #     logger.info(f"batch idx: {batch_idx}, train loss: {loss}")
    print("batch time: %.4f, data_time: %.4f" % (batch_time.avg, data_time.avg))
    print("video throughout during training: %.4f videos/s" % (train_loader.batch_size / batch_time.avg))
    return losses.avg, losses_contras.avg
