import os
from utils.misc import AverageMeter, set_seed, save_checkpoint, get_all_preds_labels
import torch
def save(config, model, optimizer, val_auc, best_val_auc, epoch):
    state_dict = model.state_dict()
    is_best = val_auc > best_val_auc
    best_val_auc = max(val_auc, best_val_auc)
    if not os.path.exists(config['data']['save_checkpoint']):
        os.makedirs(config['data']['save_checkpoint'])
    save_checkpoint({
        'epoch': epoch,
        'state_dict': state_dict,
        'auc': val_auc,
        'best_val_auc': best_val_auc,
        'optimizer': optimizer.state_dict(),
    }, is_best, config['data']['save_checkpoint'])
    return val_auc, best_val_auc

def load(logger, model, config):
    logger.info(f"Loading best model from {os.path.join(config['data']['save_checkpoint'], 'model_best.pth.tar')}")
    state_dict = torch.load(os.path.join(config['data']['save_checkpoint'], "model_best.pth.tar"))["state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    flag = True
    for k, v in state_dict.items():
        if 'module' in k:
            flag = False
            break
    if not flag:
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(
            torch.load(os.path.join(config['data']['save_checkpoint'], "model_best.pth.tar"))["state_dict"])
    #
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove module.
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    return model