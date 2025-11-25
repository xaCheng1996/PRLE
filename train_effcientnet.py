import os
import time
import numpy as np
import logging
import argparse
import torch
import torch.nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.ImageFolder import ImageFolder
import wandb
import cv2
# cv2.setNumThreads(0)
import yaml
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms
import torch
import os
import cv2
from tqdm import tqdm
from utils.heatmap_vis import visual
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from utils.misc import AverageMeter, accuracy, set_seed, save_checkpoint, get_all_preds_labels
from dataset.transform import xception_default_data_transforms as xception_transforms
from data_load_helper import data_load
from validation_helper import validation, test
from save_and_load_helper import save, load
from train_helper import train_effi, train_remove
from efficientnet_pytorch import EfficientNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/yaml/efficient.yaml')
    parser.add_argument("--progress", type=float)

    args = parser.parse_args()

    return args


def main(config=None):

    args = get_args()
    if config is None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config
    logger.info(config)
    run = wandb.init(project=config['general']['exp_name'], name=config['general']['exp_id'])
    if config['general']['seed'] != -1:
        set_seed(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['general']['gpu_id'])

    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    torch.backends.cudnn.benchmark = True

    train_dataset_list, val_dataset_list, test_dataset_list, train_loader_list, val_loader_list, test_loader_list = data_load(
        logger, config, xception_transforms)

    train_dataset = train_dataset_list['train_dataset']
    train_loader = train_loader_list['train_loader']
    val_loader = val_loader_list['val_loader']

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2,
                                         weights_path='./efficientnet-b4-6ed6700e.pth')

    model.to(device)

    if config['general']['mode'] == "train":
        # model = torch.nn.DataParallel(model).cuda()
        logger.info(f"train_dataset.size: {len(train_dataset)}")

        criterion = torch.nn.CrossEntropyLoss(reduction=config['data']['reduction'])
        # criterion = torch.nn.CrossEntropyLoss()
        criterion_cos = torch.nn.CosineEmbeddingLoss()
        best_val_auc = 0  # best validation auc of roc curve

        for param in model.parameters():
            param.requires_grad = True
        optimizer_net = torch.optim.Adam(model.parameters(), lr=2e-4)
        auc_list = validation(logger, config, model, val_dataset_list, val_loader_list, device, epoch=0)
        for i in range(config['network']['epochs_net']):
            train_loss = train_effi(config, model, optimizer_net, criterion, criterion_cos, train_loader, device=device)
            logger.info(f"Network Epoch：{i + 1}/{config['network']['epochs_net']}. train loss: {train_loss}.")

            auc_list = validation(logger, config, model, val_dataset_list, val_loader_list, device, epoch=i)

            val_auc = auc_list['ffpp_auc']
            val_auc, best_val_auc = save(config, model, optimizer_net, val_auc, best_val_auc, i)

    else:
        model = load(logger, model, config)
        model.to(device)
    val_auc_list = test(logger, config, model, test_dataset_list, test_loader_list, device)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
