import os
import shutil
import random
import time
from torchvision import transforms
import numpy as np
import logging
from tqdm import tqdm
import torch
from PIL import Image
import torch.nn.functional as F
from utils.heatmap_vis import mask_test, localization_mask
from utils.GradCAM import GradCAMCUS
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.cuda.amp import autocast, GradScaler


# def get_activation(name, activation):
#     def hook(model, input, output):
#         activation[name] = output
#     return hook
#
#
# def save_grad(name, grad_d):
#     def hook(grad):
#         grad_d[name] = grad
#     return hook


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        logging.info(f"save best model into {os.path.join(checkpoint, 'model_best.pth.tar')}\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def visual(model, img_paths_list):
    train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    target_layers = [model.model.conv4, model.model.conv3]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    for image in tqdm(img_paths_list):
        image = image.strip()
        img = Image.open(image).convert('RGB')
        input_tensor = train_transforms(
            img).cuda()  # the input tensor after data preprocessing fedded into model
        img_ori = cv2.imread(image)  # original image
        img_ori = cv2.resize(img_ori, (299, 299))  # resize original image to be overlayyed by heatmap
        # targets_0 = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))
        grayscale_cam = grayscale_cam[0, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

        result = show_cam_on_image(img_ori / 255, grayscale_cam, use_rgb=True)
        result = Image.fromarray(result)
        save_path = image.replace('FFIW', 'Visual_Wrong_w_FFIW')
        save_path = save_path.replace('.png', '_w.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(save_path)
        result.save(save_path)

def get_all_preds_labels(model, loader, device):
    # switch to evaluate mode
    model.eval()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([], dtype=torch.int32)
    batch_size = 0
    # all_labels_mask = torch.tensor([], dt ype=torch.int32)
    time_pre = time.time()
    wrong_list = []

    # wrong_list = open('/root/autodl-tmp/PRLE/wrong_xcep_wo_wrong.txt', 'r').readlines()
    # visual(model, wrong_list)
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            # GFLOPS of face model
            # flops_face = FlopCountAnalysis(model, images)
            # print("FLOPs face: ", flops_face.total() / 1e9)
            preds = F.softmax(outputs, dim=1)
            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            all_labels = torch.cat((all_labels, labels.cpu().int()), dim=0)

        #     pred_result = np.argmax(preds.cpu().detach(), axis=-1)
        #     for i in range(len(labels)):
        #         if labels[i] != pred_result[i]:
        #             wrong_list.append(path[i])
        # wrong_list.sort()
        # with open('./wrong_xcep_wo_PRLE.txt', 'w') as file:
        #     for i in wrong_list:
        #         file.writelines(i)
        #         file.writelines('\n')

    time_post = time.time()
    print("Video Throughout in this evaluation: %.4f Video/s" % (loader.batch_size * len(loader) / (time_post-time_pre)))
    # all_preds = torch.cat((all_preds, add_pre), dim=0)
    # all_labels = torch.cat((all_labels, add_lab), dim=0)
    return all_labels.numpy(), all_preds.numpy()

def get_all_preds_labels_race(model, loader, device):
    # switch to evaluate mode
    model.eval()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([], dtype=torch.int32)
    all_gender = ()
    all_race = ()

    gender_male_preds = torch.tensor([])
    gender_male_labels = torch.tensor([], dtype=torch.int32)

    gender_female_preds = torch.tensor([])
    gender_female_labels = torch.tensor([], dtype=torch.int32)

    race_black_preds = torch.tensor([])
    race_black_labels = torch.tensor([], dtype=torch.int32)

    race_white_preds = torch.tensor([])
    race_white_labels = torch.tensor([], dtype=torch.int32)

    race_asian_preds = torch.tensor([])
    race_asian_labels = torch.tensor([], dtype=torch.int32)

    add_pre = torch.tensor([1,0]*200)
    add_pre = add_pre.view((200, 2))
    add_lab = torch.ones((200))

    intersect_gender_race_preds = {
        'male':{
            'black': torch.tensor([]),
            'white': torch.tensor([]),
            'asian': torch.tensor([]),
            'other': torch.tensor([]),
        },
        'female':{
            'black': torch.tensor([]),
            'white': torch.tensor([]),
            'asian': torch.tensor([]),
            'other': torch.tensor([]),
        }
    }

    intersect_gender_race_labels = {
        'male':{
            'black': torch.tensor([], dtype=torch.int32),
            'white': torch.tensor([], dtype=torch.int32),
            'asian': torch.tensor([], dtype=torch.int32),
            'other': torch.tensor([], dtype=torch.int32),
        },
        'female':{
            'black': torch.tensor([], dtype=torch.int32),
            'white': torch.tensor([], dtype=torch.int32),
            'asian': torch.tensor([], dtype=torch.int32),
            'other': torch.tensor([], dtype=torch.int32),
        }
    }

    batch_size = 0
    # all_labels_mask = torch.tensor([], dt ype=torch.int32)
    time_pre = time.time()
    wrong_list = []

    # wrong_list = open('/root/autodl-tmp/PRLE/wrong_xcep_wo_wrong.txt', 'r').readlines()
    # visual(model, wrong_list)
    with torch.no_grad():
        for images, labels, gender, race in tqdm(loader, mininterval=10.0):
            images, labels = images.to(device), labels.to(device)
            # with autocast():
            outputs, _ = model(images)
            # outputs = model(images)

            # if isinstance(outputs, list):
            #     outputs = outputs[0]
            # else:
            #     outputs = outputs
            # GFLOPS of face model
            # flops_face = FlopCountAnalysis(model, images)
            # print("FLOPs face: ", flops_face.total() / 1e9)
            preds = F.softmax(outputs, dim=1)
            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            all_labels = torch.cat((all_labels, labels.cpu().int()), dim=0)
            all_gender = all_gender + gender
            all_race = all_race + race
            # print(gender)
            # print(len(labels))
            for ind in range(len(labels)):
                intersect_gender_race_labels[gender[ind]][race[ind]] = torch.cat((intersect_gender_race_labels[gender[ind]][race[ind]], labels[ind].unsqueeze(0).cpu().int()), dim=0)
                intersect_gender_race_preds[gender[ind]][race[ind]] = torch.cat((intersect_gender_race_preds[gender[ind]][race[ind]], preds[ind].unsqueeze(0).cpu()), dim=0)
                if gender[ind] == 'male':
                    gender_male_preds = torch.cat((gender_male_preds, preds[ind].unsqueeze(0).cpu()), dim=0)
                    gender_male_labels = torch.cat((gender_male_labels, labels[ind].unsqueeze(0).cpu().int()), dim=0)
                elif gender[ind] == 'female':
                    gender_female_preds = torch.cat((gender_female_preds, preds[ind].unsqueeze(0).cpu()), dim=0)
                    gender_female_labels = torch.cat((gender_female_labels, labels[ind].unsqueeze(0).cpu().int()), dim=0)

                if race[ind] == 'black':
                    race_black_preds = torch.cat((race_black_preds, preds[ind].unsqueeze(0).cpu()), dim=0)
                    race_black_labels = torch.cat((race_black_labels, labels[ind].unsqueeze(0).cpu().int()), dim=0)
                elif race[ind] == 'white':
                    race_white_preds = torch.cat((race_white_preds, preds[ind].unsqueeze(0).cpu()), dim=0)
                    race_white_labels = torch.cat((race_white_labels, labels[ind].unsqueeze(0).cpu().int()), dim=0)
                elif race[ind] == 'asian':
                    race_asian_preds = torch.cat((race_asian_preds, preds[ind].unsqueeze(0).cpu()), dim=0)
                    race_asian_labels = torch.cat((race_asian_labels, labels[ind].unsqueeze(0).cpu().int()), dim=0)

        #     pred_result = np.argmax(preds.cpu().detach(), axis=-1)
        #     for i in range(len(labels)):
        #         if labels[i] != pred_result[i]:
        #             wrong_list.append(path[i])
        # wrong_list.sort()
        # with open('./wrong_xcep_wo_PRLE.txt', 'w') as file:
        #     for i in wrong_list:
        #         file.writelines(i)
        #         file.writelines('\n')

    time_post = time.time()
    print("Video Throughout in this evaluation: %.4f Video/s" % (loader.batch_size * len(loader) / (time_post-time_pre)))
    # all_preds = torch.cat((all_preds, add_pre), dim=0)
    # all_labels = torch.cat((all_labels, add_lab), dim=0)
    return all_labels.numpy(), all_preds.numpy(), {'male_label': gender_male_labels.numpy(),
                                                   'male_pred': gender_male_preds.numpy(),
                                                   'female_label': gender_female_labels.numpy(),
                                                   'female_pred': gender_female_preds.numpy(),
                                                   'intersect_gender_race_labels': intersect_gender_race_labels,
                                                   'intersect_gender_race_preds': intersect_gender_race_preds}, {'white_label': race_white_labels.numpy(),
     'white_pred': race_white_preds.numpy(),
     'black_label': race_black_labels.numpy(),
     'black_pred': race_black_preds.numpy(),
     'asian_label': race_asian_labels.numpy(),
     'asian_pred': race_asian_preds.numpy()}, all_gender, all_race

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import cv2
def save_fig_npy(mask_pred, mask_label, ori_image, cnt_pred, cnt_label):
    import matplotlib.pyplot as plt
    path = './save_fig/NT_F3_Sup/'
    if not os.path.exists(path):
        os.makedirs(path)

    # new_image = cv2.bitwise_or(image, image, mask=mask)
    for ind in range(ori_image.shape[0]):
        img = Image.fromarray(ori_image[ind])

        mask_pred_i = mask_pred[ind]
        img_mask_pred_i = Image.fromarray(np.int8(mask_pred_i*255))
        img_mask_pred_i = img_mask_pred_i.convert('1')

        mask_label_i = mask_label[ind]
        img_mask_label_i = Image.fromarray(np.int8(mask_label_i*255))
        img_mask_label_i = img_mask_label_i.convert('1')

        img_pred = img.resize((mask_pred_i.shape[0], mask_pred_i.shape[1]))
        img_pred = np.array(img_pred)
        mask_t = np.int8((1-mask_pred_i)*255)
        pred_mask = cv2.bitwise_or(img_pred, img_pred, mask=np.int8((1-mask_pred_i)*255))
        pred_mask = Image.fromarray(pred_mask)

        img_label = img.resize((mask_label_i.shape[0], mask_label_i.shape[1]))
        img_label = np.array(img_label)
        label_mask = cv2.bitwise_or(img_label, img_label, mask=np.int8((1-mask_label_i)*255))
        label_mask = Image.fromarray(label_mask)

        img.save(os.path.join(path, '%05d_ori.png') % (cnt_label))
        img_mask_pred_i.save(os.path.join(path, '%05d_pred.png') % (cnt_label))
        img_mask_label_i.save(os.path.join(path, '%05d_label.png') % (cnt_label))
        pred_mask.save(os.path.join(path, '%05d_pred_mask.png') % (cnt_label))
        label_mask.save(os.path.join(path, '%05d_label_mask.png') % (cnt_label))

        cnt_label += 1

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.imshow(img_mask_pred_i)
        # plt.imshow(img_mask_label_i)
        # plt.imshow(pred_mask)
        # plt.imshow(label_mask)
        # plt.show()
        # plt.close()


    return cnt_pred, cnt_label