import logging
import os
import random
import torch.nn as nn
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from PIL import Image
import cv2
import logging
import numpy as np
from heatmap_fusion import show_pil_image, single_pix_per_heatmaps

def localization_mask(activation, grad, target=None):
    grad_weight = grad.mean(axis=(2, 3), keepdim=True)
    weight_activation = activation * grad_weight
    # cam = weight_activation.sum(axis=1)
    cam = torch.relu(weight_activation)
    return cam


def mask_test(input_image, cam):
    grayscale_cam = cam(input_tensor=input_image)
    grayscale_cam = np.int64(grayscale_cam > 0.25)
    return grayscale_cam


def calculate_ratio(npy):
    large_than_zero = np.sum(npy>=0.01)
    return large_than_zero

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def select_layer(model, model_select):
    if model_select == 'xception':
        target_layers = [model.model.conv4, model.model.conv3]
    elif model_select == 'f3net':
        target_layers = [model.FAD_xcep.conv4]
    elif model_select == 'efficient':
        target_layers = [model._conv_head]
    elif model_select == 'vgg':
        target_layers = [model.features[-1], model.features[-2], model.features[-3]]
    elif model_select == 'ViT':
        target_layers = [model.encoder.layers.encoder_layer_23.ln_1]
    else:
        target_layers = []
        print(
            'Not supporting this network, please manually edit the target layers in ./utils./heatmap_vis.py')
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    return cam, train_transforms


def select_multi_layer(model, model_select):
    # print(model)
    target_layers = [
        model.model.conv1,
        model.model.conv2,
        model.model.block1,
        model.model.block2,
        model.model.block3,
        model.model.block4,
        model.model.block5,
        model.model.block6,
        model.model.block7,
        model.model.block8,
        model.model.block9,
        model.model.block10,
        model.model.block11,
        model.model.block12,
        model.model.conv3,
        model.model.conv4,
    ]
    cam_list = []
    for layer in target_layers:
        cam = GradCAM(model=model, target_layers=[layer], use_cuda=True)
        cam_list.append(cam)
    train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    return cam_list, train_transforms


def visual_multi_layer(model, config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    model.eval()
    # for name, v in model.state_dict().items():
    #     print(name)
    # with torch.no_grad():
    label_list = {
        'fake': 0,
        'real': 1
    }
    large_than_zero = 0
    all_zr = 0

    img_path = []
    logger.info('The data path is %s' % (config['vis']['data_path']))
    heatmap_save_path = config['vis']['cam_save_path']
    for root, _, files in os.walk(config['vis']['data_path']):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                img_path.append(os.path.join(root, file))
    img_path.sort()
    random.seed(1)
    random.shuffle(img_path)

    statistics_max_value = []
    statistics_mean_value = []
    cnt = 0
    for image in tqdm(img_path):
        cam_list, train_transforms = select_multi_layer(model, model_select=config['general']['model_select'])
        img = Image.open(image).convert('RGB')  # Get your input
        # img.show()
        input_tensor = train_transforms(img).to('cuda')  # the input tensor after data preprocessing fedded into model
        img_ori = cv2.imread(image)  # original image
        img_ori = cv2.resize(img_ori, (299, 299))  # resize original image to be overlayyed by heatmap

        cnt_layer = 0
        mean_values = []
        max_values = []
        for cam in cam_list:
            targets_0 = [ClassifierOutputTarget(label_list[config['vis']['img_label']])]
            grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0))
            grayscale_cam_0 = grayscale_cam_0[0, :]

            grayscale_cam = grayscale_cam_0
            save_npy = grayscale_cam

            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = Image.fromarray(heatmap)
            #
            # result = show_cam_on_image(img_ori / 255, grayscale_cam, use_rgb=True)
            # result = Image.fromarray(result)
            # save_image_name = config['vis']['save_template'] % (cnt, config['general']['model_select'], config['vis']['dataset_select'], cnt_layer)
            # os.makedirs(os.path.join(heatmap_save_path), exist_ok=True)
            # result.save(os.path.join(heatmap_save_path, save_image_name))
            cnt_layer += 1
            flattened = grayscale_cam.flatten()
            top_1000_values = np.sort(flattened)[-1000:]
            all_value = np.sort(flattened)[:]
            max_val, mean_val = np.mean(top_1000_values), np.mean(all_value)
            mean_values.append(mean_val)
            max_values.append(max_val)
        statistics_max_value.append(max_values)
        statistics_mean_value.append(mean_values)
        cnt += 1
        if cnt > 15:
            # print(statistics_max_value)
            # print(statistics_mean_value)
            plt.figure(figsize=(12, 6))
            for i, values in enumerate(statistics_max_value):
                plt.plot(values, marker='o', color='blue')
            for i, values in enumerate(statistics_mean_value):
                plt.plot(values, marker='.', color='orange')
            plt.xlabel('Image Index')
            plt.ylabel('Value')
            plt.title('Normalized Values of Grad-CAM Images')
            plt.legend()
            plt.grid(True)
            plt.show()
            exit()



def visual(model, config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    model.eval()
    # for name, v in model.state_dict().items():
    #     print(name)
    # with torch.no_grad():
    label_list = {
        'fake': 0,
        'real': 1
    }
    large_than_zero = 0
    all_zr = 0

    img_path = []
    logger.info('The data path is %s' % (config['vis']['data_path']))
    heatmap_save_path = config['vis']['cam_save_path']
    for root, _, files in os.walk(config['vis']['data_path']):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                img_path.append(os.path.join(root, file))
    img_path.sort()
    random.seed(1)
    random.shuffle(img_path)

    cnt = 0
    for image in tqdm(img_path):
        cam, train_transforms = select_layer(model, model_select=config['general']['model_select'])
        img = Image.open(image).convert('RGB')  # Get your input
        # img.show()
        input_tensor = train_transforms(img).to('cuda')  # the input tensor after data preprocessing fedded into model
        img_ori = cv2.imread(image)  # original image
        img_ori = cv2.resize(img_ori, (299, 299))  # resize original image to be overlayyed by heatmap

        targets_0 = [ClassifierOutputTarget(label_list[config['vis']['img_label']])]
        grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0))
        grayscale_cam_0 = grayscale_cam_0[0, :]


        grayscale_cam = grayscale_cam_0 / 2
        save_npy = grayscale_cam

        large_than_zero += calculate_ratio(save_npy)
        all_zr += save_npy.size

        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = Image.fromarray(heatmap)
        #
        result = show_cam_on_image(img_ori / 255, grayscale_cam, use_rgb=True)
        result = Image.fromarray(result)
        save_image_name = config['vis']['save_template'] % (cnt, config['general']['model_select'], config['vis']['dataset_select'])
        os.makedirs(os.path.join(heatmap_save_path), exist_ok=True)
        result.save(os.path.join(heatmap_save_path, save_image_name))
        cnt += 1
        if cnt > 300:
            exit()



    # for img_label in config['vis']['img_label']:
    #     for data_split in config['vis']['data_split']:
    #         if config['vis']['subset'] is None:
    #             logger.info('Visualize the %s %s images' % (img_label, data_split))
    #             logger.info('The data path is %s' % (config['vis']['data_path']))
    #             img_paths = os.path.join(config['vis']['data_path'], data_split, img_label)
    #             if not os.path.isdir(img_paths):
    #                 continue
    #             logger.info('The data path is %s' % img_paths)
    #             out_img_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                         data_split, img_label, 'origin')
    #             res_img_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                         data_split, img_label, 'visual')
    #             heatmap_save_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                              data_split, img_label, 'mask')
    #             npy_save_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                          data_split, img_label, 'npy')
    #             logger.info('The mask save path is %s' % heatmap_save_path)
    #             if config['vis']['save_npy']:
    #                 logger.info('The npy save path is %s' % npy_save_path)
    #             #
    #             # if not os.path.exists(out_img_path):
    #             #     os.makedirs(out_img_path)
    #             # if not os.path.exists(res_img_path):
    #             #     os.makedirs(res_img_path)
    #             if config['vis']['save_npy']:
    #                 if not os.path.exists(npy_save_path):
    #                     os.makedirs(npy_save_path)
    #             if not os.path.exists(heatmap_save_path):
    #                 os.makedirs(heatmap_save_path)
    #
    #
    #
    #             # print(model)
    #
    #
    #             img_paths_list = os.listdir(img_paths)
    #             img_paths_list.sort()
    #             for image in tqdm(img_paths_list):
    #                 image_name_list = os.listdir(os.path.join(img_paths, image))
    #                 image_name_list.sort()
    #                 cnt = 1
    #                 for image_name in image_name_list:
    #                     img_path = os.path.join(img_paths, image, image_name)
    #                     img = Image.open(img_path).convert('RGB')  # Get your input
    #                     # img.show()
    #                     input_tensor = train_transforms(
    #                         img).cuda()  # the input tensor after data preprocessing fedded into model
    #                     img_ori = cv2.imread(img_path)  # original image
    #                     img_ori = cv2.resize(img_ori,
    #                                          (299, 299))  # resize original image to be overlayyed by heatmap
    #
    #                     targets_0 = [ClassifierOutputTarget(label_list[img_label])]
    #                     grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets_0)
    #                     grayscale_cam_0 = grayscale_cam_0[0, :]
    #
    #                     # targets_1 = [ClassifierOutputTarget(1)]
    #                     # grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets_1)
    #                     # grayscale_cam_0 = grayscale_cam_0[0, :]
    #
    #                     grayscale_cam = grayscale_cam_0 / 2
    #                     save_npy = grayscale_cam
    #
    #                     large_than_zero += calculate_ratio(save_npy)
    #                     all_zr += save_npy.size
    #
    #                     heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    #                     heatmap = Image.fromarray(heatmap)
    #                     #
    #                     result = show_cam_on_image(img_ori / 255, grayscale_cam, use_rgb=True)
    #                     result = Image.fromarray(result)
    #
    #                     # show = [result]
    #                     # show_pil_image(show)
    #
    #                     # if not os.path.exists(os.path.join(out_img_path, image)):
    #                     #     os.makedirs(os.path.join(out_img_path, image))
    #                     # if not os.path.exists(os.path.join(res_img_path, image)):
    #                     #     os.makedirs(os.path.join(res_img_path, image))
    #                     save_image_name = "%04d_wo_mask.png" % cnt
    #                     if config['vis']['save_npy']:
    #                         if not os.path.exists(os.path.join(npy_save_path, save_image_name)):
    #                             os.makedirs(os.path.join(npy_save_path, save_image_name))
    #                     if not os.path.exists(os.path.join(heatmap_save_path, save_image_name)):
    #                         os.makedirs(os.path.join(heatmap_save_path, save_image_name))
    #
    #                     # img.resize(size=(256, 256)).save(os.path.join(out_img_path, image, image_name))  # save image
    #                     # result.resize(size=(256, 256)).save(os.path.join(res_img_path, image, image_name))  # save image
    #                     result.save(os.path.join(heatmap_save_path, save_image_name))
    #                     if config['vis']['save_npy']:
    #                         np.save(os.path.join(npy_save_path, image, '%s.npy' % (image_name)), save_npy)
    #                     cnt += 1
    #                     if cnt > 300:
    #                         exit()
    #         else:
    #             for subset in config['vis']['subset']:
    #                 logger.info('Visualize the %s %s images of %s set' %(img_label, data_split, subset))
    #                 logger.info('The data path is %s' % (config['vis']['data_path']))
    #                 img_paths = os.path.join(config['vis']['data_path'], data_split, img_label, subset)
    #                 if not os.path.isdir(img_paths):
    #                     continue
    #                 logger.info('The data path is %s' % img_paths)
    #                 out_img_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                              data_split, img_label, subset, 'origin')
    #                 res_img_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                              data_split, img_label, subset, 'visual')
    #                 heatmap_save_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                              data_split, img_label, subset, 'mask')
    #                 npy_save_path = os.path.join(config['vis']['cam_save_path'], config['general']['model_select'],
    #                                              data_split, img_label, subset, 'npy')
    #                 logger.info('The mask save path is %s' % heatmap_save_path)
    #                 if config['vis']['save_npy']:
    #                     logger.info('The npy save path is %s' % npy_save_path)
    #                 #
    #                 # if not os.path.exists(out_img_path):
    #                 #     os.makedirs(out_img_path)
    #                 # if not os.path.exists(res_img_path):
    #                 #     os.makedirs(res_img_path)
    #                 if config['vis']['save_npy']:
    #                     if not os.path.exists(npy_save_path):
    #                         os.makedirs(npy_save_path)
    #                 if not os.path.exists(heatmap_save_path):
    #                     os.makedirs(heatmap_save_path)
    #
    #                 train_transforms = transforms.Compose([
    #                     transforms.Resize((299, 299)),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize([0.5] * 3, [0.5] * 3)
    #                 ])
    #
    #
    #                 cam = select_layer(model, model_select=config['general']['model_select'])
    #
    #                 img_paths_list = os.listdir(img_paths)
    #                 img_paths_list.sort()
    #                 for image in tqdm(img_paths_list):
    #                     image_name_list = os.listdir(os.path.join(img_paths, image))
    #                     image_name_list.sort()
    #                     cnt = 1
    #                     for image_name in image_name_list:
    #                         img_path = os.path.join(img_paths, image, image_name)
    #                         img = Image.open(img_path).convert('RGB')  # Get your input
    #                         # img.show()
    #                         input_tensor = train_transforms(img).cuda()  # the input tensor after data preprocessing fedded into model
    #                         img_ori = cv2.imread(img_path)  # original image
    #                         img_ori = cv2.resize(img_ori, (299, 299))  # resize original image to be overlayyed by heatmap
    #                         img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    #
    #                         targets_0 = [ClassifierOutputTarget(label_list[img_label])]
    #                         grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets_0)
    #                         grayscale_cam_0 = grayscale_cam_0[0, :]
    #
    #                         # targets_1 = [ClassifierOutputTarget(1)]
    #                         # grayscale_cam_0 = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets_1)
    #                         # grayscale_cam_0 = grayscale_cam_0[0, :]
    #
    #                         grayscale_cam = grayscale_cam_0 / 2
    #                         grayscale_cam = single_pix_per_heatmaps(np.expand_dims(grayscale_cam, axis=0))
    #                         save_npy = grayscale_cam
    #
    #                         large_than_zero += calculate_ratio(save_npy)
    #                         all_zr += img_ori.size
    #
    #                         if config['vis']['save_result']:
    #                             heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    #                             heatmap = Image.fromarray(heatmap)
    #                             #
    #                             result = show_cam_on_image(img_ori/255, grayscale_cam, use_rgb=True, image_weight=0.5)
    #                             result = Image.fromarray(result)
    #
    #                             # show = [result]
    #                             # show_pil_image(show)
    #
    #                             # if not os.path.exists(os.path.join(out_img_path, image)):
    #                             #     os.makedirs(os.path.join(out_img_path, image))
    #                             # if not os.path.exists(os.path.join(res_img_path, image)):
    #                             #     os.makedirs(os.path.join(res_img_path, image))
    #                             if config['vis']['save_npy']:
    #                                 if not os.path.exists(os.path.join(npy_save_path, image)):
    #                                     os.makedirs(os.path.join(npy_save_path, image))
    #                             if not os.path.exists(os.path.join(heatmap_save_path)):
    #                                 os.makedirs(os.path.join(heatmap_save_path))
    #
    #                             save_image_name = "%04d_w_mask.png" % cnt
    #                             result.save(os.path.join(heatmap_save_path, save_image_name))
    #                         if config['vis']['save_npy']:
    #                             np.save(os.path.join(npy_save_path, image, '%s.npy'%(save_image_name)), save_npy)
    #                         cnt += 1
    #                         if cnt > 300:
    #                             exit()
    #                 print("primary regions ratio in subset %s is %.4f"%(subset, large_than_zero/all_zr))