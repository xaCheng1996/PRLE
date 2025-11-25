import cv2
import os
import torch
import numpy as np
import yaml
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/yaml/heatmap_fusion.yaml')
    args = parser.parse_args()
    return args


def show_fig_list(fig_list, save_path=None):
    fig = plt.figure()
    length = len(fig_list)
    h = int(length ** 0.5)
    w = length // h + 1
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, length + 1):
        ax = fig.add_subplot(w, h, i)
        ax = plt.imshow(
            np.uint8(255 * fig_list[i - 1])
        )
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def show_pil_image(fig_list):
    fig = plt.figure()
    length = len(fig_list)
    h = int(length ** 0.5)
    w = length // h + 1
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, length + 1):
        ax = fig.add_subplot(w, h, i)
        ax = plt.imshow(
            fig_list[i-1]
        )
    plt.show()


def single_pix_per_heatmaps(img_gray=None, fusion_method='avg'):
    # img_list_sum_dim_1 = np.sum(img_list, axis=-1)
    img_list_sum_dim_1 = img_gray
    if fusion_method == 'avg':
        img_fusion = np.average(img_list_sum_dim_1, axis=0)
    elif fusion_method == 'max':
        img_fusion = np.max(img_list_sum_dim_1, axis=0)
    else:
        logger.info('Not supporting this fusion method, use avg as default')
        img_fusion = np.average(img_list_sum_dim_1, axis=0)

    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # for i in range(1, len(img_list_sum_dim_1)+1):
    #     ax = fig.add_subplot(4, 3, i)
    #     # ax = plt.imshow(cv2.cvtColor(
    #     #     cv2.applyColorMap(np.uint8(255 * img_list_sum_dim_1[i-1]), cv2.COLORMAP_JET),
    #     #     cv2.COLOR_BGR2RGB))
    #     ax = plt.imshow(
    #         np.uint8(255 * img_list_sum_dim_1[i-1])
    #     )
    # ax = fig.add_subplot(4, 3, 10)
    # # ax = plt.imshow(cv2.cvtColor(
    # #     cv2.applyColorMap(np.uint8(255 * img_fusion), cv2.COLORMAP_JET),
    # #     cv2.COLOR_BGR2RGB))
    # ax = plt.imshow(
    #     np.uint8(255 * img_fusion)
    # )
    #
    # ax = fig.add_subplot(4, 3, 11)
    # cam = 0.5 * cv2.cvtColor(img_ori[0], cv2.COLOR_BGR2RGB) + 0.5 * \
    #       cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_list_sum_dim_1[0]), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    # cam = cam / np.max(cam)
    # ax = plt.imshow(
    #     np.uint8(255 * cam)
    # )
    #
    # ax = fig.add_subplot(4, 3, 12)
    # # ax = plt.imshow(cv2.cvtColor(
    # #     cv2.applyColorMap(np.uint8(255 * img_fusion), cv2.COLORMAP_JET),
    # #     cv2.COLOR_BGR2RGB))
    # cam = 0.5 * cv2.cvtColor(img_ori[0], cv2.COLOR_BGR2RGB) + 0.5 * cv2.applyColorMap(np.uint8(255 * img_fusion), cv2.COLORMAP_JET)
    # cam = cam / np.max(cam)
    # ax = plt.imshow(
    #     np.uint8(255 * cam)
    # )
    # plt.show()
    img_fusion_new = np.zeros_like(img_fusion)  # 创建一个与输入数组相同形状的全零数组
    img_fusion_new[img_fusion > 0.3] = img_fusion[img_fusion > 0.3]
    return img_fusion_new


def masked_generation_from_fusion(img_fusion, img_gray, tau_1, tau_2, tau_3):
    img_fusion_new = np.zeros_like(img_fusion)
    for i in range(img_fusion.shape[0]):
        for j in range(img_fusion.shape[1]):
            if img_fusion[i][j] > tau_1:
                img_fusion_new[i][j] = img_fusion[i][j]
            elif img_fusion[i][j] < tau_2:
                img_fusion_new[i][j] = 0
            else:
                # img_fusion_new[i][j] = fusion_neighbour([i, j], img_gray, tau_3)
                # img_fusion_new[i][j] = 0
                img_fusion_new[i][j] = fusion_neighbour([i, j], img_gray, img_fusion, tau_3)

                # if fusion_neighbour([i, j], img_gray, tau_3):
                #     img_fusion_new[i][j] = img_fusion[i][j]
                # else:
                #     img_fusion_new[i][j] = 0

    return img_fusion_new


def fusion_neighbour(position, img_gray, img_fusion, tau_3):
    flag, max_n, max_n_pos = get_neighbour(position, img_gray, tau_3=tau_3)
    # return flag
    if flag:
        return img_fusion[max_n_pos[0]][max_n_pos[1]]
    else:
        return 0


def get_neighbour(position, img_gray, tau_3):
    i, j = position[0], position[1]

    # h=w=256
    t, h, w = img_gray.shape

    neighbour_point = []
    for h_plus in [-1, 0, 1]:
        for w_plus in [-1, 0, 1]:
            i_tmp = i + h_plus
            j_tmp = j + w_plus
            if 0 <= i_tmp < h and 0 <= j_tmp < w:
                neighbour_point.append((i_tmp, j_tmp))

    max_n = 0
    max_n_pos = []
    for m_k in img_gray:
        for m_j in img_gray:
            sum_nei = 0
            max_n_tmp = 0
            for i_n, j_n in neighbour_point:
                x_i = m_k[i][j]
                x_n = m_j[i_n][j_n]
                sum_nei += abs(x_n - x_i)
                max_n_tmp += x_n
                if x_n > max_n:
                    max_n = x_n
                    max_n_pos = [i_n, j_n]
                if x_i > max_n:
                    max_n = x_i
                    max_n_pos = [i, j]
            sum_nei = sum_nei / len(neighbour_point)
            # max_n_tmp = max_n_tmp / len(neighbour_point)
            # if max_n_tmp > max_n:
            #     max_n = max_n_tmp
            if sum_nei > tau_3:
                return True, max_n, max_n_pos
    return False, 0, [i, j]


def find_image(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if fname.lower().endswith('png'):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def main():
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # /home/heatmap
    heatmap_path = config['heatmap_path']
    for classes in config['classes']:
        heatmap_path_home = heatmap_path  # real fake
        fusion_save_path = config['fusion_save_path']
        fusion_avg_save_path = config['fusion_avg_save_path']
        logging.info('start to process the %s image @ path: %s' % (classes, heatmap_path_home))
        heatmap_model = config['heatmap_generate_model']  # [xception, f3net, srm]
        datasets = config['dataset']
        dataset_dist = config['dataset_dist']
        dataset_split = config['dataset_split']
        for dataset in datasets:
            for split in dataset_split:
                # logging.info('save path %s' % os.path.join(fusion_save_path, split, classes, dataset,))

                img_list = []
                for root, _, fnames in os.walk(os.path.join(heatmap_path_home, 'xception', split,
                                                            classes, dataset, 'npy')):
                    for fname in sorted(fnames):
                        if fname.lower().endswith('npy'):
                            path = os.path.join(os.path.basename(root), fname)
                            img_list.append(path)
                # for model in heatmap_model:
                #     for dist in dataset_dist:
                #         mask_path = os.path.join(heatmap_path_home, model, classes, '%s_%d' % (dataset, dist), 'mask')
                #         img_list = os.listdir(mask_path)
                #         img_num = len(img_list)

                img_list.sort()
                cnt = 0
                img_gray_list = []

                for img_name in tqdm(img_list):
                    img_feat = []
                    img_ori_feat = []
                    grayscale = []
                    fig_list = []
                    for model in heatmap_model:
                        gray_cam_base_path = os.path.join(heatmap_path_home, model, split, classes, dataset)
                        if not os.path.isdir(gray_cam_base_path): continue
                        gray_cam = os.path.join(gray_cam_base_path, 'npy', img_name)
                        gray_tmp = np.load(gray_cam)
                        gray_tmp = Image.fromarray(gray_tmp)
                        gray_tmp_resize = gray_tmp.resize((256, 256))
                        grayscale.append(np.expand_dims(np.array(gray_tmp_resize), axis=0))
                        # print(os.path.join(heatmap_path_home, model, classes, '%s_%d' % (dataset, dist), 'npy',
                        #                         '%s.npy' % img_name))
                        fig_list.append(gray_tmp)
                        fig_list.append(gray_tmp_resize)

                        # img_path = os.path.join(heatmap_path_home, model, classes,
                        #                         '%s_%s' % (dataset, dist), 'mask', img_name)
                        # img_feat.append(np.expand_dims(cv2.imread(img_path), axis=0))

                        ori_img_path = os.path.join('/home/harry/data/FF++_faces', split, classes, dataset, img_name.split('.npy')[0])
                        img_ori = cv2.imread(ori_img_path)
                        img_ori = cv2.resize(img_ori, (256, 256))
                        img_ori_feat.append(np.expand_dims(img_ori, axis=0))
                    # #
                    # #
                    img_ori = np.concatenate([i for i in img_ori_feat])
                    #
                    # """
                    # concat the gray image and fuse them into 1 npy file
                    # """
                    img_gray = np.concatenate([i for i in grayscale])

                    img_fusion = single_pix_per_heatmaps(img_gray, config['method_fusion'])

                    # img_hor_fusion = img_fusion

                    # img_hor_fusion = masked_generation_from_fusion(img_fusion, img_gray,
                    #                                            tau_1=config['hyper_fusion']['tau_1'],
                    #                                            tau_2=config['hyper_fusion']['tau_2'],
                    #                                            tau_3=config['hyper_fusion']['tau_3'])
                    video_name = os.path.dirname(img_name)

                    if not os.path.exists(os.path.join(fusion_avg_save_path, split, classes, dataset, video_name)):
                        os.makedirs(os.path.join(fusion_avg_save_path, split, classes, dataset, video_name))
                    np.save(os.path.join(fusion_avg_save_path, split, classes, dataset, img_name), img_fusion)

                    # if not os.path.exists(os.path.join(fusion_save_path, split, classes, dataset, video_name)):
                    #     os.makedirs(os.path.join(fusion_save_path, split, classes, dataset, video_name))
                    # np.save(os.path.join(fusion_save_path, split, classes, dataset, img_name), img_hor_fusion)

                    # fusion_img = Image.fromarray(img_hor_fusion)
                    # fusion_img.save(os.path.join(fusion_save_path, split, classes, dataset, img_name))
                    # img_gray_list.append(np.load(os.path.join(fusion_save_path, split, classes, dataset, img_name)))
                    '''
                    The 9 heatmaps from 3*3 algorithms and iterations
                    '''
                    # fig_show = []
                    # if cnt < 100:
                    #     cnt_gray = 0
                    #     cam = 0.5 * cv2.cvtColor(img_ori[0], cv2.COLOR_BGR2RGB) + \
                    #           0.5 * cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_hor_fusion), cv2.COLORMAP_JET),
                    #                              cv2.COLOR_BGR2RGB)
                    #     result = cam / np.max(cam)
                    #     # i_tmp = Image.fromarray(np.uint8(255 * result))
                    #     fig_show.append(result)
                    #     for i in range(img_gray.shape[0]):
                    #         cam = 0.5 * cv2.cvtColor(img_ori[i], cv2.COLOR_BGR2RGB) + \
                    #               0.5 * cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_gray[i]), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    #         result = cam / np.max(cam)
                    #         i_tmp = Image.fromarray(np.uint8(255*result))
                    #         i_tmp.save(os.path.join('/root/autodl-tmp/heat_map_visualization', '%03d_%02d_%s.png'%(cnt, cnt_gray, 'heat')))
                    #         # result = cam / np.max(cam)
                    #         fig_show.append(result)
                    #     show_fig_list(fig_show)
                    #     cnt_gray += 1
                    #     cnt += 1
                    """
                    the two heatmaps from ver and hor fusion
                    """
            # fig_show = []
            # if cnt < 100:
            #     cnt_gray = 0
            #     # cam = 0.5 * cv2.cvtColor(img_ori[0], cv2.COLOR_BGR2RGB) + \
            #     #       0.5 * cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_fusion), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            #     cam = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_fusion), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            #     result = cam / np.max(cam)
            #     i_tmp = Image.fromarray(np.uint8(255*result))
            #     i_tmp.save(os.path.join('/home/share/chengxuanang/heat_map_fus/hor_ver_compare', '%s_%03d_%02d_%s.png'%(dataset, cnt, cnt_gray, 'ori')))
            #     fig_show.append(result)
            #
            #     img_fusion_ver = img_fusion
            #     img_fusion_ver[img_fusion_ver < 0.5] = 0.0
            #     cam = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_fusion_ver), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            #     result = cam / np.max(cam)
            #     i_tmp = Image.fromarray(np.uint8(255*result))
            #     i_tmp.save(os.path.join('/home/share/chengxuanang/heat_map_fus/hor_ver_compare', '%s_%03d_%02d_%s.png'%(dataset, cnt, cnt_gray, 'ver')))
            #     fig_show.append(result)
            #
            #     # cam = 0.5 * cv2.cvtColor(img_ori[0], cv2.COLOR_BGR2RGB) + \
            #     #       0.5 * cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_hor_fusion), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            #     cam = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * img_hor_fusion), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            #     result = cam / np.max(cam)
            #     i_tmp = Image.fromarray(np.uint8(255*result))
            #     i_tmp.save(os.path.join('/home/share/chengxuanang/heat_map_fus/hor_ver_compare', '%s_%03d_%02d_%s.png'%(dataset, cnt, cnt_gray, 'hor')))
            #     fig_show.append(result)
            #     show_fig_list(fig_show)
            #     cnt_gray += 1
            #     cnt += 1
            # else:
            #     return 0

        import matplotlib.pyplot as plt
        # dic_1 = {
        #     '0.1': 0,
        #     '0.15': 0,
        #     '0.2': 0,
        #     '0.25': 0,
        #     '0.3': 0,
        #     '0.35': 0,
        #     '0.4': 0,
        #     '0.45': 0,
        #     '0.5': 0,
        #     '0.55': 0,
        #     '0.6': 0,
        #     '0.65': 0,
        #     '0.7': 0,
        #     '0.75': 0,
        #     '0.8': 0,
        #     '0.85': 0,
        #     '0.9': 0,
        #     '0.95': 0,
        #     '1.0': 0
        # }
        # x = np.arange(0, len(img_gray_list))
        # for img in tqdm(img_gray_list):
        #     for value_row in img:
        #         for value in value_row:
        #             if value <= 0.01:
        #                 continue
        #             elif value < 0.1:
        #                 dic_1['0.1'] += 1
        #             elif value < 0.15:
        #                 dic_1['0.15'] += 1
        #             elif value < 0.2:
        #                 dic_1['0.2'] += 1
        #             elif value < 0.25:
        #                 dic_1['0.25'] += 1
        #             elif value < 0.3:
        #                 dic_1['0.3'] += 1
        #             elif value < 0.35:
        #                 dic_1['0.35'] += 1
        #             elif value < 0.4:
        #                 dic_1['0.4'] += 1
        #             elif value < 0.45:
        #                 dic_1['0.45'] += 1
        #             elif value < 0.5:
        #                 dic_1['0.5'] += 1
        #             elif value < 0.55:
        #                 dic_1['0.55'] += 1
        #             elif value < 0.6:
        #                 dic_1['0.6'] += 1
        #             elif value < 0.65:
        #                 dic_1['0.65'] += 1
        #             elif value < 0.7:
        #                 dic_1['0.7'] += 1
        #             elif value < 0.75:
        #                 dic_1['0.75'] += 1
        #             elif value < 0.8:
        #                 dic_1['0.8'] += 1
        #             elif value < 0.85:
        #                 dic_1['0.85'] += 1
        #             elif value < 0.9:
        #                 dic_1['0.9'] += 1
        #             elif value < 0.95:
        #                 dic_1['0.95'] += 1
        #             elif value < 1.0:
        #                 dic_1['1.0'] += 1
        # print(dic_1)
        # sum_t = {
        #     '0.1': 0,
        #     '0.2':0,
        #     '0.3': 0,
        #     '0.4':0,
        #     '0.5': 0,
        #     '0.6':0,
        #     '0.7': 0,
        #     '0.8':0,
        #     '0.9': 0,
        #     '1.0':0
        # }
        # avg_t = {
        #     '0.1': 0,
        #     '0.2':0,
        #     '0.3': 0,
        #     '0.4':0,
        #     '0.5': 0,
        #     '0.6':0,
        #     '0.7': 0,
        #     '0.8':0,
        #     '0.9': 0,
        #     '1.0':0
        # }
        # for img in tqdm(img_gray_list):
        #     img = img.flatten()
        #     img = np.sort(img)[::-1]
        #     img = img[img > 0]
        #     for i in ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
        #         sum_t[str(i)] += np.sum(img[:int(float(i)*len(img))])
        #         avg_t[str(i)] += np.average(img[:int(float(i) * len(img))])
        # print(sum_t)
        # print(avg_t)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
