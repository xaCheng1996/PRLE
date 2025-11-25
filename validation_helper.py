import logging
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from utils.misc import get_all_preds_labels, get_all_preds_labels_race
import numpy as np
from utils.heatmap_vis import localization_mask
from utils.heatmap_vis import visual, mask_test, visual_multi_layer
import matplotlib.pyplot as plt
import wandb

def compute_eer(y_true, y_score):
    # 获取FPR, TPR, 阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    # 找到FPR和FNR最接近的点
    eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    threshold = thresholds[eer_threshold_index]

    return eer


def validation(logger, config, model, val_dataset_list, val_loader_list, device='CUDA', epoch=0):
    val_loader = val_loader_list['val_loader']
    dfdc_loader, df10_loader, celeb_loader = val_loader_list['dfdc_loader'], val_loader_list['df10_loader'], \
    val_loader_list['celeb_loader']
    wild_loader = val_loader_list['wild_loader']
    FFIW_loader = val_loader_list['FFIW_loader']

    y_val, y_val_pred = get_all_preds_labels(model, val_loader, device)
    val_acc = accuracy_score(y_val, np.argmax(y_val_pred, axis=-1))
    val_auc = roc_auc_score(y_true=y_val, y_score=y_val_pred[:, 1])
    ffpp_auc = val_auc
    if not config['data']['split']:
        # y_df10, y_df10_pred = get_all_preds_labels(model, df10_loader, device)
        y_celeb, y_celeb_pred = get_all_preds_labels(model, celeb_loader, device)
        y_dfdc, y_dfdc_pred = get_all_preds_labels(model, dfdc_loader, device)
        y_wild, y_wild_pred = get_all_preds_labels(model, wild_loader, device)
        y_FFIW, y_FFIW_pred = get_all_preds_labels(model, FFIW_loader, device)

        # df10_auc = roc_auc_score(y_true=y_df10, y_score=y_df10_pred[:, 1])
        celeb_auc = roc_auc_score(y_true=y_celeb, y_score=y_celeb_pred[:, 1])
        dfdc_auc = roc_auc_score(y_true=y_dfdc, y_score=y_dfdc_pred[:, 1])
        wild_auc = roc_auc_score(y_true=y_wild, y_score=y_wild_pred[:, 1])
        ffiw_auc = roc_auc_score(y_true=y_FFIW, y_score=y_FFIW_pred[:, 1])

        # val_auc += df10_auc
        val_auc += celeb_auc
        val_auc += dfdc_auc
        val_auc += wild_auc

        auc_list = {
            'val_auc': val_auc,
            'ffpp_auc': ffpp_auc,
            'df10_auc': 0,
            'dfdc_auc': dfdc_auc,
            'celeb_auc': celeb_auc,
            'wild_auc': wild_auc,
            'ffiw_auc': ffiw_auc,
        }

        logger.info(
            f"Network Epoch：{epoch + 1}/{config['network']['epochs_net']}. "
            f"val_auc: {val_auc, ffpp_auc, dfdc_auc, 0, celeb_auc, wild_auc}.\n")
    else:
        auc_list = {
            'ffpp_auc': val_auc,
        }
        logger.info(
            f"Network Epoch：{epoch + 1}/{config['network']['epochs_net']}. val_auc: {val_auc}.\n")

    return auc_list



def test(logger, config, model, test_dataset_list, test_loader_list, device='CUDA'):
    test_dataset, test_dfdc_dataset, test_df10_dataset, test_celeb_dataset = (test_dataset_list['test_dataset'],
                                                                              test_dataset_list['test_dfdc_dataset'],
                                                                              test_dataset_list['test_df10_dataset'],
                                                                              test_dataset_list['test_celeb_dataset'])

    test_wild_dataset = test_dataset_list['test_wild_dataset']

    test_loader, dfdc_loader, df10_loader, celeb_loader = test_loader_list['test_loader'], test_loader_list[
        'dfdc_loader'], test_loader_list['df10_loader'], test_loader_list['celeb_loader'],

    wild_loader = test_loader_list['wild_loader']
    FFIW_loader = test_loader_list['FFIW_loader']

    logger.info('start the testing...')
    logger.info(f"test_dataset.size: {len(test_dataset)}")
    logger.info(f"dfdc_dataset.size: {len(test_dfdc_dataset)}")
    logger.info(f"df10_dataset.size: {len(test_df10_dataset)}")
    logger.info(f"celeb_dataset.size: {len(test_celeb_dataset)}")
    logger.info(f"wild_dataset.size: {len(test_wild_dataset)}")

    if config['general']['vis']:
        logger.info('start the visual of images...')
        visual_multi_layer(model, config)
        exit()

    #
    # draw_distribution(values=y_FFIW_pred[:, 1], labels=y_FFIW)
    #
    df10_auc, celeb_auc, dfdc_auc, wild_auc, FFIW_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    df10_eer, celeb_eer, dfdc_eer, wild_eer, FFIW_eer = 0.0, 0.0, 0.0, 0.0, 0.0
    if not config['data']['split']:
        y_dfdc, y_dfdc_pred = get_all_preds_labels(model, dfdc_loader, device)
        print('*' * 80)
        logger.info('classification report on dfdc set:')
        logger.info(classification_report(y_dfdc, np.argmax(y_dfdc_pred, axis=-1), digits=4))
        logger.info('AUC on dfdc set:')
        dfdc_auc = roc_auc_score(y_true=y_dfdc, y_score=y_dfdc_pred[:, 1])
        dfdc_eer = compute_eer(y_true=y_dfdc, y_score=y_dfdc_pred[:, 1])

        y_pred_label = np.argmax(y_dfdc_pred, axis=-1)
        mask = (y_dfdc == 1)
        acc_label1 = accuracy_score(y_dfdc[mask], y_pred_label[mask])
        print("Accuracy for label=1:", acc_label1)

        logger.info(dfdc_auc)

        y_celeb, y_celeb_pred = get_all_preds_labels(model, celeb_loader, device)
        print('*' * 80)
        logger.info('classification report on celeb set:')
        logger.info(classification_report(y_celeb, np.argmax(y_celeb_pred, axis=-1), digits=4))
        logger.info('AUC on celeb set:')
        celeb_auc = roc_auc_score(y_true=y_celeb, y_score=y_celeb_pred[:, 1])
        celeb_eer = compute_eer(y_true=y_celeb, y_score=y_celeb_pred[:, 1])

        y_pred_label = np.argmax(y_celeb_pred, axis=-1)
        mask = (y_celeb == 1)
        acc_label1 = accuracy_score(y_celeb[mask], y_pred_label[mask])
        print("Accuracy for label=1:", acc_label1)

        logger.info(celeb_auc)

        y_wild, y_wild_pred = get_all_preds_labels(model, wild_loader, device)
        print('*' * 80)
        logger.info('classification report on wild set:')
        logger.info(classification_report(y_wild, np.argmax(y_wild_pred, axis=-1), digits=4))
        logger.info('AUC on wild set:')
        wild_auc = roc_auc_score(y_true=y_wild, y_score=y_wild_pred[:, 1])
        wild_eer = compute_eer(y_true=y_wild, y_score=y_wild_pred[:, 1])
        logger.info(wild_auc)

        y_FFIW, y_FFIW_pred = get_all_preds_labels(model, FFIW_loader, device)
        print('*' * 80)
        logger.info('classification report on FFIW set:')
        logger.info(classification_report(y_FFIW, np.argmax(y_FFIW_pred, axis=-1), digits=4))
        logger.info('AUC on FFIW set:')
        FFIW_auc = roc_auc_score(y_true=y_FFIW, y_score=y_FFIW_pred[:, 1])
        FFIW_eer = compute_eer(y_true=y_FFIW, y_score=y_FFIW_pred[:, 1])
        logger.info(FFIW_auc)

    y_test, y_test_pred = get_all_preds_labels(model, test_loader, device)
    print('*' * 80)
    logger.info('classification report on ffpp set:')
    logger.info(classification_report(y_test, np.argmax(y_test_pred, axis=-1), digits=4))
    logger.info('AUC on test set:')
    ffpp_auc = roc_auc_score(y_true=y_test, y_score=y_test_pred[:, 1])
    ffpp_eer = compute_eer(y_true=y_test, y_score=y_test_pred[:, 1])
    logger.info(ffpp_auc)
    y_df10, y_df10_pred = get_all_preds_labels(model, df10_loader, device)

    print('ACC/AUC/EER for all Dataset: \n FF++: %.4f  %.4f %.4f \n DFDC: %.4f  %.4f %.4f \n DF1.0: %.4f  %.4f %.4f \n Celeb-DF: %.4f  %.4f %.4f \n Wild-DF: %.4f  %.4f %.4f \n FFIW: %.4f  %.4f %.4f \n' % (
        accuracy_score(y_test, np.argmax(y_test_pred, axis=-1)), ffpp_auc, ffpp_eer,
        accuracy_score(y_dfdc, np.argmax(y_dfdc_pred, axis=-1)), dfdc_auc, dfdc_eer,
        accuracy_score(y_df10, np.argmax(y_df10_pred, axis=-1)), df10_auc, df10_eer,
        accuracy_score(y_celeb, np.argmax(y_celeb_pred, axis=-1)), celeb_auc, celeb_eer,
        accuracy_score(y_wild, np.argmax(y_wild_pred, axis=-1)), wild_auc, wild_eer,
        accuracy_score(y_FFIW, np.argmax(y_FFIW_pred, axis=-1)), FFIW_auc, FFIW_eer
    ))

    auc_list = {
        'ffpp_auc': ffpp_auc,
        'df10_auc': df10_auc,
        'dfdc_auc': dfdc_auc,
        'celeb_auc': celeb_auc,
        'wild_auc': wild_auc,
        'FFIW_auc': FFIW_auc,
    }

    return auc_list

