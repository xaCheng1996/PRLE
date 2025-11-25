import os
from dataset.ImageFolder import ImageFolder
from torch.utils.data import DataLoader


def data_load(logger, config, tranforms):
    train_dataset = ImageFolder(os.path.join(config['data']['data_path_train'], "train"),
                                config=config,
                                mask_path=config['data']['mask_path'],
                                transform=tranforms['train'],
                                )

    val_dataset = ImageFolder(os.path.join(config['data']['data_path_val'], "test"),
                              config=config,
                              mask_path=config['data']['mask_path'],
                              transform=tranforms['val'])
    test_dataset = ImageFolder(os.path.join(config['data']['data_path_test'], "test"),
                               config=config,
                               mask_path=config['data']['mask_path'],
                               transform=tranforms['test'])

    test_df10_dataset = ImageFolder(os.path.join(config['data']['data_path_df10'], "test"),
                                    config=config,
                                    mask_path=config['data']['mask_path'],
                                    transform=tranforms['test'])
    test_celeb_dataset = ImageFolder(os.path.join(config['data']['data_path_celeb_df'], "test"),
                                     config=config,
                                     mask_path=config['data']['mask_path'],
                                     transform=tranforms['test'])
    test_dfdc_dataset = ImageFolder(os.path.join(config['data']['data_path_dfdc'], "test"),
                                    config=config,
                                    mask_path=config['data']['mask_path'],
                                    transform=tranforms['test'])
    test_wild_dataset = ImageFolder(os.path.join(config['data']['data_path_wild'], "test"),
                                    config=config,
                                    mask_path=config['data']['mask_path'],
                                    transform=tranforms['test'])
    test_FFIW_dataset = ImageFolder(os.path.join(config['data']['data_path_ffiw'], "test"),
                                    config=config,
                                    mask_path=config['data']['mask_path'],
                                    transform=tranforms['test'])

    logger.info(f"train_dataset.classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)

    df10_loader = DataLoader(test_df10_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    celeb_loader = DataLoader(test_celeb_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    dfdc_loader = DataLoader(test_dfdc_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    wild_loader = DataLoader(test_wild_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    FFIW_loader = DataLoader(test_FFIW_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)

    train_dataset_list = {
        'train_dataset': train_dataset
    }

    val_dataset_list = {
        'val_dataset': val_dataset
    }

    test_dataset_list = {
        'test_dataset': test_dataset,
        'test_dfdc_dataset': test_dfdc_dataset,
        'test_df10_dataset': test_df10_dataset,
        'test_celeb_dataset': test_celeb_dataset,
        'test_wild_dataset': test_wild_dataset
    }

    train_loader_list = {
        'train_loader': train_loader
    }

    val_loader_list = {
        'val_loader': val_loader,
        'dfdc_loader': dfdc_loader,
        'df10_loader': df10_loader,
        'celeb_loader': celeb_loader,
        'wild_loader': wild_loader,
        'FFIW_loader': FFIW_loader
    }

    test_loader_list = {
        'test_loader': test_loader,
        'dfdc_loader': dfdc_loader,
        'df10_loader': df10_loader,
        'celeb_loader': celeb_loader,
        'wild_loader': wild_loader,
        'FFIW_loader': FFIW_loader,
    }

    return train_dataset_list, val_dataset_list, test_dataset_list, train_loader_list, val_loader_list, test_loader_list
