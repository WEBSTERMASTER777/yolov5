import yaml
import os
import glob
import shutil

from tqdm import tqdm
from loguru import logger
from yaml import load
from yaml import FullLoader


def create_class_list(filename):
    file = open(filename, "r")
    class_list = file.read().splitlines()
    file.close()
    return class_list


def create_config_data(train_path, val_path, classname_file,config_path):
    logger.info('Create config file')
    class_list = create_class_list(classname_file)
    data = dict(train=train_path,
                val=val_path,
                nc=len(class_list),
                names=class_list
                )
    with open(config_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


logger.info('Split dataset on train, test and val')


def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)

def split_data(datapath):
    try:
        os.mkdir('dataset')
        os.mkdir('dataset/train')
        os.mkdir('dataset/test')
        os.mkdir('dataset/val')

    except Exception:
        logger.exception('Directory with split dataset already exist')

    train_size = int(0.7 * len(os.listdir(datapath)))
    val_size = int(0.2 * len(os.listdir(datapath)))
    test_size = int(0.1 * len(os.listdir(datapath)))
    count = 0
    for data in tqdm(os.listdir(datapath)):
        if count < train_size:

            shutil.copy(datapath + '/' + data, 'dataset/train')
            count += 1

        if count >= train_size and count < train_size + val_size:

            shutil.copy(datapath + '/' + data, 'dataset/val')
            count += 1
        if count > train_size and count >= train_size + val_size:
            shutil.copy(datapath + '/' + data, 'dataset/test')
            count += 1

    logger.info('Count of train example: ' + str(train_size))
    logger.info('Count of val example: ' + str(val_size))
    logger.info("Count of test example: " + str(test_size))


if __name__ == '__main__':

    config = load_config('custom_config.yaml')

    DATA_PATH = config['DATA_PATH']
    CLASSES = config['CLASSES']
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    CONFIG_PATH = config['CONFIG_PATH']
    MODEL_PATH = config['MODEL_PATH']
    SAVE_PATH = config['SAVE_PATH']
    MODEL_NAME = config['MODEL_NAME']

    split_data(DATA_PATH)
    create_config_data('dataset/train', 'dataset/val', CLASSES, CONFIG_PATH)
    os.system(
        'python train.py --img ' +
        IMG_SIZE +
        ' --batch ' +
        BATCH_SIZE +
        ' --epochs ' +
        EPOCHS +
        ' --data ' +
        CONFIG_PATH +
        ' --cfg ' +
        MODEL_PATH +
        ' --weights ' +
        SAVE_PATH 
        )

