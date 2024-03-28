import torch

img_size_youtube = {
    'IMG_SIZE': 256,
    'IMG_SIZE2': 448,
    'SCALE_LIMIT': 100,
}

img_size_auris = {
    'IMG_SIZE': 220,
    'IMG_SIZE2': 220,
}

img_size_intuitive = {
    'IMG_SIZE': 224, # 513, 224
    'IMG_SIZE2': 224, # 513, 224
}

img_size_uavid = {
    'IMG_SIZE': 540,
    'IMG_SIZE2': 960,
    'SCALE_LIMIT': 400,
}

img_size_a2d2 = {
    'IMG_SIZE': 270,
    'IMG_SIZE2': 480,
    'SCALE_LIMIT': 200,
}

img_size_cityscapes = {
    'IMG_SIZE': 769,
    'IMG_SIZE2': 769,
    'SCALE_LIMIT': 300,
}

img_size_pascalVOC = {
    'IMG_SIZE': 513,
    'IMG_SIZE2': 513,
    'SCALE_LIMIT': 200,
}

multi_class_label_number = {
    'auris': 8,
    'intuitive': 12,
    'pets': 7,
    'uavid': 8,
    'a2d2': 19,
    'cityscapes': 19,
    'pascal_VOC': 21,
}

dataset_pretrained_paths = {
    'auris': "../pretrained_models/auris_seg_simCLRv5/best_train_acc.pth.tar",
    'intuitive': '../pretrained_models/intuitive_simCLR_v2/pretrained_epoch=884.pth.tar',
    'pets': '../pretrained_models/pets_simCLR/best_train_acc.pth.tar',
    'uavid': "../pretrained_models/UAVID_simCLRv4/best_train_acc.pth.tar",
    'a2d2': "../pretrained_models/a2d2_simCLR_v2/best_train_acc.pth.tar",
    'cityscapes': "",
    'pascal_VOC': "",
}

config = {
    "DATASET": "cityscapes", 
    "LEARNING_RATE": 1e-4,
    "MAX_PATIENCE": 10,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 4,
    "EVAL_BATCH_SIZE": 16,
    "NUM_WORKERS": 0,
    "EPOCHS": int(1e7),

    "RUNNING_COEF": 0.99,
    "EVAL_METRIC": "IoU",
    "SAMPLING_METRIC": "AUROC",
    "CONTINUE_TRAIN": False,
    "IMG_PER_CLASS": 1, 
    
    "PATCH_NUMBER": None, 
    "PATCH_SHAPE": "superpixel",
    "DOMINANT_LABELLING": True,
    
    "INIT_FRAME_PER_VIDEO": 1,
    "INIT_NUM_VIDEO": 10,
    "INIT_NUM_IMAGE": 20,
    
    "NUM_QUERY": 20,
    "NUM_ROUND": 6,
    "SAMPLING": "random", # "class_entropy_patch_multiClassAL",  # "BvSB_patch_v2", "pixelBal_v2", "revisiting_v2", "CBAL_v2", "class_entropy_ML_pred_patch"
    
    "EXP_NAME": 'ads/v7/AL_sda',
    "SAVE_CONTINUE_ROUND": False,
    
    "MODEL_ARCH": 'deeplabv3_resnet50', # 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'unet', 'vit'
    "VIT_BACKBONE": 'vit_small_patch16_384',
    "VIT_DECODER": 'mask_transformer',
    
    "START_SEED": 0,
    "TOTAL_SEEDS": 10,
    "NUM_FOLDS": 5,
    "NB_SAVED_BEST_VAL": 0,
}

config['N_LABEL'] = multi_class_label_number[config['DATASET']]
config['PRETRAINED_WEIGHT_PATH'] = dataset_pretrained_paths[config['DATASET']]
