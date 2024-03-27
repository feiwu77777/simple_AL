## AURIS DATASET
IMG_PATH = "../dataset/auris_seg/images/"
LAB_PATH = "../dataset/auris_seg/labels/"
VIDEO_FRAMES_PATH = "../dataset/auris_seg/all_frames/"

IMG_PATH_NOTEBOOK = '../../datasets/auris/auris_seg/images/'
LAB_PATH_NOTEBOOK = '../../datasets/auris/auris_seg/labels/'

FRAME_KEYWORD = 'frame'
CLASS_ID_TYPE = 'xx-xx/'
ZFILL_NB = 5
NUM_TRAIN_VAL = 15

## SKATEBOARD DATASETS
DATAPATH_SKATEBOARD = "../dataset/skateboard/"
IMG_PATH_SKATEBOARD = "../dataset/skateboard/images/"
LAB_PATH_SKATEBOARD = "../dataset/skateboard/masks/"
NUM_TRAIN_VAL_SKATEBOARD = 25

DATAPATH_SKATEBOARD_NOTEBOOK = '../../datasets/youtube_VOS/skateboard/'
IMG_PATH_SKATEBOARD_NOTEBOOK = DATAPATH_SKATEBOARD_NOTEBOOK + 'images/'
LAB_PATH_SKATEBOARD_NOTEBOOK = DATAPATH_SKATEBOARD_NOTEBOOK + 'labels/'

## PARROTS DATASETS
DATAPATH_PARROTS = "../dataset/parrot/"
IMG_PATH_PARROTS = "../dataset/parrot/images/"
LAB_PATH_PARROTS = "../dataset/parrot/masks/"
VIDEO_FRAMES_PATH_PARROTS = "../dataset/parrot/all_frames"
NUM_TRAIN_VAL_PARROTS = 46

DATAPATH_PARROTS_NOTEBOOK = "../../datasets/youtube_VIS/2019/parrot/"
IMG_PATH_PARROTS_NOTEBOOK = DATAPATH_PARROTS_NOTEBOOK + "images/"
LAB_PATH_PARROTS_NOTEBOOK = DATAPATH_PARROTS_NOTEBOOK + "masks/"

## DOG DATASETS
DATAPATH_DOG = "../dataset/dog/"
IMG_PATH_DOG = "../dataset/dog/images/"
LAB_PATH_DOG = "../dataset/dog/masks/"
VIDEO_FRAMES_PATH_DOG = "../dataset/dog/all_frames"

DATAPATH_DOG_NOTEBOOK = "../../datasets/youtube_VIS/2019/dog/"
IMG_PATH_DOG_NOTEBOOK = DATAPATH_DOG_NOTEBOOK + "images/"
LAB_PATH_DOG_NOTEBOOK = DATAPATH_DOG_NOTEBOOK + "masks/"

## PETS DATASETS
DATAPATH_PETS = "../dataset/pets/"
# DATAPATH_PETS = "../dataset/pets_small/"
IMG_PATH_PETS = DATAPATH_PETS + "images/"
LAB_PATH_PETS = DATAPATH_PETS + "masks/"
VIDEO_FRAMES_PATH_PETS = "../dataset/pets/all_frames"
NUM_TRAIN_VAL_PETS = 120

DATAPATH_PETS_NOTEBOOK = "../../datasets/youtube_VIS/2019/pets/"
# DATAPATH_PETS_NOTEBOOK = "../../datasets/youtube_VIS/2019/pets_small/"
IMG_PATH_PETS_NOTEBOOK = DATAPATH_PETS_NOTEBOOK + "images/"
LAB_PATH_PETS_NOTEBOOK = DATAPATH_PETS_NOTEBOOK + "masks/"

## YOUTUBE VIS DATASETS
DATAPATH_VIS = "/storage/workspaces/artorg_aimi/ws_00000/fei/youtube_VIS/"
IMG_PATH_VIS = DATAPATH_VIS + "images/"
LAB_PATH_VIS = DATAPATH_VIS + "masks/"
VIDEO_FRAMES_PATH_VIS = "../dataset/pets/all_frames"
NUM_TRAIN_VAL_VIS = 3271

DATAPATH_VIS_NOTEBOOK = "../../datasets/youtube_VIS/new_2019/"
IMG_PATH_VIS_NOTEBOOK = DATAPATH_VIS_NOTEBOOK + "images/"
LAB_PATH_VIS_NOTEBOOK = DATAPATH_VIS_NOTEBOOK + "masks/"

FRAME_KEYWORD_YOUTUBE = ''
CLASS_ID_TYPE_YOUTUBE = 'd0fb600c73/'
ZFILL_NB_YOUTUBE = 5

## INTUITIVE DATASET
DATAPATH_INTUITIVE = "/storage/workspaces/artorg_aimi/ws_00000/fei/intuitive/"
TRAIN_PATH = DATAPATH_INTUITIVE + 'train/'
IMG_PATH_INTUITIVE = TRAIN_PATH + 'images/'
LAB_PATH_INTUITIVE = TRAIN_PATH + 'labels/'

DATAPATH_INTUITIVE_NOTEBOOK = "../../datasets/intuitive/train/"
IMG_PATH_INTUITIVE_NOTEBOOK = DATAPATH_INTUITIVE_NOTEBOOK + "images/"
LAB_PATH_INTUITIVE_NOTEBOOK = DATAPATH_INTUITIVE_NOTEBOOK + "labels/"

FRAME_KEYWORD_INTUITIVE = 'frame'
CLASS_ID_TYPE_INTUITIVE = 'seq_xx/'
ZFILL_NB_INTUITIVE = 3

## UAVID DATASET
DATAPATH_UAVID = "../dataset/UAVID2020/"
DATAPATH_UAVID = "../dataset/UAVID2020_small/"
TRAIN_PATH = DATAPATH_UAVID + 'train/'
IMG_PATH_UAVID = TRAIN_PATH + 'images/'
LAB_PATH_UAVID = TRAIN_PATH + 'labels/'

DATAPATH_UAVID_NOTEBOOK = "../../datasets/UAVID2020/train/"
DATAPATH_UAVID_NOTEBOOK = "../../datasets/UAVID2020_small/train/"
IMG_PATH_UAVID_NOTEBOOK = DATAPATH_UAVID_NOTEBOOK + "images/"
LAB_PATH_UAVID_NOTEBOOK = DATAPATH_UAVID_NOTEBOOK + "labels/"

FRAME_KEYWORD_UAVID = ''
CLASS_ID_TYPE_UAVID = 'seqxx/'
ZFILL_NB_UAVID = 6

## A2D2 DATASET
DATAPATH_A2D2 = "/storage/workspaces/artorg_aimi/ws_00000/fei/A2D2_small/"
DATAPATH_A2D2 = "/storage/workspaces/artorg_aimi/ws_00000/fei/A2D2_small_44pool/"
IMG_PATH_A2D2 = DATAPATH_A2D2 + "images/"
LAB_PATH_A2D2 = DATAPATH_A2D2 + "labels/"

DATAPATH_A2D2_NOTEBOOK = "../../datasets/A2D2_small/"
DATAPATH_A2D2_NOTEBOOK = "../../datasets/A2D2_small_44pool/"
IMG_PATH_A2D2_NOTEBOOK = DATAPATH_A2D2_NOTEBOOK + "images/"
LAB_PATH_A2D2_NOTEBOOK = DATAPATH_A2D2_NOTEBOOK + "labels/"

FRAME_KEYWORD_A2D2 = ''
CLASS_ID_TYPE_A2D2 = '20180807_145028/'
CLASS_ID_TYPE_A2D2 = '20180807_145028_0/'
ZFILL_NB_A2D2 = len('000000091')

## Cityscapes DATASET
DATAPATH_CITY = "/storage/workspaces/artorg_aimi/ws_00000/fei/cityscapes_small/"
TRAIN_PATH = DATAPATH_CITY + 'train/'
IMG_PATH_CITY = TRAIN_PATH + "images/"
LAB_PATH_CITY = TRAIN_PATH + "labels/"

DATAPATH_CITY_NOTEBOOK = "../../datasets/cityscapes_small/train/"
IMG_PATH_CITY_NOTEBOOK = DATAPATH_CITY_NOTEBOOK + "images/"
LAB_PATH_CITY_NOTEBOOK = DATAPATH_CITY_NOTEBOOK + "labels/"

FRAME_KEYWORD_CITY = ''
CLASS_ID_TYPE_CITY = 'aachen_________/'
ZFILL_NB_CITY = len('000000091') # to be updated

## PASCAL VOC DATASET
DATAPATH_VOC = "/storage/workspaces/artorg_aimi/ws_00000/fei/pascal_VOC_small/"
IMG_PATH_VOC = DATAPATH_VOC + "images/"
LAB_PATH_VOC = DATAPATH_VOC + "labels/"

DATAPATH_VOC_NOTEBOOK = "../../datasets/pascal_VOC_small/"
IMG_PATH_VOC_NOTEBOOK = DATAPATH_VOC_NOTEBOOK + "images/"
LAB_PATH_VOC_NOTEBOOK = DATAPATH_VOC_NOTEBOOK + "labels/"

FRAME_KEYWORD_VOC = ''
CLASS_ID_TYPE_VOC = ''
ZFILL_NB_VOC = len('000000091') # to be updated


## GENERAL SETTINGS
FILE_TYPE = '.png'
CLASS_ID_CUT = '/'
PRETRAINED_PATH = "../pretrained_models/deeplabV3/"
PRINT_PATH = "results/test_auris.txt"
CONTINUE_FOLDER = './checkpoints/'
CONTINUE_PATH = './checkpoints/continue.pth'
VAL_MODEL_FOLDER = "./val_models/"
SAVE_MASK_PATH = "./ML_preds/"
IGNORE_INDEX = 255
CLASS_0_DATASETS = ['uavid', 'a2d2', 'cityscapes']