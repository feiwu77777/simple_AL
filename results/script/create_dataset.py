from routes import (
    FILE_TYPE,
    PRINT_PATH,
    CLASS_ID_CUT
)
import routes
from torch.utils.data import DataLoader
import os
import numpy as np
import collections
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as Inter
import glob
from sklearn.model_selection import KFold
from training_utils import RandomScale
from config import img_size_auris, img_size_youtube, img_size_intuitive, img_size_uavid, img_size_a2d2, img_size_cityscapes, img_size_pascalVOC
from auris_datasplits import data_split_multi_class, data_split, data_split_multi_class_OSAL

from datasets.a2d2 import DataHandlerA2D2
from datasets.auris import DataHandler, DataHandlerMCAL
from datasets.intuitive import DataHandlerIntuitive, DataHandlerIntuitiveMCAL, DataHandlerIntuitiveMCALStage2
from datasets.uavid import DataHandlerUAVID
from datasets.youtube import DataHandlerYoutube
from datasets.cityscapes import DataHandlerCityscapes, DataHandlerCityscapesMCAL
from datasets.pascal_VOC import DataHandlerPascal, DataHandlerPascalMCAL
from PIL import Image
from config import config


def CV_split(train_videos, SEED, num_folds=5):
    local_seed, local_split = divmod(SEED, num_folds)
    np.random.seed(local_seed)
    np.random.shuffle(train_videos)
    fold_size = len(train_videos) // num_folds
    splits = []
    for i in range(num_folds):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size
        fold = train_videos[fold_start:fold_end]
        splits.append(fold)

    curr_train_videos = []
    for i, split in enumerate(splits):
        if i != local_split:
            for video_id in split:
                curr_train_videos.append(video_id)
    curr_val_videos = splits[local_split]

    return curr_train_videos, curr_val_videos


def CV_split_images(train_imgs, SEED, n_splits=6):
    local_seed, local_split = divmod(SEED, n_splits)
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=local_seed)
    all_trainVal_split = []
    for train_inds, val_inds in skf.split(train_imgs):
        all_trainVal_split.append((train_inds, val_inds))

    train_inds, val_inds = all_trainVal_split[local_split]
    train_imgs, val_imgs = np.array(train_imgs)[train_inds], np.array(train_imgs)[val_inds]

    return train_imgs, val_imgs


def chose_labeled(
    train_data,
    labeled,
    img_path=routes.IMG_PATH,
    frame_keyword=routes.FRAME_KEYWORD,
    class_id_type=routes.CLASS_ID_TYPE,
):
    labeled_train = collections.defaultdict(list)
    unlabeled_train = collections.defaultdict(list)
    all_train = collections.defaultdict(list)
    ind_keyword = len(img_path + class_id_type)
    for k, v in train_data.items():
        for i in range(len(v)):
            number = v[i][0][ind_keyword + len(frame_keyword) : -len(FILE_TYPE)]
            if k in labeled and number in labeled[k]:
                labeled_train[k].append(v[i])
            else:
                unlabeled_train[k].append(v[i])
            all_train[k].append(v[i])

    return labeled_train, unlabeled_train, all_train


def divide_data_split_auris_multi_class_kfold(img_path, lab_path, num_folds=5, num_train_val=15, data_path=None, SEED=0, category=None):
    data_splitter =  data_split_multi_class

    # for multi class segmentation, it is made so there is all the classes in the training, validation and test set
    imgs_path = sorted(os.listdir(img_path))
    labels_path = sorted(os.listdir(lab_path))

    all_imgs_train = []
    all_imgs_test = []
    for name in imgs_path:
        if name[0] == ".":
            continue

        if name in data_splitter and data_splitter[name] != "":
            img_files = []
            files = [f for f in os.listdir(img_path + name) if f[0] != "."]
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == ".":
                    continue
                img_files.append(img_path + name + "/" + file)

            if data_splitter[name] == "train":
                all_imgs_train.append(img_files)
            elif data_splitter[name] == "test":
                all_imgs_test.append(img_files)

    all_labels_train = []
    all_labels_test = []
    for name in labels_path:
        if name[0] == ".":
            continue
        if name in data_splitter and data_splitter[name] != "":
            label_files = []
            files = [f for f in os.listdir(lab_path + name) if f[0] != "."]
            files = sorted(files, key=lambda x: int(x[5:-4]))
            for file in files:
                if file[0] == ".":
                    continue
                label_files.append(lab_path + name + "/" + file)

            if data_splitter[name] == "train":
                all_labels_train.append(label_files)
            elif data_splitter[name] == "test":
                all_labels_test.append(label_files)

    assert len(all_labels_train) == len(all_imgs_train)
    assert len(all_labels_test) == len(all_imgs_test)

    indexes = np.arange(len(all_labels_train))

    video_with_class = [
        ["08-02", "15-01", "11-01"],  # 4 #"03-01",
        ["10-03", "10-01"],  # 5
        ["03-05", "15-02", "03-06"],  # 7
    ]

    all_video_with_class = sum(video_with_class, [])
    for i, group in enumerate(video_with_class):
        if SEED % 2 == 0:
            np.random.shuffle(group)

    train_video_with_class = []
    for group in video_with_class:
        train_video_with_class.append(group[:len(group) // 2])

    val_video_with_class = []
    for group in video_with_class:
        val_video_with_class.append(group[len(group) // 2:])

    train_val_data = {}
    ind = len(lab_path + routes.CLASS_ID_TYPE)
    for i, n in enumerate(indexes):
        L = []
        for j, frame in enumerate(all_imgs_train[n]):
            L.append((frame, all_labels_train[n][j]))
        train_val_data[
            frame[ind - len(routes.CLASS_ID_TYPE) : ind - len(CLASS_ID_CUT)]
        ] = np.array(L)

    test_data = {}
    for i, frames in enumerate(all_imgs_test):
        L = []
        for j, frame in enumerate(frames):
            L.append((frame, all_labels_test[i][j]))
        test_data[frame[ind - len(routes.CLASS_ID_TYPE) : ind - len(CLASS_ID_CUT)]] = np.array(
            L
        )

    train_videos = list(train_val_data.keys())
    L = []
    for video_id in train_videos:
        if video_id not in all_video_with_class:
            L.append(video_id)
    train_videos = L
    curr_train_videos, curr_val_videos = CV_split(train_videos, SEED, num_folds=num_folds)

    for group in train_video_with_class:
        curr_train_videos += group
    
    for group in val_video_with_class:
        curr_val_videos += group

    train_data = {}
    val_data = {}
    for video_id in curr_train_videos:
        train_data[video_id] = train_val_data[video_id]
    for video_id in curr_val_videos:
        val_data[video_id] = train_val_data[video_id]

    return train_data, val_data, test_data


def get_dataset_variables(config, notebook=False):
    if config["DATASET"] == "auris":
        img_path = routes.IMG_PATH if not notebook else routes.IMG_PATH_NOTEBOOK
        lab_path = routes.LAB_PATH if not notebook else routes.LAB_PATH_NOTEBOOK
        data_path = None
        class_id_type = routes.CLASS_ID_TYPE
        frame_keyword = routes.FRAME_KEYWORD
        img_size = img_size_auris
        zfill_nb = routes.ZFILL_NB
        num_train_val = routes.NUM_TRAIN_VAL

        if config["N_LABEL"] == 1:
            divide_data = divide_data_split_auris
        else:
            divide_data = divide_data_split_auris_multi_class_kfold

    if config["DATASET"] == "cityscapes":
        img_path = routes.IMG_PATH_CITY if not notebook else routes.IMG_PATH_CITY_NOTEBOOK
        lab_path = routes.LAB_PATH_CITY if not notebook else routes.LAB_PATH_CITY_NOTEBOOK
        data_path = None
        class_id_type = routes.CLASS_ID_TYPE_CITY
        frame_keyword = routes.FRAME_KEYWORD_CITY
        img_size = img_size_cityscapes
        zfill_nb = routes.ZFILL_NB_CITY
        num_train_val = None

        divide_data = None

    elif config["DATASET"] == "pascal_VOC":
        img_path = routes.IMG_PATH_VOC if not notebook else routes.IMG_PATH_VOC_NOTEBOOK
        lab_path = routes.LAB_PATH_VOC if not notebook else routes.LAB_PATH_VOC_NOTEBOOK
        data_path = routes.DATAPATH_VOC if not notebook else routes.DATAPATH_VOC_NOTEBOOK
        class_id_type = routes.CLASS_ID_TYPE_VOC
        frame_keyword = routes.FRAME_KEYWORD_VOC
        img_size = img_size_pascalVOC
        zfill_nb = routes.ZFILL_NB_VOC
        num_train_val = None

        divide_data = None

    return (
        img_path,
        lab_path,
        data_path,
        class_id_type,
        frame_keyword,
        img_size,
        divide_data,
        num_train_val,
        zfill_nb,
    )


def init_image_patch_labeled_set(train_imgs, patch_number, nb_labeled=50, frame_keyword=None):
    copy_train_imgs = list(train_imgs)
    all_patches = []
    for name in copy_train_imgs:
        class_name = name.split('/')[-2]
        number = name.split('/')[-1][len(frame_keyword): -len(FILE_TYPE)]
        frame_id = '/'.join([class_name, number])
        for patch_nb in range(patch_number):
            all_patches.append((frame_id, patch_nb))
    np.random.shuffle(all_patches)
    new_selected_patches = collections.defaultdict(list)
    for frame_id, patch_nb in all_patches[:nb_labeled * patch_number]:
        new_selected_patches[frame_id].append(patch_nb)

    new_labeled = collections.defaultdict(list)
    for frame_id, patch_nbs in new_selected_patches.items():
        class_name, number = frame_id.split('/')
        if class_name not in new_labeled or number not in new_labeled[class_name]:
            new_labeled[class_name].append(number)

    return new_labeled, new_selected_patches


def get_dataset_transforms(img_size, dataset="auris"):
    im_size, im_size2 = img_size["IMG_SIZE"], img_size["IMG_SIZE2"]
    if dataset in ['cityscapes', 'pascal_VOC'] and config['MODEL_ARCH'] == 'vit':
        im_size -= 1
        im_size2 -= 1
    elif dataset == 'auris' and config['MODEL_ARCH'] == 'vit':
        im_size += 4
        im_size2 += 4
    if dataset in ["parrots", "dog", "pets", "uavid", "a2d2", 'cityscapes', 'youtube_VIS', 'pascal_VOC']:
        train_imgTrans = T.Compose(
            [
                RandomScale(
                    (im_size, im_size2),
                    interpolation=Inter.BILINEAR,
                    scale_limit=img_size['SCALE_LIMIT'],
                ),
                T.RandomCrop((im_size, im_size2)),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        train_labelTrans = T.Compose(
            [
                RandomScale(
                    (im_size, im_size2),
                    interpolation=Inter.NEAREST,
                    scale_limit=img_size['SCALE_LIMIT'],
                ),
                T.RandomCrop((im_size, im_size2)),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        test_imgTrans = T.Compose(
            [
                T.Resize(
                    (im_size, im_size2),
                    interpolation=Inter.BILINEAR,
                ),
            ]
        )
        test_labelTrans = T.Compose(
            [
                T.Resize(
                    (im_size, im_size2),
                    interpolation=Inter.NEAREST,
                )
            ]
        )

    elif dataset in ["auris", "intuitive"]:
        train_imgTrans = T.Compose(
            [
                T.RandomResizedCrop(
                    im_size,
                    scale=(0.85, 1.0),
                    interpolation=Inter.BILINEAR,
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        train_labelTrans = T.Compose(
            [
                T.RandomResizedCrop(
                    im_size, scale=(0.85, 1.0), interpolation=Inter.NEAREST
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
        )
        test_imgTrans = T.Compose(
            [
                T.Resize(
                    (im_size, im_size),
                    interpolation=Inter.BILINEAR,
                ),
            ]
        )
        test_labelTrans = T.Compose(
            [
                T.Resize(
                    (im_size, im_size),
                    interpolation=Inter.NEAREST,
                )
            ]
        )

    return train_imgTrans, train_labelTrans, test_imgTrans, test_labelTrans


def get_datasets(
    config,
    curr_labeled,
    curr_selected_patches=None,
    train_data=None,
    val_data=None,
    test_data=None,
    TRAIN_SEQ=None,
    VAL_SEQ=None,
    TEST_SEQ=None,
    notebook=False,
    print_=True,
):
    (
        img_path,
        label_path,
        data_path,
        class_id_type,
        frame_keyword,
        img_size,
        _,
        _,
        _,
    ) = get_dataset_variables(config, notebook=notebook)

    (
        train_imgTrans,
        train_labelTrans,
        test_imgTrans,
        test_labelTrans,
    ) = get_dataset_transforms(img_size, config["DATASET"])

    unlabeled_dataset = None
    if config["DATASET"] == "auris":
        labeled_train, unlabeled_train, all_train = chose_labeled(
            train_data=train_data,
            labeled=curr_labeled,
            img_path=img_path,
            frame_keyword=frame_keyword,
            class_id_type=class_id_type,
        )
        train_dataHandler = DataHandler
        train_dataset = train_dataHandler(
            data_path=labeled_train,
            img_trans=train_imgTrans,
            label_trans=train_labelTrans,
            lab_path=label_path,
            multi_class=config["N_LABEL"] > 1,
            curr_selected_patches=curr_selected_patches,
            patch_number=config["PATCH_NUMBER"],
            patch_shape=config["PATCH_SHAPE"],
            dominant_labelling=config['DOMINANT_LABELLING'],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            drop_last=True,
            num_workers=config["NUM_WORKERS"],
        )
        if config["NUM_ROUND"] > 1:
            unlabeled_dataset = DataHandler(
                data_path=unlabeled_train,
                img_trans=test_imgTrans,
                label_trans=test_labelTrans,
                lab_path=label_path,
                labeled_part=labeled_train,
                multi_class=config["N_LABEL"] > 1,
                curr_selected_patches=curr_selected_patches,
                patch_number=config["PATCH_NUMBER"],
                patch_shape=config["PATCH_SHAPE"],
            )
        train_dataset_noAug = train_dataHandler(
            data_path=labeled_train,
            img_trans=test_imgTrans,
            label_trans=test_labelTrans,
            lab_path=label_path,
            multi_class=config["N_LABEL"] > 1,
            curr_selected_patches=curr_selected_patches,
            patch_number=config["PATCH_NUMBER"],
            patch_shape=config["PATCH_SHAPE"],
            dominant_labelling=config['DOMINANT_LABELLING'],
        )
        all_train_dataset = DataHandler(
            data_path=all_train,
            img_trans=test_imgTrans,
            label_trans=test_labelTrans,
            lab_path=label_path,
            multi_class=config["N_LABEL"] > 1,
            patch_number=config["PATCH_NUMBER"],
            patch_shape=config["PATCH_SHAPE"],
        )
        val_dataset = DataHandler(
            data_path=val_data,
            img_trans=test_imgTrans,
            label_trans=test_labelTrans,
            lab_path=label_path,
            multi_class=config["N_LABEL"] > 1,
            patch_number=config["PATCH_NUMBER"],
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        test_dataset = DataHandler(
            data_path=test_data,
            img_trans=test_imgTrans,
            label_trans=test_labelTrans,
            lab_path=label_path,
            multi_class=config["N_LABEL"] > 1,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )

    if config["DATASET"] == "cityscapes":
            train_dataHandler = DataHandlerCityscapes
            train_dataset = train_dataHandler(
                                        img_path=img_path,
                                        sequences=TRAIN_SEQ,
                                        transform=train_imgTrans,
                                        label_transform=train_labelTrans,
                                        filter=curr_labeled,
                                        curr_selected_patches=curr_selected_patches,
                                        patch_number=config["PATCH_NUMBER"],
                                        patch_shape=config["PATCH_SHAPE"],
                                        dominant_labelling=config['DOMINANT_LABELLING'],)
            train_dataset_noAug = train_dataHandler(
                                        img_path=img_path,
                                        sequences=TRAIN_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans,
                                        filter=curr_labeled,
                                        curr_selected_patches=curr_selected_patches,
                                        patch_number=config["PATCH_NUMBER"],
                                        patch_shape=config["PATCH_SHAPE"],
                                        dominant_labelling=config['DOMINANT_LABELLING'],)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config['BATCH_SIZE'],
                shuffle=True,
                drop_last=True,
                num_workers=config["NUM_WORKERS"])
            
            all_train_dataset = DataHandlerCityscapes(
                img_path=img_path,
                sequences=TRAIN_SEQ,
                transform=test_imgTrans,
                label_transform=test_labelTrans,
                patch_number=config['PATCH_NUMBER'],
                patch_shape=config['PATCH_SHAPE'],)

            if config['NUM_ROUND'] > 1:
                unlabeled_dataset = DataHandlerCityscapes(
                    img_path=img_path,
                    sequences=TRAIN_SEQ,
                    transform=test_imgTrans,
                    label_transform=test_labelTrans,
                    filter=curr_labeled,
                    labeled_set=False,
                    curr_selected_patches=curr_selected_patches,
                    patch_number=config['PATCH_NUMBER'],
                    patch_shape=config['PATCH_SHAPE'],)

            val_dataset = DataHandlerCityscapes(
                                        img_path=img_path,
                                        sequences=VAL_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans,
                                        patch_number=config['PATCH_NUMBER'],
                                        )
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['BATCH_SIZE'],
                                        shuffle=False,
                                        num_workers=config["NUM_WORKERS"])
            test_dataset = DataHandlerCityscapes(
                                        img_path=img_path.replace('train', 'val'),
                                        sequences=TEST_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=False,
                                            num_workers=config["NUM_WORKERS"])
    
    elif config["DATASET"] == "pascal_VOC":
            train_dataHandler = DataHandlerPascal
            train_dataset = train_dataHandler(
                                        img_path=img_path,
                                        sequences=TRAIN_SEQ,
                                        transform=train_imgTrans,
                                        label_transform=train_labelTrans,
                                        filter=curr_labeled,
                                        curr_selected_patches=curr_selected_patches,
                                        patch_number=config["PATCH_NUMBER"],
                                        patch_shape=config["PATCH_SHAPE"],
                                        dominant_labelling=config['DOMINANT_LABELLING'],
                                        merge_SP=config['MERGE_SP'],)
            train_dataset_noAug = train_dataHandler(
                                        img_path=img_path,
                                        sequences=TRAIN_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans,
                                        filter=curr_labeled,
                                        curr_selected_patches=curr_selected_patches,
                                        patch_number=config["PATCH_NUMBER"],
                                        patch_shape=config["PATCH_SHAPE"],
                                        dominant_labelling=config['DOMINANT_LABELLING'],
                                        merge_SP=config['MERGE_SP'],)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config['BATCH_SIZE'],
                shuffle=True,
                drop_last=True,
                num_workers=config["NUM_WORKERS"])
            
            all_train_dataset = DataHandlerPascal(
                img_path=img_path,
                sequences=TRAIN_SEQ,
                transform=test_imgTrans,
                label_transform=test_labelTrans,
                patch_number=config['PATCH_NUMBER'],
                patch_shape=config['PATCH_SHAPE'],)

            if config['NUM_ROUND'] > 1:
                unlabeled_dataset = DataHandlerPascal(
                    img_path=img_path,
                    sequences=TRAIN_SEQ,
                    transform=test_imgTrans,
                    label_transform=test_labelTrans,
                    filter=curr_labeled,
                    labeled_set=False,
                    curr_selected_patches=curr_selected_patches,
                    patch_number=config['PATCH_NUMBER'],
                    patch_shape=config['PATCH_SHAPE'],)

            val_dataset = DataHandlerPascal(
                                        img_path=img_path,
                                        sequences=VAL_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans,
                                        patch_number=config['PATCH_NUMBER'],
                                        )
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config['BATCH_SIZE'],
                                        shuffle=False,
                                        num_workers=config["NUM_WORKERS"])
            test_dataset = DataHandlerPascal(
                                        img_path=img_path.replace('train', 'val'),
                                        sequences=TEST_SEQ,
                                        transform=test_imgTrans,
                                        label_transform=test_labelTrans)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=False,
                                            num_workers=config["NUM_WORKERS"])

    unlabeled_length = len(unlabeled_dataset) if unlabeled_dataset is not None else 0
    train_dataset_print = f"number of labeled train frames: {len(train_dataset)}\n"
    unlabeled_dataset_print = f"number of unlabeled frames: {unlabeled_length}\n"
    all_train_dataset_print = f"number of train frames: {len(all_train_dataset)}\n"

    if config["PATCH_NUMBER"] is not None:
        patch_number = config["PATCH_NUMBER"] ** 2 if config["PATCH_SHAPE"] == "rectangle" else config["PATCH_NUMBER"]
        train_dataset_print = f"number of labeled train frames: {len(train_dataset)} - {train_dataset.nb_of_patches} patches - {train_dataset.nb_of_patches / patch_number} whole images\n"
        unlabeled_dataset_print = f"number of unlabeled frames: {unlabeled_length}"
        all_train_dataset_print = f"number of train frames: {len(all_train_dataset)} - total patches {len(all_train_dataset) * patch_number}\n"

        if unlabeled_length > 0:
            unlabeled_dataset_print += f" - {unlabeled_dataset.nb_of_patches} patches - {unlabeled_dataset.nb_of_patches / patch_number} whole images\n"
        else:
            unlabeled_dataset_print += "\n"
    
    if notebook and print_:
            print(
                f"sampling is: {config['SAMPLING']}\n"
                + train_dataset_print
                + all_train_dataset_print
                + f"number of val frames: {len(val_dataset)}\n"
                + f"number of test frames: {len(test_dataset)}\n"
                + unlabeled_dataset_print
                + f"first dataset sample: {train_dataset.data_pool[0][0][len(img_path): ]}\n"
            )
    
    elif os.path.exists(PRINT_PATH):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"sampling is: {config['SAMPLING']}\n"
                + train_dataset_print
                + all_train_dataset_print
                + f"number of val frames: {len(val_dataset)}\n"
                + f"number of test frames: {len(test_dataset)}\n"
                + unlabeled_dataset_print
                + f"first dataset sample: {train_dataset.data_pool[0][0][len(img_path): ]}\n"
            )
    
    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        all_train_dataset,
        train_dataset_noAug,
        unlabeled_dataset,
    )


def get_image_curr_labeled(config, SEED=0, notebook=False, print_=True):
    img_path, lab_path, data_path, class_id_type, frame_keyword, img_size, divide_data, num_train_val, _ = get_dataset_variables(config, notebook=notebook)

    curr_selected_patches = None
    if config["DATASET"] == "auris":
        TRAIN_SEQ, VAL_SEQ, TEST_SEQ = None, None, None
        train_data, val_data, test_data = divide_data(
            img_path, lab_path, num_train_val=num_train_val, data_path=data_path, SEED=SEED, category=config["DATASET"]
        )
        if notebook and print_:
            print(
                f"train video: {sorted(train_data.keys())}, {len(train_data)}\n"
                + f"val video: {sorted(val_data.keys())}, {len(val_data)}\n"
                + f"test video: {sorted(test_data.keys())}, {len(test_data)}\n"
            )
        elif os.path.exists(PRINT_PATH):
            with open(PRINT_PATH, "a") as f:
                f.write(
                    f"train video: {sorted(train_data.keys())}, {len(train_data)}\n"
                    + f"val video: {sorted(val_data.keys())}, {len(val_data)}\n"
                    + f"test video: {sorted(test_data.keys())}, {len(test_data)}\n"
                )

        TRAIN_IMG = []
        for video_id, frame_paths in train_data.items():
            for imgs, _ in frame_paths:
                TRAIN_IMG.append(imgs)
        nb_labeled = config['INIT_NUM_IMAGE']
        if config['NUM_ROUND'] == 1:
            nb_labeled = len(TRAIN_IMG)
        curr_labeled, curr_selected_patches = init_image_patch_labeled_set(TRAIN_IMG, config['PATCH_NUMBER'], nb_labeled=nb_labeled, frame_keyword=frame_keyword)
        config['PATIENCE_ITER'] = len(TRAIN_IMG)

    if config["DATASET"] == "cityscapes":
        train_data, val_data, test_data = None, None, None
        TRAIN_SEQ = []
        video_list = sorted(os.listdir(img_path))
        for video_id in video_list:
            if video_id[0] == '.':
                continue
            frame_paths = glob.glob(os.path.join(img_path, video_id, '*.png'))
            frame_paths = sorted(frame_paths)
            TRAIN_SEQ += frame_paths

        TEST_SEQ = []
        video_list = sorted(os.listdir(img_path.replace('train', 'val')))
        for video_id in video_list:
            if video_id[0] == '.':
                continue
            frame_paths = glob.glob(os.path.join(img_path.replace('train', 'val'), video_id, '*.png'))
            frame_paths = sorted(frame_paths)
            TEST_SEQ += frame_paths
        
        TRAIN_SEQ, VAL_SEQ = CV_split_images(TRAIN_SEQ, SEED=SEED)
        if notebook and print_:
            print(
                f"TRAIN IMGS: {len(TRAIN_SEQ)}\n"
                + f"VAL IMGS: {len(VAL_SEQ)}\n"
                + f"TEST IMGS: {len(TEST_SEQ)}\n"
            )
        elif os.path.exists(PRINT_PATH):
            with open(PRINT_PATH, "a") as f:
                f.write(
                    f"TRAIN IMGS: {len(TRAIN_SEQ)}\n"
                    + f"VAL IMGS: {len(VAL_SEQ)}\n"
                    + f"TEST IMGS: {len(TEST_SEQ)}\n"
                )
        nb_labeled = config['INIT_NUM_IMAGE']
        if config['NUM_ROUND'] == 1:
            nb_labeled = len(TRAIN_SEQ)
        curr_labeled, curr_selected_patches = init_image_patch_labeled_set(TRAIN_SEQ, config['PATCH_NUMBER'], nb_labeled=nb_labeled, frame_keyword=frame_keyword)
        config['PATIENCE_ITER'] = len(TRAIN_SEQ)

    elif config["DATASET"] == "pascal_VOC":
        train_data, val_data, test_data = None, None, None

        # Open the file in read mode ('r')
        with open(data_path + 'train.txt', 'r') as file:
            # Read the contents of the file
            TRAIN_SEQ = file.read()
        TRAIN_SEQ = TRAIN_SEQ.split('\n')
        TRAIN_SEQ = [img_path + t + '.jpg' for t in TRAIN_SEQ if len(t) > 0]

        # Open the file in read mode ('r')
        with open(data_path + 'val.txt', 'r') as file:
            # Read the contents of the file
            TEST_SEQ = file.read()
        TEST_SEQ = TEST_SEQ.split('\n')
        TEST_SEQ = [img_path + t + '.jpg' for t in TEST_SEQ if len(t) > 0]
        
        TRAIN_SEQ, VAL_SEQ = CV_split_images(TRAIN_SEQ, SEED=SEED)
        if notebook and print_:
            print(
                f"TRAIN IMGS: {len(TRAIN_SEQ)}\n"
                + f"VAL IMGS: {len(VAL_SEQ)}\n"
                + f"TEST IMGS: {len(TEST_SEQ)}\n"
            )
        elif os.path.exists(PRINT_PATH):
            with open(PRINT_PATH, "a") as f:
                f.write(
                    f"TRAIN IMGS: {len(TRAIN_SEQ)}\n"
                    + f"VAL IMGS: {len(VAL_SEQ)}\n"
                    + f"TEST IMGS: {len(TEST_SEQ)}\n"
                )

        nb_labeled = config['INIT_NUM_IMAGE']
        if config['NUM_ROUND'] == 1:
            nb_labeled = len(TRAIN_SEQ)
        curr_labeled, curr_selected_patches = init_image_patch_labeled_set(TRAIN_SEQ, config['PATCH_NUMBER'], nb_labeled=nb_labeled, frame_keyword=frame_keyword)
        config['PATIENCE_ITER'] = len(TRAIN_SEQ)

    return curr_labeled, curr_selected_patches, train_data, val_data, test_data, TRAIN_SEQ, VAL_SEQ, TEST_SEQ