from torch.utils.data import DataLoader
import torch
import collections
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import torch.nn.functional as F
from scipy import stats
import pdb
from copy import deepcopy
from PIL import Image
import shutil
import routes
from scipy.spatial import distance
#from utils import is_adjacent, find_elbow


from routes import FILE_TYPE, CLASS_ID_CUT, PRINT_PATH, IGNORE_INDEX
#from utils import (
#    embedding_similarity,
#    euc_distance,
#    resnet_embedding,
#    simCLR_embedding,
#    center_diff,
#    average_center,
#    simCLR_projection_embedding,
#    BYOL_embedding,
#    representativeness,
#    balance_classes,
#    balance_pixels,
#    get_clusters_x_class,
#    get_clusters_x_class_video,
#    get_videos_x_class,
#    get_patch_clusters_x_class,
#    SpatialPurity,
#)

from create_dataset import get_dataset_variables
from config import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5

(
    IMG_PATH,
    LAB_PATH,
    _,
    CLASS_ID_TYPE,
    FRAME_KEYWORD,
    _,
    _,
    _,
    ZFILL_NB,
) = get_dataset_variables(config)


def random_query(unlabeled_dataset, num_query, train_dataset=None, patch_number=None, patch_shape=None):
    if patch_number is None:
        all_names = list(unlabeled_dataset.data_pool[:, 0])
        np.random.shuffle(all_names)

        new_labeled = collections.defaultdict(list)
        ind_keyword = len(IMG_PATH + CLASS_ID_TYPE)
        for n in all_names[:num_query]:
            class_name = n[
                ind_keyword - len(CLASS_ID_TYPE) : ind_keyword - len(CLASS_ID_CUT)
            ]
            number = n[ind_keyword + len(FRAME_KEYWORD) : -len(FILE_TYPE)]
            new_labeled[class_name].append(number)

        return new_labeled, None
    else:
        patch_number = (
            patch_number if patch_shape == "superpixel" else patch_number**2
        )
        # Use patch_number to select specific patches from the images
        all_names = list(unlabeled_dataset.data_pool[:, 0])
        curr_selected_patches = unlabeled_dataset.curr_selected_patches

        all_patches = []
        for name in all_names:
            class_name = name.split("/")[-2]
            number = name.split("/")[-1][len(FRAME_KEYWORD) : -len(FILE_TYPE)]
            frame_id = "/".join([class_name, number])
            for patch_nb in range(patch_number):
                if (
                    frame_id not in curr_selected_patches
                    or patch_nb not in curr_selected_patches[frame_id]
                ):
                    all_patches.append((0, "/".join([class_name, FRAME_KEYWORD + number]), patch_nb))

        np.random.shuffle(all_patches)
        if config['MULTI_CLASS_LABELING']:
            new_labeled, new_selected_patches = count_multiClassLabeling_click(
                num_query * patch_number, all_patches, train_dataset
            )
        else:
            new_selected_patches = collections.defaultdict(list)
            for _, frame_id, patch_nb in all_patches[: num_query * patch_number]:
                if 'frame' in frame_id:
                    frame_id = frame_id.split('/')[-2] + '/' + frame_id.split('/')[-1][len('frame'):]
                new_selected_patches[frame_id].append(patch_nb)

            new_labeled = collections.defaultdict(list)
            for frame_id, patch_nbs in new_selected_patches.items():
                class_name, number = frame_id.split("/")
                if class_name not in new_labeled or number not in new_labeled[class_name]:
                    new_labeled[class_name].append(number)

        return new_labeled, new_selected_patches


def density_query(train_dataset, unlabeled_dataset, num_query):
    unlabeled_frames = list(unlabeled_dataset.data_pool[:, 0])
    labeled_frames = list(train_dataset.data_pool[:, 0])

    ind_keyword = len(IMG_PATH + CLASS_ID_TYPE)
    labeled_distances = collections.defaultdict(list)
    for frame in labeled_frames:
        classId = frame[ind_keyword - len(CLASS_ID_TYPE) : ind_keyword - 1]
        frameNb = int(frame[ind_keyword + len(FRAME_KEYWORD) : -len(FILE_TYPE)])
        labeled_distances[classId].append(frameNb)

    unlabeled_distances = collections.defaultdict(list)
    for frame in unlabeled_frames:
        classId = frame[ind_keyword - len(CLASS_ID_TYPE) : ind_keyword - 1]
        frameNb = int(frame[ind_keyword + len(FRAME_KEYWORD) : -len(FILE_TYPE)])
        unlabeled_distances[classId].append(frameNb)
        if classId not in labeled_distances:
            labeled_distances[classId] = []

    new_labeled = collections.defaultdict(list)
    n_count = 0
    keys = sorted(
        [(k, v) for k, v in labeled_distances.items()], key=lambda x: len(x[1])
    )
    keys = [k for k, v in keys]
    while n_count < num_query:
        for k in keys:
            v = labeled_distances[k]
            D = collections.defaultdict(list)
            unlab_video = unlabeled_distances[k]

            if len(v) == 0:
                mid = int(len(unlab_video) / 2)
                new_labeled[k].append(str(unlab_video[mid]).zfill(ZFILL_NB))
                labeled_distances[k].append(unlab_video[mid])
                n_count += 1

            else:
                for dist in unlab_video:
                    for labeled_dist in v:
                        D[dist].append(abs(dist - labeled_dist))

                max_k = None
                max_v = 0
                for k2, v2 in D.items():
                    D[k2] = min(v2)
                    if D[k2] > max_v:
                        max_v = D[k2]
                        max_k = k2
                if max_k is not None:
                    new_labeled[k].append(str(max_k).zfill(ZFILL_NB))
                    labeled_distances[k].append(max_k)
                    n_count += 1

            if n_count == num_query:
                return new_labeled

    return new_labeled


def entropy_query(ML_entropy, num_query):
    ML_entropy = sorted(
        [(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1], reverse=True
    )
    count = 0
    new_labeled = collections.defaultdict(list)
    for selected, score in ML_entropy:
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break
    return new_labeled


def entropy_patch_query(ML_entropy, num_query, patch_number=None, patch_shape=None):
    patch_number = patch_number if patch_shape == "superpixel" else patch_number**2
    patch_ML_entropy = []
    for k, v in ML_entropy.items():
        for patch_nb, score in enumerate(v):
            patch_ML_entropy.append((f"{k}/{patch_nb}", score))

    patch_ML_entropy = sorted(patch_ML_entropy, key=lambda x: x[1], reverse=True)
    count = 0
    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for selected, score in patch_ML_entropy:
        video_id, frame_nb, patch_nb = selected.split("/")
        frame_nb = frame_nb[len(FRAME_KEYWORD) :]

        frame_id = "/".join([video_id, frame_nb])
        new_labeled[video_id].append(frame_nb)
        new_selected_patches[frame_id].append(int(patch_nb))

        count += 1
        if count == num_query * patch_number:
            break
    return new_labeled, new_selected_patches


def class_entropy_query(ML_class_entropy, train_dataset, num_query, SEED, n_round):
    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, img_dict = balance_classes(
        img_per_class, num_query, n_class=config["N_LABEL"]
    )

    new_labeled = collections.defaultdict(list)
    img_should_be = {}
    for c in added_imgs.keys():
        ML_entropy = sorted(
            [(k, value[c]) for k, value in ML_class_entropy.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        ind = 0
        count = 0
        while True:
            selected, score = ML_entropy[ind]
            class_name = selected[: len(CLASS_ID_TYPE) - 1]
            number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]

            if class_name in new_labeled and number in new_labeled[class_name]:
                ind += 1
                continue

            new_labeled[class_name].append(number)
            img_should_be[selected] = (c, score)
            count += 1
            ind += 1
            if count == added_imgs[c]:
                break

    true_added_imgs = {}
    true_added_pixels = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    total_acc = 0
    img_query_acc = {}

    for i, (k, v) in enumerate(new_labeled.items()):
        for j, v2 in enumerate(v):
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = train_dataset.open_path(img_path, lab_path)

            ## global acc
            wanted_class, corresponding_ent = img_should_be[f"{k}/{FRAME_KEYWORD}{v2}"]
            unique_classes = np.unique(y)
            if wanted_class in unique_classes:
                total_acc += 1
                img_query_acc[wanted_class] = img_query_acc.get(wanted_class, 0) + 1

            for c in range(start, config["N_LABEL"]):
                if torch.sum(y == c) > 0:
                    true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                    true_added_pixels[c] = (
                        true_added_pixels.get(c, 0) + torch.sum(y == c).item()
                    )

    for c in range(start, config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)
    with open(f"results/img_query_acc_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(img_query_acc, f)
    with open(f"results/added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(added_imgs, f)
    return new_labeled


def class_entropy_ML_pred_patch_query(
    model,
    all_train_dataset,
    train_dataset,
    num_query,
    SEED,
    n_round,
    patch_number,
    patch_shape,
    smooth=1e-7,
):
    ML_class_entropy = {}
    model.eval()
    dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    curr_selected_patches = train_dataset.curr_selected_patches
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img)  # shape = (batch_size, n_class, h, w)

            all_proba = torch.softmax(
                pred, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            log_proba = torch.log(all_proba + smooth)
            global_entropy = -(all_proba * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            global_class_entropy = all_proba * global_entropy.reshape(
                (pred.shape[0], 1, pred.shape[2], pred.shape[3])
            )  # shape = (batch_size, n_class, h, w)
            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]
                
                if patch_shape == "superpixel":
                    superpixel_lab = all_train_dataset.load_superpixel(name, transform=True)

                    all_patch_class_entropy = []
                    for patch_id in range(patch_number):
                        patch_class_entropy = global_class_entropy[
                            i, :, superpixel_lab == patch_id
                        ]  # shape = (n_class, n_pixel)
                        if patch_class_entropy.shape[1] == 0 or patch_id in selected_patches:
                            patch_class_entropy = np.zeros(config['N_LABEL'])
                        else:
                            patch_class_entropy = np.max(
                                patch_class_entropy.numpy(), axis=1
                            )  # shape = (n_class) CHANGE MAX TO MEAN?
                        all_patch_class_entropy.append(patch_class_entropy)

                    ML_class_entropy[name] = np.stack(
                        all_patch_class_entropy, axis=0
                    )  # shape = (patch_number**2, n_class)

    patch_number = patch_number if patch_shape == "superpixel" else patch_number**2
    # if patch_shape == 'superpixel':
    #     patch_number = len(next(iter(train_dataset.curr_selected_patches.values())))

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0

    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        if patch_shape == "superpixel":
            superpixel_lab = train_dataset.load_superpixel(n, transform=True)
        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]
        for patch_id in patch_ids:
            if patch_shape == "rectangle":
                i, j = divmod(patch_id, patch_number)
                patch_size_x = y.shape[0] // patch_number
                patch_size_y = y.shape[1] // patch_number
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == patch_number - 1:
                    end_x = y.shape[0]
                end_y = start_y + patch_size_y
                if j == patch_number - 1:
                    end_y = y.shape[1]

                patch_y = y[start_x:end_x, start_y:end_y]
            elif patch_shape == "superpixel":
                patch_y = y[superpixel_lab == patch_id]

            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, img_dict = balance_classes(
        img_per_class, num_query * patch_number, n_class=config["N_LABEL"]
    )

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    img_should_be = {}
    for c in added_imgs.keys():
        ML_entropy = {}
        for k, v in ML_class_entropy.items():
            for i in range(len(v)):  # v is of shape (patch_number, n_class)
                ML_entropy[k + f"/{i}"] = v[i][c]

        ML_entropy = sorted(
            [(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1], reverse=True
        )
        ind = 0
        count = 0
        while True:
            selected, score = ML_entropy[ind]
            class_name, number, patch_id = selected.split("/")
            number = number[len(FRAME_KEYWORD) :]
            frame_id = "/".join([class_name, number])

            if (
                frame_id in new_selected_patches
                and int(patch_id) in new_selected_patches[frame_id]
            ):
                ind += 1
                continue
            if class_name not in new_labeled or number not in new_labeled[class_name]:
                new_labeled[class_name].append(number)
            new_selected_patches[frame_id].append(int(patch_id))

            img_should_be[selected] = (c, score)
            count += 1
            ind += 1
            if count == added_imgs[c]:
                break

    return new_labeled, new_selected_patches


def class_entropy_video_query(
    ML_class_entropy, curr_labeled, train_dataset, all_train_dataset, num_query
):
    # get labeled frames per video
    labeled_frames_per_video = {}
    unlabeled_frames_per_video = {}
    for path, label_path in all_train_dataset.data_pool:
        video_id, frame_nb = path[len(IMG_PATH) : -4].split("/")
        if (
            video_id in curr_labeled
            and frame_nb[len(FRAME_KEYWORD) :] in curr_labeled[video_id]
        ):
            frames = labeled_frames_per_video.get(video_id, [])
            frames.append(frame_nb)
            labeled_frames_per_video[video_id] = frames
        else:
            frames = unlabeled_frames_per_video.get(video_id, [])
            frames.append(frame_nb)
            unlabeled_frames_per_video[video_id] = frames

    # get distribution of classes in the dataset
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    new_labeled = collections.defaultdict(list)
    total_labeled = 0

    while total_labeled <= num_query:
        img_per_class = {}
        start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
        for video_id, frames in labeled_frames_per_video.items():
            for number in frames:
                img_path = f"{IMG_PATH}{video_id}/{number}{img_file_type}"
                lab_path = f"{LAB_PATH}{video_id}/{number}{lab_file_type}"
                x, y = train_dataset.open_path(img_path, lab_path, name=None)

                for c in range(start, config["N_LABEL"]):
                    if torch.sum(y == c) > 0:
                        img_per_class[c] = img_per_class.get(c, 0) + 1

        added_imgs, _ = balance_classes(
            img_per_class,
            num_query - total_labeled,
            n_class=config["N_LABEL"],
            start=start,
        )

        videos_x_class = get_videos_x_class(
            unlabeled_frames_per_video, ML_class_entropy, added_imgs, start=start
        )

        max_cluster_entropy_per_video = {}
        for video_id, class_ in videos_x_class.items():
            frames = unlabeled_frames_per_video[video_id]
            max_class_entropy = -float("inf")
            nb = None
            for frame in frames:
                if ML_class_entropy[f"{video_id}/{frame}"][class_] > max_class_entropy:
                    max_class_entropy = ML_class_entropy[f"{video_id}/{frame}"][class_]
                    nb = frame
            max_cluster_entropy_per_video[video_id] = (max_class_entropy, nb)

        sorted_videos = []
        for video_id in videos_x_class.keys():
            nb_of_labeled_frames = len(labeled_frames_per_video.get(video_id, []))
            entropy, frame_nb = max_cluster_entropy_per_video[video_id]
            sorted_videos.append((video_id, frame_nb, nb_of_labeled_frames, -entropy))
        sorted_videos = sorted(sorted_videos, key=lambda x: (x[2], x[3]))

        for video_id, frame_nb, nb_of_labeled_frames, _ in sorted_videos:
            labeled_list = labeled_frames_per_video.get(video_id, [])
            labeled_list.append(frame_nb)
            labeled_frames_per_video[video_id] = labeled_list

            unlabeled_list = unlabeled_frames_per_video.get(video_id, [])
            # remove nb from unlabeled list
            L = []
            for n in unlabeled_list:
                if n != frame_nb:
                    L.append(n)
            unlabeled_frames_per_video[video_id] = L

            frame_nb = frame_nb[len(FRAME_KEYWORD) :]
            new_labeled[video_id].append(frame_nb)

            total_labeled += 1
            if total_labeled >= num_query:
                return new_labeled


def class_entropy_patch_query(
    ML_class_entropy,
    train_dataset,
    num_query,
    SEED,
    n_round,
    patch_number=None,
    patch_shape=None,
):

    patch_number = patch_number if patch_shape == "superpixel" else patch_number**2
    # if patch_shape == 'superpixel':
    #     patch_number = len(next(iter(train_dataset.curr_selected_patches.values())))

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    # start = 0 # use this for start=0 experiments

    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        if patch_shape == "superpixel":
            superpixel_lab = train_dataset.load_superpixel(n, transform=True)
        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]
        for patch_id in patch_ids:
            if patch_shape == "rectangle":
                i, j = divmod(patch_id, patch_number)
                patch_size_x = y.shape[0] // patch_number
                patch_size_y = y.shape[1] // patch_number
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == patch_number - 1:
                    end_x = y.shape[0]
                end_y = start_y + patch_size_y
                if j == patch_number - 1:
                    end_y = y.shape[1]

                patch_y = y[start_x:end_x, start_y:end_y]
            elif patch_shape == "superpixel":
                patch_y = y[superpixel_lab == patch_id]

            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, img_dict = balance_classes(
        img_per_class, num_query * patch_number, n_class=config["N_LABEL"]
    )  # , start=start) # uncomment start=start for start=1 experiments

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    img_should_be = {}
    for c in added_imgs.keys():
        ML_entropy = {}
        for k, v in ML_class_entropy.items():
            for i in range(len(v)):  # v is of shape (patch_number, n_class)
                ML_entropy[k + f"/{i}"] = v[i][c]

        ML_entropy = sorted(
            [(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1], reverse=True
        )
        ind = 0
        count = 0
        while True:
            selected, score = ML_entropy[ind]
            class_name, number, patch_id = selected.split("/")
            number = number[len(FRAME_KEYWORD) :]
            frame_id = "/".join([class_name, number])

            if (
                frame_id in new_selected_patches
                and int(patch_id) in new_selected_patches[frame_id]
            ):
                ind += 1
                continue
            if class_name not in new_labeled or number not in new_labeled[class_name]:
                new_labeled[class_name].append(number)
            new_selected_patches[frame_id].append(int(patch_id))

            img_should_be[selected] = (c, score)
            count += 1
            ind += 1
            if count == added_imgs[c]:
                break

    ## evaluate accuracy of class to be added
    # true_added_imgs = {}
    # true_added_pixels = {}
    # start = 1 if config['DATASET'] not in routes.CLASS_0_DATASETS else 0
    # img_file_type = '.png'
    # lab_file_type = '.png'
    # if config['DATASET'] == 'pets':
    #     img_file_type = '.jpg'
    #     lab_file_type = ''

    # total_acc = 0
    # img_query_acc = {}

    # for i, (k, v) in enumerate(new_labeled.items()):
    #     for j, v2 in enumerate(v):
    #         img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
    #         lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
    #         x, y = train_dataset.open_path(img_path, lab_path)
    #         patch_ids = new_selected_patches[f"{k}/{v2}"]

    #         if patch_shape == 'superpixel':
    #             superpixel_lab = train_dataset.load_superpixel(f"{k}/{FRAME_KEYWORD}{v2}")

    #         for patch_id in patch_ids:
    #             if patch_shape == 'rectangle':
    #                 i, j = divmod(patch_id, patch_number)
    #                 patch_size_x = y.shape[0] // patch_number
    #                 patch_size_y = y.shape[1] // patch_number
    #                 start_x = i * patch_size_x
    #                 start_y = j * patch_size_y

    #                 end_x = start_x + patch_size_x
    #                 if i == patch_number - 1:
    #                     end_x = y.shape[0]
    #                 end_y = start_y + patch_size_y
    #                 if j == patch_number - 1:
    #                     end_y = y.shape[1]

    #                 patch_y = y[start_x:end_x, start_y:end_y]
    #             elif patch_shape == 'superpixel':
    #                 patch_y = y[superpixel_lab == patch_id]

    #             ## global acc
    #             wanted_class, corresponding_ent = img_should_be[f"{k}/{FRAME_KEYWORD}{v2}/{patch_id}"]
    #             unique_classes = np.unique(patch_y)
    #             if wanted_class in unique_classes:
    #                 total_acc += 1
    #                 img_query_acc[wanted_class] = img_query_acc.get(wanted_class, 0) + 1

    #             for c in range(start, config['N_LABEL']):
    #                 if torch.sum(patch_y == c) > 0:
    #                     true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
    #                     true_added_pixels[c] = true_added_pixels.get(c, 0) + torch.sum(patch_y == c).item()

    # for c in range(start, config['N_LABEL']):
    #     with open(PRINT_PATH, 'a') as f:
    #         f.write(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}\n')
    #     # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')
    # with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(true_added_imgs, f)
    # with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(true_added_pixels, f)
    # with open(f"results/img_query_acc_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(img_query_acc, f)
    # with open(f"results/added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(added_imgs, f)
    return new_labeled, new_selected_patches


def class_entropy_patch_multiClass_query(
    ML_class_entropy,
    train_dataset,
    num_query,
    SEED,
    n_round,
    patch_number=None,
    patch_shape=None,
):

    patch_number = patch_number if patch_shape == "superpixel" else patch_number**2
    # if patch_shape == 'superpixel':
    #     patch_number = len(next(iter(train_dataset.curr_selected_patches.values())))

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    # start = 0 # use this for start=0 experiments
    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        y = data["labels"]  # shape = (patch_number, num_classes)
        frame_id = data['names']
        n = '/'.join([frame_id.split('/')[-2], frame_id.split('/')[-1][len(FRAME_KEYWORD):]])
        patch_ids = curr_selected_patches[n]
        for c in range(start, config["N_LABEL"]):
            img_per_class[c] = img_per_class.get(c, 0) + torch.sum(y[patch_ids, c]).item()

    selection_count = num_query * patch_number
    added_imgs, img_dict = balance_classes(
        img_per_class, selection_count, n_class=config["N_LABEL"]
    )  # , start=start) # uncomment start=start for start=1 experiments

    selected_samples_per_class = collections.defaultdict(list)
    already_selected = collections.defaultdict(list)
    for c in added_imgs.keys():
        ML_entropy = {}
        for k, v in ML_class_entropy.items():
            for i in range(len(v)):  # v is of shape (patch_number, n_class)
                ML_entropy[k + f"/{i}"] = v[i][c]

        ML_entropy = sorted(
            [(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1], reverse=True
        )
        ind = 0
        count = 0
        while count < added_imgs[c] and ind < len(ML_entropy):
            selected, score = ML_entropy[ind]
            class_name, number, patch_id = selected.split("/")
            # number = number[len(FRAME_KEYWORD) :]

            frame_id = "/".join([class_name, number])
            patch_id = int(patch_id)

            ind += 1
            if frame_id in already_selected and patch_id in already_selected[frame_id]:
                continue

            selected_samples_per_class[c].append((score, frame_id, patch_id))
            already_selected[frame_id].append(int(patch_id))
            count += 1

    # selected_samples = sorted(scores, reverse=True)
    assert (
        len(np.concatenate(list(selected_samples_per_class.values())))
        == selection_count
    )
    # organize it like due to multi class label cost
    selected_samples = []
    ind = 0
    while len(selected_samples) < selection_count:
        for c, samples in selected_samples_per_class.items():
            if ind < len(samples):
                selected_samples.append(samples[ind])
        ind += 1

    new_labeled, new_selected_patches = count_multiClassLabeling_click(
        selection_count, selected_samples, train_dataset
    )
    return new_labeled, new_selected_patches


def count_multiClassLabeling_click(selection_count, selected_samples, train_dataset):
    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)

    selected_count = 0
    """ Active Selection """
    for (
        _,
        frame_id,
        patch_id,
    ) in (
        selected_samples
    ):  # sorted tuples: (score, file_path *join of three paths, suppix_id)
        """jump out the loop when exceeding max_selection_count"""
        # fair counting
        patch_id = int(patch_id)
        num_cls = train_dataset.multi_hot_cls[frame_id][patch_id].sum()
        selected_count += num_cls

        class_name, number = frame_id.split("/")
        number = number[len(FRAME_KEYWORD) :]
        frame_id = "/".join([class_name, number])

        new_selected_patches[frame_id].append(patch_id)
        if class_name not in new_labeled or number not in new_labeled[class_name]:
            new_labeled[class_name].append(number)

        if selected_count > selection_count:
            break

    return new_labeled, new_selected_patches


def CBAL_query(
    train_dataset,
    unlabeled_dataset,
    model,
    num_query,
    n_round,
    SEED,
    patch_number,
    smooth=1e-7,
):
    # https://github.com/Javadzb/Class-Balanced-AL/blob/main/query_strategies/entropy_sampling.py
    ML_entropy = {}
    ML_proba = {}
    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img)  # shape = (batch_size, n_class, h, w)

            all_proba = torch.softmax(
                pred, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            log_proba = torch.log(all_proba + smooth)
            global_entropy = (all_proba * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                all_patch_proba = []
                all_patch_entropy = []
                for patch_id in np.unique(superpixel_lab):
                    patch_entropy = global_entropy[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_entropy = -torch.mean(patch_entropy).item()
                    all_patch_entropy.append(patch_entropy)

                    patch_proba = all_proba[
                        i, :, superpixel_lab == patch_id
                    ]  # shape = (n_class, n_pixel)
                    patch_proba = (
                        torch.mean(patch_proba, dim=1).cpu().numpy()
                    )  # shape = (n_class)
                    all_patch_proba.append(patch_proba)

                ML_entropy[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)
                ML_proba[name] = np.stack(
                    all_patch_proba, axis=0
                )  # shape = (patch_number**2, n_class)

    torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    torch.save(ML_proba, f"results/ML_proba_SEED={SEED}_round={n_round}.pt")

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0

    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        superpixel_lab = train_dataset.load_superpixel(n, transform=True)
        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]
        for patch_id in patch_ids:
            patch_y = y[superpixel_lab == patch_id]

            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, _ = balance_classes(
        img_per_class, num_query * patch_number, n_class=config["N_LABEL"]
    )  # , start=start)
    added_imgs = np.array([added_imgs.get(k, 0) for k in range(config["N_LABEL"])])
    added_imgs = added_imgs / np.sum(added_imgs)

    final_scores = []
    lambda_ = 2
    for k, v in ML_entropy.items():
        for i in range(len(v)):
            proba = ML_proba[k][i]  # [start:]
            score = v[i] - lambda_ * np.sum(np.abs(proba - added_imgs))
            final_scores.append((k + f"/{i}", score))

    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for k, score in final_scores[: num_query * patch_number]:
        video_id, frame_nb, patch_id = k.split("/")
        frame_nb = frame_nb[len(FRAME_KEYWORD) :]
        frame_id = "/".join([video_id, frame_nb])
        if video_id not in new_labeled or frame_nb not in new_labeled[video_id]:
            new_labeled[video_id].append(frame_nb)
        new_selected_patches[frame_id].append(int(patch_id))

    return new_labeled, new_selected_patches


def CBAL_v2_query(
    train_dataset,
    unlabeled_dataset,
    model,
    num_query,
    n_round,
    SEED,
    patch_number,
    smooth=1e-7,
):
    # https://github.com/Javadzb/Class-Balanced-AL/blob/main/query_strategies/entropy_sampling.py
    ML_entropy = {}
    ML_proba = {}
    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    curr_selected_patches = train_dataset.curr_selected_patches

    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img)  # shape = (batch_size, n_class, h, w)

            all_proba = torch.softmax(
                pred, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            log_proba = torch.log(all_proba + smooth)
            global_entropy = (all_proba * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]

                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                all_patch_proba = []
                all_patch_entropy = []
                for patch_id in range(patch_number):
                    patch_entropy = global_entropy[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    if len(patch_entropy) == 0 or patch_id in selected_patches:
                        patch_entropy = 0
                    else:
                        patch_entropy = -torch.mean(patch_entropy).item()                    
                    all_patch_entropy.append(patch_entropy)

                    patch_proba = all_proba[
                        i, :, superpixel_lab == patch_id
                    ]  # shape = (n_class, n_pixel)
                    if patch_proba.shape[1] == 0 or patch_id in selected_patches:
                        patch_proba = torch.zeros(config["N_LABEL"])
                    else:
                        patch_proba = (
                            torch.mean(patch_proba, dim=1).cpu().numpy()
                        )  # shape = (n_class)
                    all_patch_proba.append(patch_proba)

                ML_entropy[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)
                ML_proba[name] = np.stack(
                    all_patch_proba, axis=0
                )  # shape = (patch_number**2, n_class)

    torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    torch.save(ML_proba, f"results/ML_proba_SEED={SEED}_round={n_round}.pt")

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0

    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        superpixel_lab = train_dataset.load_superpixel(n, transform=True)
        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]
        for patch_id in patch_ids:
            patch_y = y[superpixel_lab == patch_id]

            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, _ = balance_classes(
        img_per_class, num_query * patch_number, n_class=config["N_LABEL"]
    )  # , start=start)
    added_imgs = np.array([added_imgs.get(k, 0) for k in range(config["N_LABEL"])])
    added_imgs = added_imgs / np.sum(added_imgs)

    final_scores = []
    lambda_ = 2
    for k, v in ML_entropy.items():
        for i in range(len(v)):
            proba = ML_proba[k][i]  # [start:]
            if v[i] == 0 and np.sum(proba) == 0:
                score = float("-inf")
            else:
                score = v[i] - lambda_ * np.sum(np.abs(proba - added_imgs))
            final_scores.append((k + f"/{i}", score))

    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for k, score in final_scores[: num_query * patch_number]:
        video_id, frame_nb, patch_id = k.split("/")
        frame_nb = frame_nb[len(FRAME_KEYWORD) :]
        frame_id = "/".join([video_id, frame_nb])
        if video_id not in new_labeled or frame_nb not in new_labeled[video_id]:
            new_labeled[video_id].append(frame_nb)
        new_selected_patches[frame_id].append(int(patch_id))

    return new_labeled, new_selected_patches


def CBAL_multiClass_query(
    train_dataset,
    unlabeled_dataset,
    model,
    num_query,
    n_round,
    SEED,
    patch_number,
    smooth=1e-7,
):
    # https://github.com/Javadzb/Class-Balanced-AL/blob/main/query_strategies/entropy_sampling.py
    ML_entropy = {}
    ML_proba = {}
    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img)  # shape = (batch_size, n_class, h, w)

            all_proba = torch.softmax(
                pred, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            log_proba = torch.log(all_proba + smooth)
            global_entropy = (all_proba * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                all_patch_proba = []
                all_patch_entropy = []
                for patch_id in np.unique(superpixel_lab):
                    patch_entropy = global_entropy[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_entropy = -torch.mean(patch_entropy).item()
                    all_patch_entropy.append(patch_entropy)

                    patch_proba = all_proba[
                        i, :, superpixel_lab == patch_id
                    ]  # shape = (n_class, n_pixel)
                    patch_proba = (
                        torch.mean(patch_proba, dim=1).cpu().numpy()
                    )  # shape = (n_class)
                    all_patch_proba.append(patch_proba)

                ML_entropy[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)
                ML_proba[name] = np.stack(
                    all_patch_proba, axis=0
                )  # shape = (patch_number**2, n_class)

    torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    torch.save(ML_proba, f"results/ML_proba_SEED={SEED}_round={n_round}.pt")

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0

    curr_selected_patches = train_dataset.curr_selected_patches
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        y = data["labels"]  # shape = (patch_number, num_classes)
        frame_id = data['names']        
        n = '/'.join([frame_id.split('/')[-2], frame_id.split('/')[-1][len(FRAME_KEYWORD):]])
        patch_ids = curr_selected_patches[n]
        for c in range(start, config["N_LABEL"]):
            img_per_class[c] = img_per_class.get(c, 0) + torch.sum(y[patch_ids, c]).item()

    selection_count = num_query * patch_number
    added_imgs, _ = balance_classes(
        img_per_class, selection_count, n_class=config["N_LABEL"]
    )  # , start=start)
    added_imgs = np.array([added_imgs.get(k, 0) for k in range(config["N_LABEL"])])
    added_imgs = added_imgs / np.sum(added_imgs)

    final_scores = []
    lambda_ = 2
    for k, v in ML_entropy.items():
        for i in range(len(v)):
            proba = ML_proba[k][i]  # [start:]
            score = v[i] - lambda_ * np.sum(np.abs(proba - added_imgs))
            final_scores.append((score, k, i))

    final_scores = sorted(final_scores, key=lambda x: x[0], reverse=True)

    new_labeled, new_selected_patches = count_multiClassLabeling_click(
        selection_count, final_scores, train_dataset
    )

    return new_labeled, new_selected_patches


def density_entropy_query(ML_entropy, curr_labeled, num_query):
    ML_entropy_per_video = collections.defaultdict(list)
    for k, v in ML_entropy.items():
        classID = k[: len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        ML_entropy_per_video[classID].append((k, v))

    for k, v in ML_entropy_per_video.items():
        ML_entropy_per_video[k] = sorted(v, key=lambda x: x[1], reverse=True)

    count = 0
    selected = {}
    ind_per_class = {}
    for k, v in ML_entropy_per_video.items():
        selected[k] = np.zeros(len(v), dtype=bool)
        ind_per_class[k] = 0
    new_labeled = collections.defaultdict(list)

    # prioritize videos with lowest entropy score and videos that has less selected frames
    for k in ML_entropy_per_video.keys():
        if k not in curr_labeled:
            curr_labeled[k] = []

    lengths = np.unique([len(v) for v in curr_labeled.values()])
    lengths = sorted(lengths)
    if len(lengths) > 2:
        l1 = lengths[-2]
        l2 = lengths[-1]
    else:
        l1 = min(lengths)
        l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k in ML_entropy_per_video.keys():
        if len(curr_labeled[k]) == l1:
            viv1.append(ML_entropy_per_video[k][0])
        elif l2 != l1 and len(curr_labeled[k]) == l2:
            viv2.append(ML_entropy_per_video[k][0])
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])
    viv = [path for path, score in viv]

    while True:
        for img_name in viv:
            class_name = img_name[: len(CLASS_ID_TYPE) - 1]
            frames = ML_entropy_per_video[class_name]
            stay = True
            thresh = THRESH
            while stay:
                ind = ind_per_class[class_name]
                img_name, score = frames[ind]
                number = img_name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
                # new_path = IMG_PATH + img_name + ".jpg"
                # if addSample(new_labeled, curr_labeled, unlabeled_dataset,
                #              new_path, class_name, IMG_PATH, thresh):
                new_labeled[class_name].append(number)
                count += 1
                selected[class_name][ind] = True
                stay = False

                ind_per_class[class_name] = ind + 1
                if ind_per_class[class_name] == len(frames):
                    ind_per_class[class_name] = 0
                    thresh += 0.05
                    with open(PRINT_PATH, "a") as f:
                        f.write(f"=== video {class_name} thresh is now: {thresh}\n")
                if count == num_query:
                    return new_labeled


def density_classEntropyV2_query(ML_class_entropy, curr_labeled, num_query):
    ML_class_entropy_per_video = collections.defaultdict(list)
    for k, v in ML_class_entropy.items():
        classID = k[: len(CLASS_ID_TYPE) - 1]
        number = k[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        ML_class_entropy_per_video[classID].append((k, np.mean(v)))

    for k, v in ML_class_entropy_per_video.items():
        ML_class_entropy_per_video[k] = sorted(v, key=lambda x: x[1], reverse=True)

    count = 0
    selected = {}
    ind_per_class = {}
    for k, v in ML_class_entropy_per_video.items():
        selected[k] = np.zeros(len(v), dtype=bool)
        ind_per_class[k] = 0
    new_labeled = collections.defaultdict(list)

    # prioritize videos with lowest entropy score and videos that has less selected frames
    for k in ML_class_entropy_per_video.keys():
        if k not in curr_labeled:
            curr_labeled[k] = []

    lengths = np.unique([len(v) for v in curr_labeled.values()])
    lengths = sorted(lengths)
    if len(lengths) > 2:
        l1 = lengths[-2]
        l2 = lengths[-1]
    else:
        l1 = min(lengths)
        l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k in ML_class_entropy_per_video.keys():
        if len(curr_labeled[k]) == l1:
            viv1.append(ML_class_entropy_per_video[k][0])
        elif l2 != l1 and len(curr_labeled[k]) == l2:
            viv2.append(ML_class_entropy_per_video[k][0])
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])
    viv = [path for path, score in viv]

    while True:
        for img_name in viv:
            class_name = img_name[: len(CLASS_ID_TYPE) - 1]
            frames = ML_class_entropy_per_video[class_name]
            stay = True
            thresh = THRESH
            while stay:
                ind = ind_per_class[class_name]
                img_name, score = frames[ind]
                number = img_name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
                # new_path = IMG_PATH + img_name + ".jpg"
                # if addSample(new_labeled, curr_labeled, unlabeled_dataset,
                #              new_path, class_name, IMG_PATH, thresh):
                new_labeled[class_name].append(number)
                count += 1
                selected[class_name][ind] = True
                stay = False

                ind_per_class[class_name] = ind + 1
                if ind_per_class[class_name] == len(frames):
                    ind_per_class[class_name] = 0
                    thresh += 0.05
                    with open(PRINT_PATH, "a") as f:
                        f.write(f"=== video {class_name} thresh is now: {thresh}\n")
                if count == num_query:
                    return new_labeled


def coreset_entropy_query(
    ML_entropy,
    all_train_dataset,
    train_dataset,
    copy_model,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="",
    use_sphere=False,
):
    count = 0
    new_labeled = collections.defaultdict(list)

    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif embedding_method == "model":
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        copy_model.backbone.layer4.register_forward_hook(get_activation("features"))

        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        embeddings = {}

        with torch.no_grad():
            for img, label, names in all_train_dataloader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred = copy_model(img)

                features = pooling_layer(activation["features"]).squeeze()
                for i, name in enumerate(names):
                    embeddings[name] = features[i].cpu()
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if use_sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    labeled_frames = [
        path[len(IMG_PATH) : -len(FILE_TYPE)]
        for path, label_path in train_dataset.data_pool
    ]
    labeled_embeddings = {}
    unlabeled_embeddings = {}
    for k, v in embeddings.items():
        if k in labeled_frames:
            labeled_embeddings[k] = embeddings[k]
        else:
            unlabeled_embeddings[k] = embeddings[k]

    while count < num_query:
        unlabeled_keys = list(unlabeled_embeddings.keys())
        unlabeled_ML_entropy = torch.tensor([ML_entropy[k] for k in unlabeled_keys])
        distances = torch.cdist(
            torch.stack(list(unlabeled_embeddings.values())),
            torch.stack(list(labeled_embeddings.values())),
        )  # shape = (len(unlabeled_embeddings), len(labeled_embeddings))
        # torch.save(distances, f'./results/distances_count={count}.pth')
        distances = torch.min(
            distances, axis=1
        ).values  # shape = (len(unlabeled_embeddings), )
        distances = (
            distances * unlabeled_ML_entropy
        )  # shape = (len(unlabeled_embeddings), )

        ind = torch.argmax(distances).item()
        selected = unlabeled_keys[ind]

        class_ID = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]

        assert selected not in labeled_embeddings
        labeled_embeddings[selected] = unlabeled_embeddings[selected]
        del unlabeled_embeddings[selected]

        new_labeled[class_ID].append(number)
        count += 1

    return new_labeled


def coreset_query(
    all_train_dataset,
    train_dataset,
    copy_model,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="",
    use_sphere=False,
):
    count = 0
    new_labeled = collections.defaultdict(list)

    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif embedding_method == "model":
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        copy_model.backbone.layer4.register_forward_hook(get_activation("features"))

        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        embeddings = {}

        with torch.no_grad():
            for img, label, names in all_train_dataloader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred = copy_model(img)

                features = pooling_layer(activation["features"]).squeeze()
                for i, name in enumerate(names):
                    embeddings[name] = features[i].cpu()
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if use_sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    labeled_frames = [
        path[len(IMG_PATH) : -len(FILE_TYPE)]
        for path, label_path in train_dataset.data_pool
    ]
    labeled_embeddings = {}
    unlabeled_embeddings = {}
    for k, v in embeddings.items():
        if k in labeled_frames:
            labeled_embeddings[k] = embeddings[k]
        else:
            unlabeled_embeddings[k] = embeddings[k]

    while count < num_query:
        distances = torch.cdist(
            torch.stack(list(unlabeled_embeddings.values())),
            torch.stack(list(labeled_embeddings.values())),
        )  # shape = (len(unlabeled_embeddings), len(labeled_embeddings))
        # torch.save(distances, f'./results/distances_count={count}.pth')
        distances = torch.min(
            distances, axis=1
        ).values  # shape = (len(unlabeled_embeddings), )
        # distances = torch.mean(distances, axis=1)
        # torch.save(distances, f'./results/averaged_distances_count={count}.pth')

        ind = torch.argmax(distances).item()
        selected = list(unlabeled_embeddings.keys())[ind]

        class_ID = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]

        assert selected not in labeled_embeddings
        labeled_embeddings[selected] = unlabeled_embeddings[selected]
        del unlabeled_embeddings[selected]

        new_labeled[class_ID].append(number)
        count += 1

    return new_labeled


def GT_query(num_query, n_round, SEED):
    ##### sample with MLvsGT and no similarity #########
    count = 0
    new_labeled = collections.defaultdict(list)
    with open(f"results/MLvsGT_scores_SEED={SEED}_round={n_round}.json", "r") as f:
        MLvsGT_scores = json.load(f)
    candidates = sorted([(k, v) for k, v in MLvsGT_scores.items()], key=lambda x: x[1])
    for selected, score in candidates:
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break
    return new_labeled


def GTxSim_query(train_dataset, unlabeled_dataset, num_query, n_round, SEED):
    count = 0
    new_labeled = collections.defaultdict(list)
    ##### sample with MLvsGT and similarity #########
    with open(f"results/MLvsGT_scores_SEED={SEED}_round={n_round}.json", "r") as f:
        MLvsGT_scores = json.load(f)
    labeled_frames = list(train_dataset.data_pool[:, 0])
    unlabeled_frames = list(unlabeled_dataset.data_pool[:, 0])

    while count < num_query:
        distances, names = embedding_similarity(labeled_frames, unlabeled_frames)
        distances = (distances - torch.min(distances)) / (
            torch.max(distances) - torch.min(distances)
        )

        dist_dict = {}
        for i, n in enumerate(names):
            dist_dict[n] = distances[i].item()

        with open(f"results/embedding_dist_SEED={SEED}_round={n_round}.json", "w") as f:
            json.dump(dist_dict, f)

        selected = None
        max_score = 0
        for k, v in dist_dict.items():
            score = 2 * (v * (1 - MLvsGT_scores[k])) / (v + (1 - MLvsGT_scores[k]))
            if score > max_score:
                max_score = score
                selected = k

        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]

        new_path = IMG_PATH + selected + FILE_TYPE
        labeled_frames.append(new_path)
        prev_length = len(unlabeled_frames)
        unlabeled_frames.remove(new_path)
        assert len(unlabeled_frames) == prev_length - 1

        new_labeled[class_name].append(number)
        count += 1

    return new_labeled


def suggestive_annotation_query(
    ML_entropy,
    unlabeled_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method,
    weight_path,
    sphere=False,
    smooth=1e-12,
):
    # written myself I think.
    ### get ML entropy step 1 ###
    # ML_entropy = {}
    # copy_model.eval()
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    # if embedding_method == "model":
    #     activation = {}

    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activation[name] = output.detach()

    #         return hook

    #     if config["DROPOUT_MODEL"]:
    #         copy_model.aspp.register_forward_hook(get_activation("features"))
    #     else:
    #         copy_model.classifier[0].register_forward_hook(get_activation("features"))
    #     # copy_model.backbone.layer4.register_forward_hook(get_activation("features"))

    #     pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
    #     embeddings = {}

    # with torch.no_grad():
    #     for img, label, names in unlabeled_dataloader:
    #         img, label = img.to(DEVICE), label.to(DEVICE)
    #         pred = copy_model(img)
    #         proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
    #         proba_neg = 1 - proba_pos
    #         all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

    #         log_proba = torch.log(all_proba + smooth)
    #         entropy = (all_proba * log_proba).sum(-1)

    #         if embedding_method == "model":
    #             features = pooling_layer(activation["features"]).squeeze()
    #         for i, name in enumerate(names):
    #             ML_entropy[name] = torch.mean(entropy[i]).item()
    #             if embedding_method == "model":
    #                 embeddings[name] = features[i].cpu()

    # # didnt put the "-" sign in entropy so we just sort from smaller
    # # to bigger instead of bigger to smaller with the "-" sign
    # with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(ML_entropy, f)
    # ML_entropy = sorted([(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1])

    ML_entropy = sorted(
        [(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1], reverse=True
    )
    ML_entropy = ML_entropy[
        : num_query * 2
    ]  # [(filename, entropy), ...] num_query*2 because in the paper they do 2x (k=8 and K=16)

    ### get image embeddings
    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(unlabeled_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(unlabeled_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            unlabeled_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif embedding_method == "model":
        pass
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    # embedding = {filename: embedding, ...} and embedding.shape = (feature size, )
    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # Initialize Sa and its representativeness
    Sa = {}
    max_rep = 0

    # Step 2
    for _ in range(num_query):
        best_img = None
        for k, v in ML_entropy:
            if k not in Sa:
                rep = representativeness(
                    list(Sa.values()) + [embeddings[k]], list(embeddings.values())
                )
                if rep > max_rep:
                    max_rep = rep
                    best_img = k
        Sa[best_img] = embeddings[best_img]

    new_labeled = collections.defaultdict(list)
    for selected, emb in Sa.items():
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    return new_labeled


def suggestive_annotation_patch_query(
    ML_entropy,
    unlabeled_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method,
    weight_path,
    sphere=False,
    smooth=1e-12,
    patch_shape=None,
    patch_number=None,
):
    # written myself I think.
    ### get ML entropy step 1 ###
    # ML_entropy = {}
    # copy_model.eval()
    if patch_number is not None:
        unlabeled_dataset.return_patches = True
        patch_number = (
            patch_number if patch_shape == "superpixel" else patch_number**2
        )

    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    ML_patch_entropy = {}
    for k, v in ML_entropy.items():
        for nb, score in enumerate(v):
            ML_patch_entropy[f"{k}/{nb}"] = score

    ML_patch_entropy = sorted(
        [(k, v) for k, v in ML_patch_entropy.items()], key=lambda x: x[1], reverse=True
    )
    ML_patch_entropy = ML_patch_entropy[
        : num_query * patch_number * 2
    ]  # [(filename, entropy), ...] num_query*2 because in the paper they do 2x (k=8 and K=16)
    ML_entropy = ML_patch_entropy

    ### get image embeddings
    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(unlabeled_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(
            unlabeled_dataloader, weight_path=weight_path, patch_number=patch_number
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            unlabeled_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif embedding_method == "model":
        pass
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    # embedding = {filename: embedding, ...} and embedding.shape = (feature size, )
    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)
    if patch_number is not None:
        unlabeled_dataset.return_patches = False

    ## cluster the patch embeddings
    patch_embeddings = {}
    for patch_id, embedding in enumerate(embeddings):
        for k, v in embedding.items():
            patch_embeddings[k + f"/{patch_id}"] = v
    # Initialize Sa and its representativeness
    Sa = {}
    max_rep = 0

    # Step 2
    for _ in range(num_query * patch_number):
        best_img = None
        for k, v in ML_entropy:
            if k not in Sa:
                rep = representativeness(
                    list(Sa.values()) + [patch_embeddings[k]],
                    list(patch_embeddings.values()),
                )
                if rep > max_rep:
                    max_rep = rep
                    best_img = k
        Sa[best_img] = patch_embeddings[best_img]

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for selected, emb in Sa.items():
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        class_name, number, patch_id = selected.split("/")
        number = number[len(FRAME_KEYWORD) :]

        frame_id = f"{class_name}/{number}"
        new_selected_patches[frame_id].append(int(patch_id))

        if class_name not in new_labeled and number not in new_labeled[class_name]:
            new_labeled[class_name].append(number)

    return new_labeled, new_selected_patches


def BADGE_query(copy_model, unlabeled_dataset, num_query):
    # https://github.com/JordanAsh/badge/tree/master
    # kmeans ++ initialization
    def init_centers(X, K):
        embs = torch.Tensor(X)
        ind = torch.argmax(torch.norm(embs, 2, 1)).item()
        embs = embs.cuda()
        mu = [embs[ind]]
        indsAll = [ind]
        centInds = [0.0] * len(embs)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
            else:
                newD = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
                for i in range(len(embs)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if sum(D2) == 0.0:
                pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2**2) / sum(D2**2)
            customDist = stats.rv_discrete(
                name="custm", values=(np.arange(len(D2)), Ddist)
            )
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll:
                ind = customDist.rvs(size=1)[0]
            mu.append(embs[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def get_grad_embedding(self, X, Y, model=[]):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        copy_model.backbone.layer4.register_forward_hook(get_activation("features"))
        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))

        embeddings = {}
        with torch.no_grad():
            for img, label, names in all_train_dataloader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred = copy_model(img)

                features = pooling_layer(activation["features"]).squeeze()
                for i, name in enumerate(names):
                    embeddings[name] = features[i].cpu()

        embDim = model.get_embedding_dim()
        model.eval()
        nLab = 2
        embedding = np.zeros(
            [len(Y), embDim * nLab]
        )  # shape = (batch_size, emb_dim * n_classes)

        with torch.no_grad():
            for x, y, names in loader_te:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)  # shape = (batch_size, n_classes, 224, 224)
                emb = pooling_layer(
                    activation["features"]
                ).squeeze()  # shape = (batch_size, emb_dim)
                batchProbs = (
                    torch.sigmoid(out).cpu().numpy()
                )  # shape = (batch_size, n_classes, 224, 224)
                maxInds = (batchProbs > 0.5).astype(
                    int
                )  # shape = (batch_size, n_classes, 224, 224)

                for j, name in enumerate(names):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[name][embDim * c : embDim * (c + 1)] = deepcopy(
                                emb[j]
                            ) * (1 - batchProbs[j])
                        else:
                            embedding[name][embDim * c : embDim * (c + 1)] = deepcopy(
                                emb[j]
                            ) * (-1 * batchProbs[j])

            return torch.Tensor(embedding)

    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    gradEmbedding = get_grad_embedding(
        self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]
    ).numpy()
    chosen = init_centers(gradEmbedding, num_query)

    count = 0
    new_labeled = collections.defaultdict(list)
    for selected, score in ML_entropy:
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break

    return new_labeled


def MC_dropout_query(
    copy_model, unlabeled_dataset, num_query, SEED=0, n_round=0, smooth=1e-12
):
    # inspired from viewAL paper and REDAL paper
    MC_STEPS = 20

    ML_entropy = {}
    copy_model.eval()

    # Turn on Dropout
    for m in copy_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)

            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            for step in range(1, MC_STEPS):
                pred = copy_model(img)
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba += torch.stack((proba_pos, proba_neg), axis=-1)

            all_proba /= MC_STEPS

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                ML_entropy[name] = torch.mean(entropy[i]).item()

    copy_model.eval()
    # with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(ML_entropy, f)

    # didnt put the "-" sign in entropy so we just sort from smaller
    # to bigger instead of bigger to smaller with the "-" sign
    ML_entropy = sorted([(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1])

    count = 0
    new_labeled = collections.defaultdict(list)
    for selected, score in ML_entropy:
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break

    return new_labeled


def BALD_query(
    copy_model, unlabeled_dataset, num_query, SEED=0, n_round=0, smooth=1e-12
):
    MC_STEPS = 20

    ML_entropy = {}
    copy_model.eval()

    # Turn on Dropout
    for m in copy_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)

            all_MC_steps_proba = []
            for step in range(MC_STEPS):
                pred = copy_model(img)
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba = torch.stack(
                    (proba_pos, proba_neg), axis=-1
                )  # (batch_size, 220, 220, 2)
                all_MC_steps_proba.append(all_proba)

            all_MC_steps_proba = torch.stack(
                all_MC_steps_proba, axis=0
            )  # (MC_STEPS, batch_size, 220, 220, 2)

            pb = all_MC_steps_proba.mean(0)
            entropy1 = (-pb * torch.log(pb + smooth)).sum(-1)  # (batch_size, 220, 220)
            entropy2 = (
                (-all_MC_steps_proba * torch.log(all_MC_steps_proba + smooth))
                .sum(-1)
                .mean(0)
            )  # (batch_size, 220, 220)
            U = entropy2 - entropy1  # (batch_size, 220, 220)

            for i, name in enumerate(names):
                ML_entropy[name] = torch.mean(U[i]).item()

    # code from Alpha mix paper and built upon MC dropout code
    # def query(self, n):
    # 	idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    # 	probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.args.n_drop)
    # 	pb = probs.mean(0)
    # 	entropy1 = (-pb*torch.log(pb)).sum(1)
    # 	entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    # 	U = entropy2 - entropy1

    # 	probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
    # 	selected = U.sort()[1][:n]
    # 	return idxs_unlabeled[selected], embeddings, pb.max(dim=1)[1], probs, selected, None

    copy_model.eval()
    # with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
    #     json.dump(ML_entropy, f)

    # didnt put the "-" sign in entropy so we just sort from smaller
    # to bigger instead of bigger to smaller with the "-" sign
    ML_entropy = sorted([(k, v) for k, v in ML_entropy.items()], key=lambda x: x[1])

    count = 0
    new_labeled = collections.defaultdict(list)
    for selected, score in ML_entropy:
        class_name = selected[: len(CLASS_ID_TYPE) - 1]
        number = selected[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)
        count += 1
        if count == num_query:
            break

    return new_labeled


def batchBALD(copy_model, unlabeled_dataset, num_query, n_round, SEED, smooth=1e-12):
    MC_STEPS = 20

    ML_entropy = {}
    copy_model.eval()

    # Turn on Dropout
    for m in copy_model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)

            all_MC_steps_proba = []
            for step in range(MC_STEPS):
                pred = copy_model(img)
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba = torch.stack(
                    (proba_pos, proba_neg), axis=-1
                )  # (batch_size, 220, 220, 2)
                all_MC_steps_proba.append(all_proba)

            all_MC_steps_proba = torch.stack(
                all_MC_steps_proba, axis=0
            )  # (MC_STEPS, batch_size, 220, 220, 2)

            pb = all_MC_steps_proba.mean(0)
            entropy1 = (-pb * torch.log(pb + smooth)).sum(-1)  # (batch_size, 220, 220)
            entropy2 = (
                (-all_MC_steps_proba * torch.log(all_MC_steps_proba + smooth))
                .sum(-1)
                .mean(0)
            )  # (batch_size, 220, 220)
            U = entropy2 - entropy1  # (batch_size, 220, 220)

            for i, name in enumerate(names):
                ML_entropy[name] = torch.mean(U[i]).item()

            N, K, C = log_probs_N_K_C.shape

            batch_size = min(num_query, N)

            candidate_indices = []
            candidate_scores = []

            conditional_entropies_N = entropy2

            batch_joint_entropy = joint_entropy.DynamicJointEntropy(
                num_samples, batch_size - 1, K, C
            )
            ## num_samples = 1000, batch_size = 10, K = 20, C = 2

            # We always keep these on the CPU.
            scores_N = torch.empty(
                N, dtype=torch.double, pin_memory=torch.cuda.is_available()
            )

            for i in range(batch_size):
                if i > 0:
                    latest_index = candidate_indices[-1]
                    batch_joint_entropy.add_variables(
                        log_probs_N_K_C[latest_index : latest_index + 1]
                    )

                shared_conditinal_entropies = conditional_entropies_N[
                    candidate_indices
                ].sum()  # 0 if i == 0

                batch_joint_entropy.compute_batch(
                    log_probs_N_K_C, output_entropies_B=scores_N
                )

                scores_N -= conditional_entropies_N + shared_conditinal_entropies
                scores_N[candidate_indices] = -float("inf")

                candidate_score, candidate_index = scores_N.max(dim=0)

                candidate_indices.append(candidate_index.item())
                candidate_scores.append(candidate_score.item())


def VAAL_query(train_dataset, unlabeled_dataset, num_query, n_round, num_iter):
    latent_dim = 32
    vae = VAE(latent_dim)
    discriminator = Discriminator(latent_dim)
    vae = vae.to(config["DEVICE"])
    discriminator = discriminator.to(config["DEVICE"])

    if n_round > 0:
        vae.load_state_dict(torch.load(f"checkpoints/vae_weight.pth"))
        discriminator.load_state_dict(
            torch.load(f"checkpoints/discriminator_weight.pth")
        )

    with open(PRINT_PATH, "a") as f:
        f.write(f"train VAE for: {num_iter} iterations\n")
    for iter in range(num_iter // 2):
        vae, discriminator = train_VAAL(
            train_dataset,
            unlabeled_dataset,
            vae,
            discriminator,
            batch_size=config["BATCH_SIZE"],
            print_=iter == num_iter - 1,
        )

    torch.save(vae.state_dict(), f"checkpoints/vae_weight.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_weight.pth")

    vae.eval()
    discriminator.eval()

    unlabed_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    all_preds = {}
    with torch.no_grad():
        for images, _, names in unlabed_dataloader:
            images = images.to(DEVICE)
            _, _, mu, _ = vae(images)
            preds = discriminator(mu)

            for i, name in enumerate(names):
                all_preds[name] = preds[i].item()

    ## need to multiply by -1 to be able to use torch.topk
    # all_preds *= -1
    ## select the points which the discriminator things are the most likely to be unlabeled
    # _, querry_indices = torch.topk(all_preds, int(self.budget))
    all_preds = sorted([(k, v) for k, v in all_preds.items()], key=lambda x: x[1])

    new_labeled = collections.defaultdict(list)
    for n in range(num_query):
        name, score = all_preds[n]
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    return new_labeled


def k_means_entropy_query(
    copy_model,
    unlabeled_dataset,
    num_query,
    n_cluster,
    n_round,
    SEED,
    smooth=1e-7,
    embedding_method="resnet",
    weight_path=None,
):
    ML_entropy = {}
    copy_model.eval()
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir("results/embeddings/"):
        os.mkdir("results/embeddings/")

    with torch.no_grad():
        for img, label, names in unlabeled_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)
            proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
            proba_neg = 1 - proba_pos
            all_proba = torch.stack((proba_pos, proba_neg), axis=-1)

            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(-1)

            for i, name in enumerate(names):
                entropy_value = torch.mean(entropy[i]).item()
                ML_entropy[name] = entropy_value
    with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(ML_entropy, f)

    # {classID/frame000: array([]), ...}
    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(unlabeled_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        selected = {str(i): [] for i in range(n_cluster)}
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(unlabeled_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        selected = {str(i): [] for i in range(n_cluster)}
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")
        with open(
            f"results/embeddings/selected_SEED={SEED}_round={n_round - 1}.json", "r"
        ) as f:
            selected = json.load(f)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    labels = kmeans.labels_

    cluster_scores = collections.defaultdict(list)
    for i, k in enumerate(embeddings.keys()):
        if k in ML_entropy:
            label = labels[i]
            cluster_scores[label].append((k, ML_entropy[k]))
    for k, v in cluster_scores.items():
        cluster_scores[k] = sorted(v, key=lambda x: x[1])

    lengths = np.unique([len(v) for v in selected.values()])
    if len(lengths) > 2:
        l1 = sorted(lengths)[-2]
    else:
        l1 = min(lengths)
    l2 = max(lengths)
    viv1 = []
    viv2 = []
    for k, v in cluster_scores.items():
        selected_frames = selected[str(k)]
        if len(selected_frames) == l1:
            for name, score in v:
                if name not in selected_frames:
                    viv1.append((name, score, k))
                    break
        elif l2 != l1 and len(selected_frames) == l2:
            for name, score in v:
                if name not in selected_frames:
                    viv2.append((name, score, k))
                    break
    viv = sorted(viv1, key=lambda x: x[1]) + sorted(viv2, key=lambda x: x[1])

    # {classID: ['000', ...], ...}
    new_labeled = collections.defaultdict(list)
    for i in range(num_query):
        name, score, cluster = viv[i]
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

        selected[str(cluster)].append(name)

    with open(
        f"results/embeddings/selected_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(selected, f)
    return new_labeled


def COWAL_center_query(
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/auris_seg_simCLR/checkpoint.pth",
    sphere=False,
    use_kmedian=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir("results/embeddings/"):
        os.mkdir("results/embeddings/")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "CARL":
        embeddings = CARL_embedding(all_train_dataset, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # kmeans = CustomKMeans(n_clusters=len(train_dataset) + num_query, random_state=0, sphere=sphere)
    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_
    # centers, _, _ = CustomKMeans_simpler(
    #     embeddings,
    #     n_clusters=len(train_dataset) + num_query,
    #     random_state=0,
    #     sphere=sphere,
    # )

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]
    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        min_ = 100000
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue
            dist = euc_distance(torch.tensor(center), v).item()
            if dist < min_:
                min_ = dist
                name = k
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_entropy_query(
    copy_model,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    smooth=1e-7,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    use_kmedian=False,
    hung_matching=False,
):
    ML_entropy = {}
    copy_model.eval()
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if embedding_method == "model":
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        copy_model.backbone.layer4.register_forward_hook(get_activation("features"))

        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        embeddings = {}

    with torch.no_grad():
        for img, label, names in all_train_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)  # shape = (batch_size, num_classes, 224, 224)

            if not all_train_dataset.multi_class:
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba = torch.stack((proba_pos, proba_neg), axis=1)
            else:
                # INSTRUMENT_CLASS = [1, 2, 3, 6, 7, 8, 9, 11]
                # BACKGROUND_CLASS = [0, 4, 5, 10]
                # all_proba = torch.softmax(pred, dim=1).cpu() # shape = (batch_size, num_classes, 224, 224)
                # instrument_pred = torch.sum(all_proba[:, INSTRUMENT_CLASS], dim=1) # shape = (batch_size, 224, 224)
                # background_pred = torch.sum(all_proba[:, BACKGROUND_CLASS], dim=1) # shape = (batch_size, 224, 224)
                # all_proba = torch.stack((background_pred, instrument_pred), axis=1) # shape = (batch_size, 2, 224, 224)
                all_proba = torch.softmax(pred, dim=1).cpu()

            log_proba = torch.log(
                all_proba + smooth
            )  # shape = (batch_size, 2, 224, 224)
            entropy = (all_proba * log_proba).sum(1)  # shape = (batch_size, 224, 224)

            if embedding_method == "model":
                features = pooling_layer(activation["features"]).squeeze()
            for i, name in enumerate(names):
                entropy_value = torch.mean(entropy[i]).item()  # shape = (224, 224)
                ML_entropy[name] = -entropy_value
                if embedding_method == "model":
                    embeddings[name] = features[i].cpu()

    with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(ML_entropy, f)

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_
    # centers, _, _ = CustomKMeans_simpler(
    #     embeddings,
    #     n_clusters=len(train_dataset) + num_query,
    #     random_state=0,
    #     sphere=sphere,
    # )

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]
    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        ent = -float("inf")
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue
            if ML_entropy[k] > ent:
                ent = ML_entropy[k]
                name = k
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_entropy_video_query(
    copy_model,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    smooth=1e-7,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    use_kmedian=False,
    hung_matching=False,
):
    ML_entropy = {}
    copy_model.eval()
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if embedding_method == "model":
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        copy_model.backbone.layer4.register_forward_hook(get_activation("features"))

        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        embeddings = {}

    with torch.no_grad():
        for img, label, names in all_train_dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = copy_model(img)  # shape = (batch_size, num_classes, 224, 224)

            if not all_train_dataset.multi_class:
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba = torch.stack((proba_pos, proba_neg), axis=1)
            else:
                # INSTRUMENT_CLASS = [1, 2, 3, 6, 7, 8, 9, 11]
                # BACKGROUND_CLASS = [0, 4, 5, 10]
                # all_proba = torch.softmax(pred, dim=1).cpu() # shape = (batch_size, num_classes, 224, 224)
                # instrument_pred = torch.sum(all_proba[:, INSTRUMENT_CLASS], dim=1) # shape = (batch_size, 224, 224)
                # background_pred = torch.sum(all_proba[:, BACKGROUND_CLASS], dim=1) # shape = (batch_size, 224, 224)
                # all_proba = torch.stack((background_pred, instrument_pred), axis=1) # shape = (batch_size, 2, 224, 224)
                all_proba = torch.softmax(pred, dim=1).cpu()

            log_proba = torch.log(
                all_proba + smooth
            )  # shape = (batch_size, 2, 224, 224)
            entropy = (all_proba * log_proba).sum(1)  # shape = (batch_size, 224, 224)

            if embedding_method == "model":
                features = pooling_layer(activation["features"]).squeeze()
            for i, name in enumerate(names):
                entropy_value = torch.mean(entropy[i]).item()  # shape = (224, 224)
                ML_entropy[name] = -entropy_value
                if embedding_method == "model":
                    embeddings[name] = features[i].cpu()

    with open(f"results/ML_entropy_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(ML_entropy, f)

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # get labeled frames per video
    labeled_frames_per_video = {}
    for path, label_path in train_dataset.data_pool:
        video_id, frame_nb = path[len(IMG_PATH) : -4].split("/")
        labeled_frames = labeled_frames_per_video.get(video_id, [])
        labeled_frames.append(frame_nb)
        labeled_frames_per_video[video_id] = labeled_frames

    # split embedding per video
    embeddings_per_video = {}
    for k, v in embeddings.items():
        video_id, frame_nb = k.split("/")
        frame_embeddings = embeddings_per_video.get(video_id, {})
        frame_embeddings[frame_nb] = v
        embeddings_per_video[video_id] = frame_embeddings

    new_labeled = collections.defaultdict(list)
    total_labeled = 0

    while total_labeled < num_query:
        # use kmeans on each video's frames
        video_centers = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            labeled_nb = len(labeled_frames_per_video.get(video_id, []))
            kmeans = KMeans(n_clusters=labeled_nb + 1, random_state=0)
            kmeans.fit(np.stack(list(frame_embeddings.values())))
            centers = kmeans.cluster_centers_
            video_centers[video_id] = centers

        # get the embedding of labeled frame from each video
        labeled_embedding_per_video = {}
        for video_id, frames in labeled_frames_per_video.items():
            for frame_nb in frames:
                labeled_embedding = labeled_embedding_per_video.get(video_id, {})
                labeled_embedding[frame_nb] = embeddings_per_video[video_id][frame_nb]
                labeled_embedding_per_video[video_id] = labeled_embedding

        # get the fixed cluster inside each video
        fixed_cluster_per_video = {}
        for video_id, labeled_embedding in labeled_embedding_per_video.items():
            centers = video_centers[video_id]
            fixed_cluster, centers = get_fixed_clusters(
                centers, labeled_embedding, hung_matching=False
            )
            fixed_cluster_per_video[video_id] = fixed_cluster
            video_centers[video_id] = centers

        # get the new centers for each video
        new_centers_per_video = {}
        k_means_centers_name_per_video = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
                embeddings=frame_embeddings,
                centers=video_centers[video_id],
                fixed_cluster=fixed_cluster_per_video.get(video_id, {}),
                sphere=sphere,
            )
            new_centers_per_video[video_id] = new_centers
            k_means_centers_name_per_video[video_id] = all_k_means_centers_name[-1]

        # get the frame with the highest entropy for each video
        max_cluster_entropy_per_video = {}
        for video_id, new_centers in new_centers_per_video.items():
            fixed_cluster = fixed_cluster_per_video.get(video_id, {})
            k_means_centers_name = k_means_centers_name_per_video[video_id]
            assert len(new_centers) == len(fixed_cluster) + 1
            for i, center in enumerate(new_centers):
                if i in fixed_cluster:
                    continue
                ent = -float("inf")
                name = None
                for k in k_means_centers_name[i]:
                    if ML_entropy[f"{video_id}/{k}"] > ent:
                        ent = ML_entropy[f"{video_id}/{k}"]
                        name = k
                max_cluster_entropy_per_video[video_id] = (ent, name)

        # sort the videos by the ones with the fewest labeled frames first
        # then the video with the highest entropy
        sorted_videos = []
        for video_id in embeddings_per_video.keys():
            nb_of_labeled_frames = len(labeled_frames_per_video.get(video_id, []))
            video_highest_entropy, frame_nb = max_cluster_entropy_per_video[video_id]
            sorted_videos.append(
                (video_id, frame_nb, nb_of_labeled_frames, -video_highest_entropy)
            )
        sorted_videos = sorted(sorted_videos, key=lambda x: (x[2], x[3]))

        # add the frame with the highest entropy to the labeled frames
        for video_id, frame_nb, nb_of_labeled_frames, _ in sorted_videos:
            number = frame_nb[len(FRAME_KEYWORD) :]
            new_labeled[video_id].append(number)
            labeled_list = labeled_frames_per_video.get(video_id, [])
            labeled_list.append(frame_nb)
            labeled_frames_per_video[video_id] = labeled_list
            total_labeled += 1
            if total_labeled >= num_query:
                return new_labeled


def COWAL_entropy_patch_query(
    ML_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    smooth=1e-7,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
    patch_number=None,
    patch_shape=None,
):
    if patch_number is not None:
        all_train_dataset.return_patches = True
        patch_number = (
            patch_number if patch_shape == "superpixel" else patch_number**2
        )
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(
            all_train_dataloader, weight_path=weight_path, patch_number=patch_number
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)
    if patch_number is not None:
        all_train_dataset.return_patches = False

    ## cluster the patch embeddings
    patch_embeddings = {}
    for patch_id, embedding in enumerate(embeddings):
        for k, v in embedding.items():
            patch_embeddings[k + f"/{patch_id}"] = v

    curr_selected_patches = train_dataset.curr_selected_patches
    number_of_patches = len(np.concatenate(list(curr_selected_patches.values())))
    kmeans = KMeans(
        n_clusters=number_of_patches + num_query * patch_number, random_state=0
    )
    kmeans.fit(np.stack(list(patch_embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_embedding = {}
    for k, patches in curr_selected_patches.items():
        k = "/".join([k.split("/")[0], FRAME_KEYWORD + k.split("/")[1]])
        for patch_id in patches:
            labeled_embedding[k + f"/{patch_id}"] = patch_embeddings[k + f"/{patch_id}"]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=patch_embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]
    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        ent = -float("inf")
        name = None
        for k in k_means_centers_name[i]:
            video_id, frame_nb, patch_id = k.split("/")
            frame_id = "/".join([video_id, frame_nb])
            patch_id = int(patch_id)
            if ML_entropy[frame_id][patch_id] > ent:
                ent = ML_entropy[frame_id][patch_id]
                name = k

        class_name, number, patch_id = name.split("/")
        number = number[len(FRAME_KEYWORD) :]

        frame_id = "/".join([class_name, number])
        new_labeled[class_name].append(number)
        new_selected_patches[frame_id].append(int(patch_id))

    return new_labeled, new_selected_patches


def COWAL_classEntropy_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_
    # centers, _, _ = CustomKMeans_simpler(
    #     embeddings,
    #     n_clusters=len(train_dataset) + num_query,
    #     random_state=0,
    #     sphere=sphere,
    # )

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, _ = balance_classes(
        img_per_class, num_query, n_class=config["N_LABEL"], start=start
    )
    copy_added_imgs = {k: v for k, v in added_imgs.items()}

    ### cluster max v1 ###
    clusters_x_class = get_clusters_x_class(
        k_means_centers_name, fixed_cluster, ML_class_entropy, added_imgs, start=start
    )

    new_labeled = collections.defaultdict(list)
    img_should_be = {}
    for cluster, class_ in clusters_x_class.items():
        candidates = k_means_centers_name[cluster]
        max_class_entropy = -float("inf")
        final_candidate = None
        for candidate in candidates:
            if ML_class_entropy[candidate][class_] > max_class_entropy:
                max_class_entropy = ML_class_entropy[candidate][class_]
                final_candidate = candidate
        video_id, nb = final_candidate.split("/")
        nb = nb[len(FRAME_KEYWORD) :]
        new_labeled[video_id].append(nb)
        frame_id = "/".join([video_id, nb])
        img_should_be[frame_id] = (class_, max_class_entropy)

    # ### cluster max v2 ###
    # k_means_name_centers = {}
    # for cluster, names in k_means_centers_name.items():
    #     for name in names:
    #         k_means_name_centers[name] = cluster

    # ML_entropy = torch.load(f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    # tuples = []
    # start = 1
    # if config['DATASET'] in routes.CLASS_0_DATASETS:
    #     start = 0
    # for frame_id, classEnt in ML_class_entropy.items():
    #     entropy = ML_entropy[frame_id]
    #     for class_ in range(start, len(classEnt)):
    #         cluster = k_means_name_centers[frame_id]
    #         if cluster in fixed_cluster:
    #             continue
    #         tuples.append((frame_id, class_, cluster, classEnt[class_] * entropy))
    # tuples = sorted(tuples, key=lambda x: x[-1], reverse=True)

    # new_labeled = collections.defaultdict(list)
    # already_picked_cluster = {}
    # for selected, class_, cluster, score in tuples:
    #     remaining = added_imgs.get(class_, 0)
    #     if remaining > 0 and cluster not in already_picked_cluster:
    #         assert cluster not in fixed_cluster

    #         video_id, nb = selected.split('/')
    #         added_imgs[class_] -= 1
    #         already_picked_cluster[cluster] = True

    #         nb = nb[len(FRAME_KEYWORD):]
    #         if video_id not in new_labeled or nb not in new_labeled[video_id]:
    #             new_labeled[video_id].append(nb)
    #         frame_id = '/'.join([video_id, nb])

    ## check the actual added classes
    true_added_imgs = {}
    true_added_pixels = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    img_query_acc = {}
    for k, v in new_labeled.items():
        for v2 in v:
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = train_dataset.open_path(img_path, lab_path, name=None)

            ## global acc
            # wanted_class, corresponding_ent = img_should_be[f"{k}/{v2}"]
            # unique_classes = np.unique(y)
            # if wanted_class in unique_classes:
            #     img_query_acc[wanted_class] = img_query_acc.get(wanted_class, 0) + 1

            for c in range(config["N_LABEL"]):
                if torch.sum(y == c) > 0:
                    true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                    true_added_pixels[c] = (
                        true_added_pixels.get(c, 0) + torch.sum(y == c).item()
                    )

    for c in range(config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {copy_added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)
    with open(f"results/copy_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(copy_added_imgs, f)
    with open(f"results/img_query_acc_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(img_query_acc, f)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_classEntropy_video_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # get labeled frames per video
    labeled_frames_per_video = {}
    for path, label_path in train_dataset.data_pool:
        video_id, frame_nb = path[len(IMG_PATH) : -4].split("/")
        labeled_frames = labeled_frames_per_video.get(video_id, [])
        labeled_frames.append(frame_nb)
        labeled_frames_per_video[video_id] = labeled_frames

    # split embedding per video
    embeddings_per_video = {}
    for k, v in embeddings.items():
        video_id, frame_nb = k.split("/")
        frame_embeddings = embeddings_per_video.get(video_id, {})
        frame_embeddings[frame_nb] = v
        embeddings_per_video[video_id] = frame_embeddings

    new_labeled = collections.defaultdict(list)
    total_labeled = 0

    # get distribution of classes in the dataset
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    while total_labeled < num_query:
        img_per_class = {}
        start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
        for video_id, frames in labeled_frames_per_video.items():
            for number in frames:
                img_path = f"{IMG_PATH}{video_id}/{number}{img_file_type}"
                lab_path = f"{LAB_PATH}{video_id}/{number}{lab_file_type}"
                x, y = train_dataset.open_path(img_path, lab_path, name=None)

                for c in range(start, config["N_LABEL"]):
                    if torch.sum(y == c) > 0:
                        img_per_class[c] = img_per_class.get(c, 0) + 1

        added_imgs, _ = balance_classes(
            img_per_class,
            num_query - total_labeled,
            n_class=config["N_LABEL"],
            start=start,
        )

        # use kmeans on each video's frames
        video_centers = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            labeled_nb = len(labeled_frames_per_video.get(video_id, []))
            kmeans = KMeans(n_clusters=labeled_nb + 1, random_state=0)
            kmeans.fit(np.stack(list(frame_embeddings.values())))
            centers = kmeans.cluster_centers_
            video_centers[video_id] = centers

        # get the embedding of labeled frame from each video
        labeled_embedding_per_video = {}
        for video_id, frames in labeled_frames_per_video.items():
            for frame_nb in frames:
                labeled_embedding = labeled_embedding_per_video.get(video_id, {})
                labeled_embedding[frame_nb] = embeddings_per_video[video_id][frame_nb]
                labeled_embedding_per_video[video_id] = labeled_embedding

        # get the fixed cluster inside each video
        fixed_cluster_per_video = {}
        for video_id, labeled_embedding in labeled_embedding_per_video.items():
            centers = video_centers[video_id]
            fixed_cluster, centers = get_fixed_clusters(
                centers, labeled_embedding, hung_matching=False
            )
            fixed_cluster_per_video[video_id] = fixed_cluster
            video_centers[video_id] = centers

        # get the new centers for each video
        new_centers_per_video = {}
        k_means_centers_name_per_video = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
                embeddings=frame_embeddings,
                centers=video_centers[video_id],
                fixed_cluster=fixed_cluster_per_video.get(video_id, {}),
                sphere=sphere,
            )
            new_centers_per_video[video_id] = new_centers
            k_means_centers_name_per_video[video_id] = all_k_means_centers_name[-1]

        ### cluster max v1 ###
        videos_x_class = get_clusters_x_class_video(
            k_means_centers_name_per_video,
            fixed_cluster_per_video,
            ML_class_entropy,
            added_imgs,
            start=start,
        )

        # get the frame with the highest entropy for each video
        max_cluster_entropy_per_video = {}
        for video_id, class_ in videos_x_class.items():
            fixed_cluster = fixed_cluster_per_video.get(video_id, {})
            k_means_centers_name = k_means_centers_name_per_video[video_id]
            assert len(k_means_centers_name) == len(fixed_cluster) + 1
            for cluster, candidates in k_means_centers_name.items():
                if cluster not in fixed_cluster:
                    break
            entropy = -float("inf")
            name = None
            for k in candidates:
                if ML_class_entropy[f"{video_id}/{k}"][class_] > entropy:
                    entropy = ML_class_entropy[f"{video_id}/{k}"][class_]
                    name = k
            max_cluster_entropy_per_video[video_id] = (entropy, name)

        sorted_videos = []
        for video_id in videos_x_class.keys():
            nb_of_labeled_frames = len(labeled_frames_per_video.get(video_id, []))
            entropy, frame_nb = max_cluster_entropy_per_video[video_id]
            sorted_videos.append((video_id, frame_nb, nb_of_labeled_frames, -entropy))
        sorted_videos = sorted(sorted_videos, key=lambda x: (x[2], x[3]))

        # add the frame with the highest entropy to the labeled frames
        for video_id, frame_nb, nb_of_labeled_frames, _ in sorted_videos:
            number = frame_nb[len(FRAME_KEYWORD) :]
            new_labeled[video_id].append(number)

            labeled_list = labeled_frames_per_video.get(video_id, [])
            labeled_list.append(frame_nb)
            labeled_frames_per_video[video_id] = labeled_list

            total_labeled += 1
            if total_labeled >= num_query:
                return new_labeled


def COWAL_classEntropy_v2_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]
    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        ent = -float("inf")
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue

            avg_class_ent = np.mean(ML_class_entropy[k])
            if avg_class_ent > ent:
                ent = avg_class_ent
                name = k
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_classEntropy_v2_video_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # get labeled frames per video
    labeled_frames_per_video = {}
    for path, label_path in train_dataset.data_pool:
        video_id, frame_nb = path[len(IMG_PATH) : -4].split("/")
        labeled_frames = labeled_frames_per_video.get(video_id, [])
        labeled_frames.append(frame_nb)
        labeled_frames_per_video[video_id] = labeled_frames

    # split embedding per video
    embeddings_per_video = {}
    for k, v in embeddings.items():
        video_id, frame_nb = k.split("/")
        frame_embeddings = embeddings_per_video.get(video_id, {})
        frame_embeddings[frame_nb] = v
        embeddings_per_video[video_id] = frame_embeddings

    new_labeled = collections.defaultdict(list)
    total_labeled = 0

    while total_labeled < num_query:
        # use kmeans on each video's frames
        video_centers = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            labeled_nb = len(labeled_frames_per_video.get(video_id, []))
            kmeans = KMeans(n_clusters=labeled_nb + 1, random_state=0)
            kmeans.fit(np.stack(list(frame_embeddings.values())))
            centers = kmeans.cluster_centers_
            video_centers[video_id] = centers

        # get the embedding of labeled frame from each video
        labeled_embedding_per_video = {}
        for video_id, frames in labeled_frames_per_video.items():
            for frame_nb in frames:
                labeled_embedding = labeled_embedding_per_video.get(video_id, {})
                labeled_embedding[frame_nb] = embeddings_per_video[video_id][frame_nb]
                labeled_embedding_per_video[video_id] = labeled_embedding

        # get the fixed cluster inside each video
        fixed_cluster_per_video = {}
        for video_id, labeled_embedding in labeled_embedding_per_video.items():
            centers = video_centers[video_id]
            fixed_cluster, centers = get_fixed_clusters(
                centers, labeled_embedding, hung_matching=False
            )
            fixed_cluster_per_video[video_id] = fixed_cluster
            video_centers[video_id] = centers

        # get the new centers for each video
        new_centers_per_video = {}
        k_means_centers_name_per_video = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
                embeddings=frame_embeddings,
                centers=video_centers[video_id],
                fixed_cluster=fixed_cluster_per_video.get(video_id, {}),
                sphere=sphere,
            )
            new_centers_per_video[video_id] = new_centers
            k_means_centers_name_per_video[video_id] = all_k_means_centers_name[-1]

        # get the frame with the highest entropy for each video
        max_cluster_entropy_per_video = {}
        for video_id, new_centers in new_centers_per_video.items():
            fixed_cluster = fixed_cluster_per_video.get(video_id, {})
            k_means_centers_name = k_means_centers_name_per_video[video_id]
            assert len(new_centers) == len(fixed_cluster) + 1
            for i, center in enumerate(new_centers):
                if i in fixed_cluster:
                    continue
                ent = -float("inf")
                name = None
                for k in k_means_centers_name[i]:
                    avg_class_ent = np.mean(ML_class_entropy[f"{video_id}/{k}"])
                    if avg_class_ent > ent:
                        ent = avg_class_ent
                        name = k
                max_cluster_entropy_per_video[video_id] = (ent, name)

        # sort the videos by the ones with the fewest labeled frames first
        # then the video with the highest entropy
        sorted_videos = []
        for video_id in embeddings_per_video.keys():
            nb_of_labeled_frames = len(labeled_frames_per_video.get(video_id, []))
            video_highest_entropy, frame_nb = max_cluster_entropy_per_video[video_id]
            sorted_videos.append(
                (video_id, frame_nb, nb_of_labeled_frames, -video_highest_entropy)
            )
        sorted_videos = sorted(sorted_videos, key=lambda x: (x[2], x[3]))

        # add the frame with the highest entropy to the labeled frames
        for video_id, frame_nb, nb_of_labeled_frames, _ in sorted_videos:
            number = frame_nb[len(FRAME_KEYWORD) :]
            new_labeled[video_id].append(number)
            labeled_list = labeled_frames_per_video.get(video_id, [])
            labeled_list.append(frame_nb)
            labeled_frames_per_video[video_id] = labeled_list
            total_labeled += 1
            if total_labeled >= num_query:
                return new_labeled


def COWAL_classEntropy_v2_1_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, _ = balance_classes(
        img_per_class, num_query, n_class=config["N_LABEL"], start=start
    )
    class_weights = {k: v / num_query for k, v in added_imgs.items()}

    new_labeled = collections.defaultdict(list)
    for i, center in enumerate(new_centers):
        if i in fixed_cluster:
            continue
        ent = -float("inf")
        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue

            avg_class_ent = 0
            for c in range(start, config["N_LABEL"]):
                avg_class_ent += ML_class_entropy[k][c] * class_weights.get(c, 0)
            if avg_class_ent > ent:
                ent = avg_class_ent
                name = k
        class_name = name[: len(CLASS_ID_TYPE) - 1]
        number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
        new_labeled[class_name].append(number)

    ## check the actual added classes
    true_added_imgs = {}
    true_added_pixels = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    for k, v in new_labeled.items():
        for v2 in v:
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = train_dataset.open_path(img_path, lab_path, name=None)

            for c in range(config["N_LABEL"]):
                if torch.sum(y == c) > 0:
                    true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                    true_added_pixels[c] = (
                        true_added_pixels.get(c, 0) + torch.sum(y == c).item()
                    )

    for c in range(config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)
    with open(f"results/copy_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(added_imgs, f)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_classEntropy_v2_2_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    kmeans = KMeans(n_clusters=len(train_dataset) + num_query, random_state=0)
    kmeans.fit(np.stack(list(embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_frames = [
        path[len(IMG_PATH) : -4] for path, label_path in train_dataset.data_pool
    ]
    labeled_embedding = {}
    for k in labeled_frames:
        labeled_embedding[k] = embeddings[k]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1

    new_labeled = collections.defaultdict(list)

    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    already_picked_cluster = {}
    for curr_query in range(num_query):

        added_imgs, _ = balance_classes(
            img_per_class,
            num_query - curr_query,
            n_class=config["N_LABEL"],
            start=start,
        )
        class_weights = {k: v / (num_query - curr_query) for k, v in added_imgs.items()}

        highest_entropy_per_cluster = {}
        for i, center in enumerate(new_centers):
            if i in fixed_cluster:
                continue

            ent = -float("inf")
            name = None
            for k in k_means_centers_name[i]:
                avg_class_ent = 0
                for c in range(start, config["N_LABEL"]):
                    avg_class_ent += ML_class_entropy[k][c] * class_weights.get(c, 0)
                if avg_class_ent > ent:
                    ent = avg_class_ent
                    name = k
            highest_entropy_per_cluster[i] = (name, ent)

        highest_entropy_per_cluster = sorted(
            highest_entropy_per_cluster.items(), key=lambda x: x[-1][-1], reverse=True
        )

        for cluster, (name, ent) in highest_entropy_per_cluster:
            if cluster not in already_picked_cluster:
                already_picked_cluster[cluster] = True
                class_name = name[: len(CLASS_ID_TYPE) - 1]
                number = name[len(CLASS_ID_TYPE) + len(FRAME_KEYWORD) :]
                new_labeled[class_name].append(number)
                break

        img_path = f"{IMG_PATH}{class_name}/{FRAME_KEYWORD}{number}{img_file_type}"
        lab_path = f"{LAB_PATH}{class_name}/{FRAME_KEYWORD}{number}{lab_file_type}"
        x, y = train_dataset.open_path(img_path, lab_path, name=None)

        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1

    ## check the actual added classes
    true_added_imgs = {}
    true_added_pixels = {}
    for k, v in new_labeled.items():
        for v2 in v:
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = train_dataset.open_path(img_path, lab_path, name=None)

            for c in range(config["N_LABEL"]):
                if torch.sum(y == c) > 0:
                    true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                    true_added_pixels[c] = (
                        true_added_pixels.get(c, 0) + torch.sum(y == c).item()
                    )

    for c in range(config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)
    with open(f"results/copy_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(added_imgs, f)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled


def COWAL_classEntropy_v2_2_video_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
):
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(all_train_dataloader, weight_path=weight_path)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)

    # get labeled frames per video
    labeled_frames_per_video = {}
    for path, label_path in train_dataset.data_pool:
        video_id, frame_nb = path[len(IMG_PATH) : -4].split("/")
        labeled_frames = labeled_frames_per_video.get(video_id, [])
        labeled_frames.append(frame_nb)
        labeled_frames_per_video[video_id] = labeled_frames

    # split embedding per video
    embeddings_per_video = {}
    for k, v in embeddings.items():
        video_id, frame_nb = k.split("/")
        frame_embeddings = embeddings_per_video.get(video_id, {})
        frame_embeddings[frame_nb] = v
        embeddings_per_video[video_id] = frame_embeddings

    new_labeled = collections.defaultdict(list)
    total_labeled = 0

    # get distribution of classes in the dataset
    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        for c in range(start, config["N_LABEL"]):
            if torch.sum(y == c) > 0:
                img_per_class[c] = img_per_class.get(c, 0) + 1
    added_imgs, _ = balance_classes(
        img_per_class, num_query - total_labeled, n_class=config["N_LABEL"], start=start
    )
    class_weights = {k: v / (num_query - total_labeled) for k, v in added_imgs.items()}

    while total_labeled < num_query:
        # use kmeans on each video's frames
        video_centers = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            labeled_nb = len(labeled_frames_per_video.get(video_id, []))
            kmeans = KMeans(n_clusters=labeled_nb + 1, random_state=0)
            kmeans.fit(np.stack(list(frame_embeddings.values())))
            centers = kmeans.cluster_centers_
            video_centers[video_id] = centers

        # get the embedding of labeled frame from each video
        labeled_embedding_per_video = {}
        for video_id, frames in labeled_frames_per_video.items():
            for frame_nb in frames:
                labeled_embedding = labeled_embedding_per_video.get(video_id, {})
                labeled_embedding[frame_nb] = embeddings_per_video[video_id][frame_nb]
                labeled_embedding_per_video[video_id] = labeled_embedding

        # get the fixed cluster inside each video
        fixed_cluster_per_video = {}
        for video_id, labeled_embedding in labeled_embedding_per_video.items():
            centers = video_centers[video_id]
            fixed_cluster, centers = get_fixed_clusters(
                centers, labeled_embedding, hung_matching=False
            )
            fixed_cluster_per_video[video_id] = fixed_cluster
            video_centers[video_id] = centers

        # get the new centers for each video
        new_centers_per_video = {}
        k_means_centers_name_per_video = {}
        for video_id, frame_embeddings in embeddings_per_video.items():
            new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
                embeddings=frame_embeddings,
                centers=video_centers[video_id],
                fixed_cluster=fixed_cluster_per_video.get(video_id, {}),
                sphere=sphere,
            )
            new_centers_per_video[video_id] = new_centers
            k_means_centers_name_per_video[video_id] = all_k_means_centers_name[-1]

        already_picked_video = {}
        for _ in range(len(embeddings_per_video)):  # nb of videos
            # get the frame with the highest entropy for each video's unique cluster
            max_cluster_entropy_per_video = {}
            for video_id, new_centers in new_centers_per_video.items():
                fixed_cluster = fixed_cluster_per_video.get(video_id, {})
                k_means_centers_name = k_means_centers_name_per_video[video_id]
                assert len(new_centers) == len(fixed_cluster) + 1
                for i, center in enumerate(new_centers):
                    if i in fixed_cluster:
                        continue
                    ent = -float("inf")
                    name = None
                    for k in k_means_centers_name[i]:
                        avg_class_ent = 0
                        for c in range(start, config["N_LABEL"]):
                            avg_class_ent += ML_class_entropy[f"{video_id}/{k}"][
                                c
                            ] * class_weights.get(c, 0)
                        if avg_class_ent > ent:
                            ent = avg_class_ent
                            name = k
                    max_cluster_entropy_per_video[video_id] = (ent, name)

            # sort the videos by the ones with the fewest labeled frames first
            # then the video with the highest entropy
            sorted_videos = []
            for video_id in embeddings_per_video.keys():
                nb_of_labeled_frames = len(labeled_frames_per_video.get(video_id, []))
                video_highest_entropy, frame_nb = max_cluster_entropy_per_video[
                    video_id
                ]
                sorted_videos.append(
                    (video_id, frame_nb, nb_of_labeled_frames, -video_highest_entropy)
                )
            sorted_videos = sorted(sorted_videos, key=lambda x: (x[2], x[3]))

            # add the frame with the highest entropy to the labeled frames
            for video_id, frame_nb, _, _ in sorted_videos:
                if video_id in already_picked_video:
                    continue
                already_picked_video[video_id] = True
                number = frame_nb[len(FRAME_KEYWORD) :]
                new_labeled[video_id].append(number)
                labeled_list = labeled_frames_per_video.get(video_id, [])
                labeled_list.append(frame_nb)
                labeled_frames_per_video[video_id] = labeled_list
                total_labeled += 1
                if total_labeled >= num_query:
                    return new_labeled
                break

            img_file_type = ".png"
            lab_file_type = ".png"
            if config["DATASET"] == "pets":
                img_file_type = ".jpg"
                lab_file_type = ""

            img_path = f"{IMG_PATH}{video_id}/{FRAME_KEYWORD}{number}{img_file_type}"
            lab_path = f"{LAB_PATH}{video_id}/{FRAME_KEYWORD}{number}{lab_file_type}"
            x, y = train_dataset.open_path(img_path, lab_path, name=None)

            for c in range(start, config["N_LABEL"]):
                if torch.sum(y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

            added_imgs, _ = balance_classes(
                img_per_class,
                num_query - total_labeled,
                n_class=config["N_LABEL"],
                start=start,
            )
            class_weights = {
                k: v / (num_query - total_labeled) for k, v in added_imgs.items()
            }


# kmeans applied to each group of patche_id
def COWAL_classEntropy_patch_v2_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
    patch_number=None,
):
    if patch_number is not None:
        all_train_dataset.return_patches = True
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(
            all_train_dataloader, weight_path=weight_path, patch_number=patch_number
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)
    if patch_number is not None:
        all_train_dataset.return_patches = False

    all_kmeans_centers_name = []
    all_fixed_clusters = []
    curr_selected_patches = train_dataset.curr_selected_patches
    img_per_patch = {}
    for k, v in curr_selected_patches.items():
        for nb in v:
            img_per_patch[nb] = img_per_patch.get(nb, []) + [k]
    for patch_id, embedding in enumerate(embeddings):
        k_means_centers_name, fixed_cluster = cluster_algorithm_patch(
            embedding,
            img_per_patch[
                patch_id
            ],  # set number of fixed id to number of labeled patch in this patch_id instead of the whole dataset length
            FRAME_KEYWORD=FRAME_KEYWORD,
            num_query=num_query,  # * patch_number ** 2) // 2, # get 450 cluster per patch id
            sphere=sphere,
            hung_matching=hung_matching,
            notebook=True,
        )
        all_kmeans_centers_name.append(k_means_centers_name)
        all_fixed_clusters.append(fixed_cluster)

    img_per_class = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]
        for patch_id in patch_ids:
            i, j = divmod(patch_id, patch_number)
            patch_size_x = y.shape[0] // patch_number
            patch_size_y = y.shape[1] // patch_number
            start_x = i * patch_size_x
            start_y = j * patch_size_y

            end_x = start_x + patch_size_x
            if i == patch_number - 1:
                end_x = y.shape[0]
            end_y = start_y + patch_size_y
            if j == patch_number - 1:
                end_y = y.shape[1]

            patch_y = y[start_x:end_x, start_y:end_y]
            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1

    added_imgs, _ = balance_classes(
        img_per_class, num_query * patch_number**2, n_class=config["N_LABEL"]
    )
    copy_added_imgs = {k: v for k, v in added_imgs.items()}

    all_clusters_x_class = get_patch_clusters_x_class(
        all_kmeans_centers_name,
        all_fixed_clusters,
        ML_class_entropy,
        added_imgs,
        dataset_name=config["DATASET"],
    )

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for patch_id, clusters_x_class in enumerate(all_clusters_x_class):
        for cluster, class_ in clusters_x_class.items():
            candidates = all_kmeans_centers_name[patch_id][cluster]
            max_class_entropy = -float("inf")
            final_candidate = None
            for candidate in candidates:
                if ML_class_entropy[candidate][patch_id, class_] > max_class_entropy:
                    max_class_entropy = ML_class_entropy[candidate][patch_id, class_]
                    final_candidate = candidate
            video_id, nb = final_candidate.split("/")
            nb = nb[len(FRAME_KEYWORD) :]
            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            frame_id = "/".join([video_id, nb])
            new_selected_patches[frame_id].append(patch_id)

    ## check the actual added classes
    true_added_imgs = {}
    true_added_pixels = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    for k, v in new_labeled.items():
        for v2 in v:
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = all_train_dataset.open_path(img_path, lab_path)
            patch_ids = new_selected_patches[f"{k}/{v2}"]
            for patch_id in patch_ids:
                i, j = divmod(patch_id, patch_number)
                patch_size_x = y.shape[0] // patch_number
                patch_size_y = y.shape[1] // patch_number
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == patch_number - 1:
                    end_x = y.shape[0]
                end_y = start_y + patch_size_y
                if j == patch_number - 1:
                    end_y = y.shape[1]

                patch_y = y[start_x:end_x, start_y:end_y]
                for c in range(start, config["N_LABEL"]):
                    if torch.sum(patch_y == c) > 0:
                        true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                        true_added_pixels[c] = (
                            true_added_pixels.get(c, 0) + torch.sum(patch_y == c).item()
                        )

    total_foreground_pixel = sum(
        [
            v
            for k, v in true_added_pixels.items()
            if k != 0 or config["DATASET"] in routes.CLASS_0_DATASETS
        ]
    )
    for c in range(start, config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {copy_added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}, {true_added_pixels.get(c, 0) / total_foreground_pixel: 0.03f}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')

    # save true added imgs and pixels
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled, new_selected_patches


# each patch is an individual sample
def COWAL_classEntropy_patch_query(
    ML_class_entropy,
    train_dataset,
    all_train_dataset,
    num_query,
    n_round,
    SEED,
    embedding_method="resnet",
    weight_path="../pretrained_models/skateboard_carl/checkpoint.pth",
    sphere=False,
    hung_matching=False,
    patch_number=None,
    patch_shape=None,
):
    if patch_number is not None:
        all_train_dataset.return_patches = True
        patch_number = (
            patch_number if patch_shape == "superpixel" else patch_number**2
        )
    all_train_dataloader = DataLoader(
        all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    if not os.path.isdir(f"results/embeddings"):
        os.mkdir(f"results/embeddings")

    if n_round == 0 and embedding_method == "resnet":
        embeddings = resnet_embedding(all_train_dataloader)
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "simCLR":
        embeddings = simCLR_embedding(
            all_train_dataloader, weight_path=weight_path, patch_number=patch_number
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "projection_simCLR":
        embeddings = simCLR_projection_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
    elif n_round == 0 and embedding_method == "BYOL":
        all_train_dataset = DataHandlerBYOL(all_train_dataset.data_path)
        all_train_dataloader = DataLoader(
            all_train_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
        )
        projections, embeddings = BYOL_embedding(
            all_train_dataloader, weight_path=weight_path
        )
        torch.save(embeddings, f"results/embeddings/embeddings_SEED={SEED}.pth")
        torch.save(
            projections, f"results/embeddings/projection_embeddings_SEED={SEED}.pth"
        )
    else:
        embeddings = torch.load(f"results/embeddings/embeddings_SEED={SEED}.pth")

    if sphere:
        for k, v in embeddings.items():
            embeddings[k] = F.normalize(v, dim=0)
    if patch_number is not None:
        all_train_dataset.return_patches = False

    ## cluster the patch embeddings
    patch_embeddings = {}
    for patch_id, embedding in enumerate(embeddings):
        for k, v in embedding.items():
            patch_embeddings[k + f"/{patch_id}"] = v

    curr_selected_patches = train_dataset.curr_selected_patches
    number_of_patches = len(np.concatenate(list(curr_selected_patches.values())))
    kmeans = KMeans(
        n_clusters=number_of_patches + num_query * patch_number, random_state=0
    )
    kmeans.fit(np.stack(list(patch_embeddings.values())))
    centers = kmeans.cluster_centers_

    labeled_embedding = {}
    for k, patches in curr_selected_patches.items():
        k = "/".join([k.split("/")[0], FRAME_KEYWORD + k.split("/")[1]])
        for patch_id in patches:
            labeled_embedding[k + f"/{patch_id}"] = patch_embeddings[k + f"/{patch_id}"]

    fixed_cluster, centers = get_fixed_clusters(
        centers, labeled_embedding, hung_matching=hung_matching
    )
    new_centers, all_centers, all_k_means_centers_name = CustomKMeans_simpler(
        embeddings=patch_embeddings,
        centers=centers,
        fixed_cluster=fixed_cluster,
        sphere=sphere,
    )

    k_means_centers_name = all_k_means_centers_name[-1]

    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    ## find out how many patches to query per class
    patch_per_class = {}
    pixel_per_class = {}
    for i in range(len(train_dataset)):
        x, y, n = train_dataset[i]
        if patch_shape == "superpixel":
            superpixel_lab = train_dataset.load_superpixel(n, transform=True)

        n = "/".join([n.split("/")[-2], n.split("/")[-1][len(FRAME_KEYWORD) :]])
        patch_ids = curr_selected_patches[n]

        for patch_id in patch_ids:
            if patch_shape == "rectangle":
                i, j = divmod(patch_id, int(np.sqrt(patch_number)))
                patch_size_x = y.shape[0] // int(np.sqrt(patch_number))
                patch_size_y = y.shape[1] // int(np.sqrt(patch_number))
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == int(np.sqrt(patch_number)) - 1:
                    end_x = y.shape[0]
                end_y = start_y + patch_size_y
                if j == int(np.sqrt(patch_number)) - 1:
                    end_y = y.shape[1]

                patch_y = y[start_x:end_x, start_y:end_y]
            elif patch_shape == "superpixel":
                patch_y = y[superpixel_lab == patch_id]

            for c in range(start, config["N_LABEL"]):
                if torch.sum(patch_y == c) > 0:
                    patch_per_class[c] = patch_per_class.get(c, 0) + 1
                    pixel_per_class[c] = (
                        pixel_per_class.get(c, 0) + torch.sum(patch_y == c).item()
                    )

    ### always sample even number of image per class a t every round ###
    # patch_per_class = {k: 0 for k in range(start, config['N_LABEL'])}
    ###############################################################

    added_imgs, _ = balance_classes(
        patch_per_class,
        num_query * patch_number,
        n_class=config["N_LABEL"],
        start=start,
    )
    copy_added_imgs = {k: v for k, v in added_imgs.items()}

    ### balance pixel ###
    # added_pixels, _ = balance_pixels(pixel_per_class, num_query * y.shape[0] * y.shape[1], n_class=config['N_LABEL'], start=start)
    # copy_added_pixels = {k: v for k, v in added_pixels.items()}

    ## assign class to query per cluster
    # class_entropy_per_cluster = {}
    # for cluster, frames in k_means_centers_name.items():
    #     if cluster in fixed_cluster:
    #         continue
    #     avg_class_entropy = []
    #     for frame in frames:
    #         patch_id = int(frame.split('/')[-1])
    #         frame = '/'.join(frame.split('/')[:-1])
    #         avg_class_entropy.append(ML_class_entropy[frame][patch_id])
    #     avg_class_entropy = np.stack(avg_class_entropy)
    #     avg_class_entropy = np.mean(avg_class_entropy, axis=0) # changed from mean to max
    #     class_entropy_per_cluster[cluster] = avg_class_entropy

    # tuples = []
    # start = 1
    # if config['DATASET'] in routes.CLASS_0_DATASETS:
    #     start = 0
    # for cluster, row in class_entropy_per_cluster.items():
    #     for class_nb in range(start, len(row)):
    #         tuples.append((row[class_nb], cluster, class_nb))
    # tuples.sort(reverse=True)

    # clusters_x_class = {}
    # for entropy, cluster, class_ in tuples:
    #     remaining = added_imgs.get(class_, 0)
    #     if remaining > 0 and cluster not in clusters_x_class:
    #         clusters_x_class[cluster] = class_
    #         added_imgs[class_] -= 1

    # ## select the highest class entropy sample from the corresponding cluster
    # new_labeled = collections.defaultdict(list)
    # new_selected_patches = collections.defaultdict(list)
    # for cluster, class_ in clusters_x_class.items():
    #     candidates = k_means_centers_name[cluster]
    #     max_class_entropy = -float("inf")
    #     final_candidate = None
    #     selected_patch = None
    #     for candidate in candidates:
    #         patch_id = int(candidate.split('/')[-1])
    #         candidate = '/'.join(candidate.split('/')[:-1])
    #         if ML_class_entropy[candidate][patch_id, class_] > max_class_entropy:
    #             max_class_entropy = ML_class_entropy[candidate][patch_id, class_]
    #             final_candidate = candidate
    #             selected_patch = patch_id
    #     video_id, nb = final_candidate.split('/')
    #     nb = nb[len(FRAME_KEYWORD):]
    #     if video_id not in new_labeled or nb not in new_labeled[video_id]:
    #         new_labeled[video_id].append(nb)
    #     frame_id = '/'.join([video_id, nb])
    #     new_selected_patches[frame_id].append(selected_patch)

    ### cluser max v2 ###
    ## another way of seleting the samples with the hogest class entropy per class
    k_means_name_centers = {}
    for cluster, names in k_means_centers_name.items():
        for name in names:
            k_means_name_centers[name] = cluster

    ML_entropy = torch.load(f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    tuples = []
    start = 1
    if config["DATASET"] in routes.CLASS_0_DATASETS:
        start = 0
    for frame_id, patch_classEnt in ML_class_entropy.items():
        for patch_id in range(patch_classEnt.shape[0]):
            entropy = ML_entropy[frame_id][patch_id]
            for class_ in range(start, patch_classEnt.shape[1]):
                cluster = k_means_name_centers[frame_id + f"/{patch_id}"]
                if cluster in fixed_cluster:
                    continue
                tuples.append(
                    (
                        frame_id + f"/{patch_id}",
                        class_,
                        cluster,
                        patch_classEnt[patch_id, class_] * entropy,
                    )
                )
    tuples = sorted(tuples, key=lambda x: x[-1], reverse=True)

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    already_picked_cluster = {}
    for selected, class_, cluster, score in tuples:
        remaining = added_imgs.get(class_, 0)
        # remaining = added_pixels.get(class_, 0) # pixel balance
        if remaining > 0 and cluster not in already_picked_cluster:
            assert cluster not in fixed_cluster

            video_id, nb, patch_id = selected.split("/")

            # ###  balance pixels ###
            # predicted_mask = torch.load(f'./ML_preds/{video_id}/{nb}.pt')

            # i, j = divmod(int(patch_id), patch_number)
            # patch_size_x = y.shape[0] // patch_number
            # patch_size_y = y.shape[1] // patch_number
            # start_x = i * patch_size_x
            # start_y = j * patch_size_y

            # end_x = start_x + patch_size_x
            # if i == patch_number - 1:
            #     end_x = predicted_mask.shape[0]
            # end_y = start_y + patch_size_y
            # if j == patch_number - 1:
            #     end_y = predicted_mask.shape[1]

            # predicted_patch = predicted_mask[start_x:end_x, start_y:end_y]
            # added_pixels[class_] -= torch.sum(predicted_patch == class_).item()
            #######################
            added_imgs[class_] -= 1
            already_picked_cluster[cluster] = True

            nb = nb[len(FRAME_KEYWORD) :]
            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            frame_id = "/".join([video_id, nb])
            new_selected_patches[frame_id].append(int(patch_id))

    ## check the actual added classes
    true_added_imgs = {}
    true_added_pixels = {}
    start = 1 if config["DATASET"] not in routes.CLASS_0_DATASETS else 0
    img_file_type = ".png"
    lab_file_type = ".png"
    if config["DATASET"] == "pets":
        img_file_type = ".jpg"
        lab_file_type = ""

    for k, v in new_labeled.items():
        for v2 in v:
            img_path = f"{IMG_PATH}{k}/{FRAME_KEYWORD}{v2}{img_file_type}"
            lab_path = f"{LAB_PATH}{k}/{FRAME_KEYWORD}{v2}{lab_file_type}"
            x, y = all_train_dataset.open_path(img_path, lab_path)
            patch_ids = new_selected_patches[f"{k}/{v2}"]

            if patch_shape == "superpixel":
                superpixel_lab = train_dataset.load_superpixel(
                    f"{k}/{FRAME_KEYWORD}{v2}", transform=True
                )

            for patch_id in patch_ids:
                if patch_shape == "rectangle":
                    i, j = divmod(patch_id, int(np.sqrt(patch_number)))
                    patch_size_x = y.shape[0] // int(np.sqrt(patch_number))
                    patch_size_y = y.shape[1] // int(np.sqrt(patch_number))
                    start_x = i * patch_size_x
                    start_y = j * patch_size_y

                    end_x = start_x + patch_size_x
                    if i == int(np.sqrt(patch_number)) - 1:
                        end_x = y.shape[0]
                    end_y = start_y + patch_size_y
                    if j == int(np.sqrt(patch_number)) - 1:
                        end_y = y.shape[1]

                    patch_y = y[start_x:end_x, start_y:end_y]
                elif patch_shape == "superpixel":
                    patch_y = y[superpixel_lab == patch_id]
                for c in range(start, config["N_LABEL"]):
                    if torch.sum(patch_y == c) > 0:
                        true_added_imgs[c] = true_added_imgs.get(c, 0) + 1
                        true_added_pixels[c] = (
                            true_added_pixels.get(c, 0) + torch.sum(patch_y == c).item()
                        )

    total_foreground_pixel = sum(
        [
            v
            for k, v in true_added_pixels.items()
            if k != 0 or config["DATASET"] in routes.CLASS_0_DATASETS
        ]
    )
    for c in range(start, config["N_LABEL"]):
        with open(PRINT_PATH, "a") as f:
            f.write(
                f"Class: {c}, {copy_added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}, {true_added_pixels.get(c, 0) / total_foreground_pixel: 0.03f}\n"
            )
        # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')

        # pixel balance
        # with open(PRINT_PATH, 'a') as f:
        #     f.write(f'Class: {c}, {copy_added_pixels.get(c, 0)}, {true_added_pixels.get(c, 0)}\n')
        # # print(f'Class: {c}, {added_imgs.get(c, 0)}, {true_added_imgs.get(c, 0)}')

    # save true added imgs and pixels
    with open(f"results/true_added_imgs_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_imgs, f)
    with open(f"results/true_added_pixels_SEED={SEED}_round={n_round}.json", "w") as f:
        json.dump(true_added_pixels, f)

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k
    with open(
        f"results/k_means_name_centers_SEED={SEED}_round={n_round}.json", "w"
    ) as f:
        json.dump(k_means_name_centers, f)
    return new_labeled, new_selected_patches


def RIPU_PA_query(model, dataset, num_query, n_round=0, smooth=1e-7):
    if n_round == 0:
        if os.path.isdir(dataset.lab_path.replace("labels", "partial_labels")):
            shutil.rmtree(dataset.lab_path.replace("labels", "partial_labels"))

    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    mask_radius = 1

    calculate_purity = SpatialPurity(
        in_channels=config["N_LABEL"], size=2 * mask_radius + 1
    ).to(DEVICE)

    new_labeled = collections.defaultdict(list)
    with torch.no_grad():
        for imgs, labs, names in dataloader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)

            total_pixel_to_query = num_query * imgs.shape[2] * imgs.shape[3]
            pixel_per_img = total_pixel_to_query // len(dataset)

            out = model(imgs)  # shape = (batch_size, n_class, h, w)
            all_proba = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            # calculate entropy
            log_proba = torch.log(all_proba + smooth)
            entropy = (all_proba * log_proba).sum(1)  # shape = (batch_size, h, w)

            # calculate purity
            pseudo_label = torch.argmax(all_proba, dim=1)  # shape = (batch_size, h, w)
            one_hot = F.one_hot(
                pseudo_label, num_classes=config["N_LABEL"]
            ).float()  # shape = (batch_size, h, w, n_class)
            one_hot = one_hot.permute(
                (0, 3, 1, 2)
            )  # shape = (batch_size, n_class, h, w)
            purity = calculate_purity(
                one_hot.to(DEVICE)
            ).squeeze()  # shape = (batch_size, h, w)

            all_scores = purity * entropy.to(DEVICE)
            for i, name in enumerate(names):
                lab = labs[i]
                partial_lab = torch.zeros_like(lab).bool()
                partial_lab[lab != IGNORE_INDEX] = True
                partial_lab[lab == IGNORE_INDEX] = False

                void_regions = torch.zeros_like(lab).bool()
                void_regions[lab != IGNORE_INDEX] = True

                kernel_size = 2 * mask_radius + 1
                kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32).to(
                    DEVICE
                )
                void_regions = void_regions[None, None, ...].float()
                void_regions = F.conv2d(
                    void_regions, kernel[None, None, ...], padding=mask_radius
                )
                void_regions = (void_regions[0, 0, ...] > 0).bool()

                score = all_scores[i]
                score[void_regions] = -float("inf")

                for _ in range(pixel_per_img):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1

                    score[start_h:end_h, start_w:end_w] = -float("inf")
                    void_regions[start_h:end_h, start_w:end_w] = True
                    # active sampling
                    partial_lab[h, w] = True

                if (
                    os.path.isdir(
                        dataset.lab_path.replace("labels", "partial_labels")
                        + name.split("/")[0]
                    )
                    == False
                ):
                    os.makedirs(
                        dataset.lab_path.replace("labels", "partial_labels")
                        + name.split("/")[0]
                    )

                torch.save(
                    partial_lab,
                    dataset.lab_path.replace("labels", "partial_labels")
                    + name
                    + ".pth",
                )

                video_id, nb = name.split("/")
                nb = nb[len(FRAME_KEYWORD) :]
                new_labeled[video_id].append(nb)
    return new_labeled


def BvSB_patch_query(
    model, unlabeled_dataset, num_query, train_dataset=None, SEED=0, n_round=0, patch_number=None, patch_shape=None
):
    BvSB = {}
    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    # pixel_per_img = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            # pixel_per_img = img.shape[2] * img.shape[3]
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            softmax_output = np.sort(softmax_output, axis=1)
            uncertainty_output = (
                softmax_output[:, -2, :, :] / softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)
                patch_ids = np.unique(superpixel_lab)

                all_patch_entropy = []
                for patch_id in patch_ids:
                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_entropy = np.mean(patch_entropy)
                    all_patch_entropy.append(patch_entropy)

                BvSB[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)

    torch.save(BvSB, f"results/BvSB_SEED={SEED}_round={n_round}.pt")

    patch_number = patch_number if patch_shape == "superpixel" else patch_number**2
    patch_BvSB = []
    for k, v in BvSB.items():
        for patch_nb, score in enumerate(v):
            patch_BvSB.append((score, k, patch_nb))
    patch_BvSB = sorted(patch_BvSB, key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, patch_BvSB, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = patch_BvSB[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            frame_id = "/".join([video_id, nb])
            new_selected_patches[frame_id].append(int(patch_nb))

    return new_labeled, new_selected_patches


def BvSB_patch_v2_query(
    model, 
    unlabeled_dataset, 
    num_query, 
    train_dataset=None, 
    SEED=0, n_round=0, patch_number=None, patch_shape=None
):
    BvSB = {}
    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    curr_selected_patches = train_dataset.curr_selected_patches
    # pixel_per_img = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            # pixel_per_img = img.shape[2] * img.shape[3]
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            softmax_output = np.sort(softmax_output, axis=1)
            uncertainty_output = (
                softmax_output[:, -2, :, :] / softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]

                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                all_patch_entropy = []
                for patch_id in range(patch_number):
                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    if len(patch_entropy) == 0 or patch_id in selected_patches:
                        patch_entropy = 0
                    else:
                        patch_entropy = np.mean(patch_entropy)
                    all_patch_entropy.append(patch_entropy)

                BvSB[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)

    torch.save(BvSB, f"results/BvSB_SEED={SEED}_round={n_round}.pt")

    patch_BvSB = []
    for k, v in BvSB.items():
        for patch_nb, score in enumerate(v):
            patch_BvSB.append((score, k, patch_nb))
    patch_BvSB = sorted(patch_BvSB, key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, patch_BvSB, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = patch_BvSB[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            frame_id = "/".join([video_id, nb])
            new_selected_patches[frame_id].append(int(patch_id))

    return new_labeled, new_selected_patches

def revisiting_query(
    model,
    unlabeled_dataset,
    num_query,
    train_dataset=None,
    SEED=0,
    n_round=0,
    patch_number=None,
    smooth=1e-7,
):
    ML_entropy = {}
    BvSB = {}
    region_dominant_label = {}
    weight = {}

    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    # pixel_per_img = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            # pixel_per_img = img.shape[2] * img.shape[3]
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            # calculate entropy
            log_proba = torch.log(softmax_output + smooth)
            global_entropy = (softmax_output * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            softmax_output = np.sort(softmax_output, axis=1)
            uncertainty_output = (
                softmax_output[:, -2, :, :] / softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)
            pred = torch.argmax(out, dim=1).cpu().numpy()  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transorm=True)
                patch_ids = np.unique(superpixel_lab)

                all_patch_ML_entropy = []
                all_patch_entropy = []
                all_dominant_labels = []
                for patch_id in patch_ids:
                    # save entropy value
                    patch_ML_entropy = global_entropy[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_ML_entropy = -torch.mean(patch_ML_entropy).item()
                    all_patch_ML_entropy.append(patch_ML_entropy)
                    ######

                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_entropy = np.mean(patch_entropy)
                    all_patch_entropy.append(patch_entropy)

                    unique, unique_counts = np.unique(
                        pred[i, superpixel_lab == patch_id], return_counts=True
                    )
                    dominant_label = unique[np.argmax(unique_counts)]
                    all_dominant_labels.append(dominant_label)

                    weight[dominant_label] = weight.get(dominant_label, 0) + 1

                ML_entropy[name] = np.stack(
                    all_patch_ML_entropy, axis=0
                )  # shape = (patch_number**2)
                BvSB[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)
                region_dominant_label[name] = np.stack(
                    all_dominant_labels, axis=0
                )  # shape = (patch_number**2)

    torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    # torch.save(BvSB, f"results/BvSB_SEED={SEED}_round={n_round}.pt")
    # torch.save(region_dominant_label, f"results/region_dominant_label_SEED={SEED}_round={n_round}.pt")

    total_region_num = sum(weight.values())
    for k, v in weight.items():
        weight[k] = np.exp(-v / total_region_num)

    # torch.save(weight, f"results/weight_SEED={SEED}_round={n_round}.pt")
    final_scores = []
    for k, v in BvSB.items():
        for patch_id, uncertainty in enumerate(v):
            final_scores.append(
                (uncertainty * weight[region_dominant_label[k][patch_id]], k, patch_id)
            )
    final_scores.sort(key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, final_scores, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        # pixel_budget = num_query * pixel_per_img
        # i = 0
        # while pixel_budget > 0:
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = final_scores[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            new_selected_patches["/".join([video_id, nb])].append(int(patch_id))

            # superpixel_lab = dataset.load_superpixel(frame_id)
            # i += 1
            # pixel_budget -= np.sum(superpixel_lab == patch_id)

    return new_labeled, new_selected_patches

# do not select already selected patches
def revisiting_v2_query(
    model,
    unlabeled_dataset,
    num_query,
    train_dataset=None,
    SEED=0,
    n_round=0,
    patch_number=None,
    smooth=1e-7,
):
    ML_entropy = {}
    BvSB = {}
    region_dominant_label = {}
    weight = {}

    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    curr_selected_patches = train_dataset.curr_selected_patches
    # pixel_per_img = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            # pixel_per_img = img.shape[2] * img.shape[3]
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            # calculate entropy
            log_proba = torch.log(softmax_output + smooth)
            global_entropy = (softmax_output * log_proba).sum(
                1
            )  # shape = (batch_size, h, w)

            softmax_output = np.sort(softmax_output, axis=1)
            uncertainty_output = (
                softmax_output[:, -2, :, :] / softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)
            pred = torch.argmax(out, dim=1).cpu().numpy()  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]

                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                all_patch_ML_entropy = [] # normal entropy
                all_patch_entropy = [] #  BvSB
                all_dominant_labels = []
                for patch_id in range(patch_number):
                    # save entropy value
                    patch_ML_entropy = global_entropy[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    if len(patch_ML_entropy) == 0 or patch_id in selected_patches:
                        patch_ML_entropy = 0
                    else:
                        patch_ML_entropy = -torch.mean(patch_ML_entropy).item()
                    all_patch_ML_entropy.append(patch_ML_entropy)
                    ######

                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    if len(patch_entropy) == 0 or patch_id in selected_patches:
                        patch_entropy = 0
                    else:
                        patch_entropy = np.mean(patch_entropy)
                    all_patch_entropy.append(patch_entropy)

                    unique, unique_counts = np.unique(
                        pred[i, superpixel_lab == patch_id], return_counts=True
                    )
                    if len(unique) > 0:
                        dominant_label = unique[np.argmax(unique_counts)]
                        all_dominant_labels.append(dominant_label)

                        weight[dominant_label] = weight.get(dominant_label, 0) + 1
                    else:
                        all_dominant_labels.append(None)

                ML_entropy[name] = np.stack(
                    all_patch_ML_entropy, axis=0
                )  # shape = (patch_number**2)
                BvSB[name] = np.stack(
                    all_patch_entropy, axis=0
                )  # shape = (patch_number**2)
                region_dominant_label[name] = np.stack(
                    all_dominant_labels, axis=0
                )  # shape = (patch_number**2)

    torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
    # torch.save(BvSB, f"results/BvSB_SEED={SEED}_round={n_round}.pt")
    # torch.save(region_dominant_label, f"results/region_dominant_label_SEED={SEED}_round={n_round}.pt")

    total_region_num = sum(weight.values())
    for k, v in weight.items():
        weight[k] = np.exp(-v / total_region_num)

    # torch.save(weight, f"results/weight_SEED={SEED}_round={n_round}.pt")
    final_scores = []
    for k, v in BvSB.items():
        for patch_id, uncertainty in enumerate(v):
            dominant = region_dominant_label[k][patch_id]
            if dominant:
                weight_ = weight[dominant]
            else:
                weight_ = 0
            final_scores.append(
                (uncertainty * weight_, k, patch_id)
            )
    final_scores.sort(key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, final_scores, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        # pixel_budget = num_query * pixel_per_img
        # i = 0
        # while pixel_budget > 0:
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = final_scores[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            new_selected_patches["/".join([video_id, nb])].append(int(patch_id))

            # superpixel_lab = dataset.load_superpixel(frame_id)
            # i += 1
            # pixel_budget -= np.sum(superpixel_lab == patch_id)

    return new_labeled, new_selected_patches

def pixelBal_query(
    model, unlabeled_dataset, num_query, train_dataset=None, SEED=0, n_round=0, patch_number=None, smooth=1e-7
):

    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    mu = config["MU"]

    with torch.no_grad():
        weight = {}
        total_pixel = 0
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            for i, name in enumerate(names):
                for c in range(config["N_LABEL"]):
                    weight[c] = weight.get(c, 0) + np.sum(softmax_output[i, c].numpy())
                    total_pixel += (
                        softmax_output[i, c].shape[0] * softmax_output[i, c].shape[1]
                    )

        for k, v in weight.items():
            weight[k] = v / total_pixel

        torch.save(weight, f"results/weight_SEED={SEED}_round={n_round}.pt")

        final_scores = []
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)
            sorted_softmax_output = np.sort(
                softmax_output, axis=1
            )  # shape = (batch_size, n_class, h, w)
            uncertainty_output = (
                sorted_softmax_output[:, -2, :, :] / sorted_softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)
            pred = (
                torch.argmax(softmax_output, dim=1).cpu().numpy()
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)
                patch_ids = np.unique(superpixel_lab)

                for patch_id in patch_ids:
                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_pred = pred[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_weight = np.array(
                        [weight[c] for c in patch_pred]
                    )  # shape = (n_pixel)

                    patch_entropy = (
                        patch_entropy / (1 + mu * patch_weight) ** 2
                    )  # shape = (n_pixel)
                    patch_entropy = np.mean(patch_entropy)  # shape = (1)

                    final_scores.append((patch_entropy, name, patch_id))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, final_scores, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = final_scores[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            new_selected_patches["/".join([video_id, nb])].append(int(patch_id))

    return new_labeled, new_selected_patches


def pixelBal_v2_query(
    model, 
    unlabeled_dataset, 
    num_query, 
    train_dataset=None, 
    SEED=0, n_round=0, patch_number=None, smooth=1e-7
):

    model.eval()
    dataloader = DataLoader(
        unlabeled_dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )
    mu = config["MU"]

    curr_selected_patches = train_dataset.curr_selected_patches

    weight = {}
    total_pixel = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            for i, name in enumerate(names):
                for c in range(config["N_LABEL"]):
                    weight[c] = weight.get(c, 0) + np.sum(softmax_output[i, c].numpy())
                    total_pixel += (
                        softmax_output[i, c].shape[0] * softmax_output[i, c].shape[1]
                    )

        for k, v in weight.items():
            weight[k] = v / total_pixel

        torch.save(weight, f"results/weight_SEED={SEED}_round={n_round}.pt")

        final_scores = []
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)  # shape = (batch_size, n_class, h, w)

            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)
            sorted_softmax_output = np.sort(
                softmax_output, axis=1
            )  # shape = (batch_size, n_class, h, w)
            uncertainty_output = (
                sorted_softmax_output[:, -2, :, :] / sorted_softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)
            pred = (
                torch.argmax(softmax_output, dim=1).cpu().numpy()
            )  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]
                
                superpixel_lab = unlabeled_dataset.load_superpixel(name, transform=True)

                for patch_id in range(patch_number):
                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_pred = pred[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    patch_weight = np.array(
                        [weight[c] for c in patch_pred]
                    )  # shape = (n_pixel)

                    if len(patch_entropy) == 0 or patch_id in selected_patches:
                        patch_entropy = 0
                    else:
                        patch_entropy = (
                            patch_entropy / (1 + mu * patch_weight) ** 2
                        )  # shape = (n_pixel)
                        patch_entropy = np.mean(patch_entropy)  # shape = (1)

                    final_scores.append((patch_entropy, name, patch_id))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    if config['MULTI_CLASS_LABELING']:
        new_labeled, new_selected_patches = count_multiClassLabeling_click(
            num_query * patch_number, final_scores, train_dataset
        )
    else:
        new_labeled = collections.defaultdict(list)
        new_selected_patches = collections.defaultdict(list)
        for i in range(num_query * patch_number):
            score, frame_id, patch_id = final_scores[i]
            video_id, nb = frame_id.split("/")
            nb = nb[len(FRAME_KEYWORD) :]

            if video_id not in new_labeled or nb not in new_labeled[video_id]:
                new_labeled[video_id].append(nb)
            new_selected_patches["/".join([video_id, nb])].append(int(patch_id))

    return new_labeled, new_selected_patches


def revisiting_adaptiveSP_query(
    model, dataset, num_query, SEED=0, n_round=0, patch_number=None
):
    BvSB = {}
    patch_sizes = {}
    predicted_softmax = {}
    region_dominant_label = {}
    weight = {}

    model.eval()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"]
    )

    # pixel_per_img = 0
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            # pixel_per_img = img.shape[2] * img.shape[3]
            out = model(img)  # shape = (batch_size, n_class, h, w)
            softmax_output = torch.softmax(
                out, dim=1
            ).cpu()  # shape = (batch_size, n_class, h, w)

            sorted_softmax_output = np.sort(softmax_output, axis=1)
            uncertainty_output = (
                sorted_softmax_output[:, -2, :, :] / sorted_softmax_output[:, -1, :, :]
            )  # shape = (batch_size, h, w)
            pred = torch.argmax(out, dim=1).cpu().numpy()  # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                video_id, frame_nb = name.split("/")
                save_path = f"./ML_preds/{video_id}"
                if not os.path.isdir(save_path):
                    os.makedirs(save_path, exist_ok=True)
                torch.save(softmax_output[i], save_path + f"/{frame_nb}.pt")

                superpixel_lab = dataset.load_superpixel(name, transform=True)
                patch_ids = np.unique(superpixel_lab)

                all_patch_entropy = {}
                all_patch_size = {}
                all_dominant_labels = {}
                all_patch_softmax = {}
                for patch_id in patch_ids:
                    patch_entropy = uncertainty_output[
                        i, superpixel_lab == patch_id
                    ]  # shape = (n_pixel)
                    all_patch_size[patch_id] = len(patch_entropy)
                    patch_entropy = np.sum(patch_entropy)
                    all_patch_entropy[patch_id] = patch_entropy

                    unique, unique_counts = np.unique(
                        pred[i, superpixel_lab == patch_id], return_counts=True
                    )
                    dominant_label = unique[np.argmax(unique_counts)]
                    all_dominant_labels[patch_id] = dominant_label
                    weight[dominant_label] = weight.get(dominant_label, 0) + np.sum(
                        superpixel_lab == patch_id
                    )

                    patch_softmax = softmax_output[
                        i, :, superpixel_lab == patch_id
                    ]  # shape = (n_class, n_pixel)
                    patch_softmax = torch.mean(
                        patch_softmax, axis=1
                    ).numpy()  # shape = (n_class)
                    all_patch_softmax[patch_id] = patch_softmax

                BvSB[name] = all_patch_entropy  # shape = (patch_number)
                region_dominant_label[
                    name
                ] = all_dominant_labels  # shape = (patch_number)
                predicted_softmax[
                    name
                ] = all_patch_softmax  # shape = (patch_number, n_class)
                patch_sizes[name] = all_patch_size  # shape = (patch_number)

    torch.save(BvSB, f"results/BvSB_SEED={SEED}_round={n_round}.pt")
    torch.save(
        region_dominant_label,
        f"results/region_dominant_label_SEED={SEED}_round={n_round}.pt",
    )
    torch.save(
        predicted_softmax, f"results/predicted_softmax_SEED={SEED}_round={n_round}.pt"
    )
    torch.save(weight, f"results/weight_SEED={SEED}_round={n_round}.pt")
    torch.save(patch_sizes, f"results/patch_sizes_SEED={SEED}_round={n_round}.pt")

    ### merge the top 10% patches with highest BvSB score ###
    all_patch_BvSB = []
    for k, patches in BvSB.items():
        for patch, score in patches.items():
            all_patch_BvSB.append((f"{k}/{patch}", score / patch_sizes[k][patch]))

    all_patch_BvSB.sort(key=lambda x: x[1], reverse=True)
    all_patch_BvSB = all_patch_BvSB[: int(0.1 * len(all_patch_BvSB))]

    merged_nodes_per_frame = {}
    for patch_id, score in all_patch_BvSB:
        video_id, frame_nb, patch_nb = patch_id.split("/")
        frame_id = f"{video_id}/{frame_nb}"
        curr_root = int(patch_nb)

        superpixel_lab = dataset.load_superpixel(frame_id, transform=True)

        # BFS
        adjacent_patches = [curr_root]

        frame_merged_nodes = merged_nodes_per_frame.get(frame_id, {})
        if len(frame_merged_nodes) == 0:
            all_merged_nodes = []
        else:
            all_merged_nodes = list(np.concatenate(list(frame_merged_nodes.values())))
        curr_merged_nodes = []
        while len(adjacent_patches) > 0:
            curr_root = adjacent_patches.pop(0)
            curr_merged_nodes.append(curr_root)
            all_merged_nodes.append(curr_root)
            p = predicted_softmax[frame_id][curr_root]
            for n in np.unique(superpixel_lab):
                if n in all_merged_nodes:
                    continue
                q = predicted_softmax[frame_id][n]
                if (
                    is_adjacent(curr_root, n, superpixel_lab)
                    and distance.jensenshannon(p, q) < 0.1
                ):
                    adjacent_patches.append(n)
                    all_merged_nodes.append(n)
        frame_merged_nodes[int(patch_nb)] = curr_merged_nodes
        merged_nodes_per_frame[frame_id] = frame_merged_nodes

    torch.save(
        merged_nodes_per_frame,
        f"results/merged_nodes_per_frame_SEED={SEED}_round={n_round}.pt",
    )
    #########################################################

    ### select patches ###
    total_pixel_num = sum(weight.values())
    for k, v in weight.items():
        weight[k] = np.exp(-v / total_pixel_num)

    merged_DO_label_per_frame = {}
    final_scores = []
    for k, v in BvSB.items():
        if k in merged_nodes_per_frame:
            merged_nodes = merged_nodes_per_frame[k]
            all_frame_merged_nodes = set(np.concatenate(list(merged_nodes.values())))
            for patch_id, uncertainty in v.items():
                if patch_id in merged_nodes:
                    uncertainty = 0
                    pixel_num = 0
                    merged_DO_label = []
                    for n in merged_nodes[patch_id]:
                        uncertainty += BvSB[k][n]
                        pixel_num += patch_sizes[k][n]
                        merged_DO_label.append(region_dominant_label[k][n])
                    uncertainty /= pixel_num

                    unique, unique_counts = np.unique(
                        merged_DO_label, return_counts=True
                    )
                    merged_DO_label = unique[np.argmax(unique_counts)]

                    # # save merged DO label #
                    DO_label_dict = merged_DO_label_per_frame.get(k, {})
                    DO_label_dict[int(patch_id)] = merged_DO_label
                    merged_DO_label_per_frame[k] = DO_label_dict

                    final_scores.append(
                        (k, uncertainty * weight[merged_DO_label], patch_id)
                    )
                elif patch_id not in all_frame_merged_nodes:
                    final_scores.append(
                        (
                            k,
                            uncertainty * weight[region_dominant_label[k][patch_id]],
                            patch_id,
                        )
                    )

        else:
            for patch_id, uncertainty in v.items():
                final_scores.append(
                    (
                        k,
                        uncertainty * weight[region_dominant_label[k][patch_id]],
                        patch_id,
                    )
                )

    final_scores.sort(key=lambda x: x[1], reverse=True)

    new_labeled = collections.defaultdict(list)
    new_selected_patches = collections.defaultdict(list)
    for i in range(num_query * patch_number):
        frame_id, score, patch_id = final_scores[i]
        video_id, nb = frame_id.split("/")
        nb = nb[len(FRAME_KEYWORD) :]

        if video_id not in new_labeled or nb not in new_labeled[video_id]:
            new_labeled[video_id].append(nb)
        new_selected_patches["/".join([video_id, nb])].append(int(patch_id))

    ### sieving ###
    for frame_id, patches in new_selected_patches.items():
        video_id, nb = frame_id.split("/")
        nb = FRAME_KEYWORD + nb
        frame_id = f"{video_id}/{nb}"
        superpixel_lab = dataset.load_superpixel(frame_id, transform=True)

        for patch_nb in patches:
            if patch_nb in merged_nodes:
                curr_merged_nodes = merged_nodes[patch_nb]
                merged_SP = np.zeros_like(superpixel_lab)
                merged_DO_label = []
                for n in curr_merged_nodes:
                    merged_SP[superpixel_lab == n] = 1
                    merged_DO_label.append(region_dominant_label[frame_id][n])

                unique, unique_counts = np.unique(merged_DO_label, return_counts=True)
                merged_DO_label = unique[np.argmax(unique_counts)]
            else:
                merged_SP = np.zeros_like(superpixel_lab)
                merged_SP[superpixel_lab == patch_nb] = 1
                merged_DO_label = region_dominant_label[frame_id][patch_nb]

            ### sieving ###
            pred_mask = torch.load(
                f"./ML_preds/{frame_id}.pt"
            ).numpy()  # shape = (n_class, h, w)
            DO_label_distribution = pred_mask[
                merged_DO_label, merged_SP.astype(bool)
            ]  # shape = (n_pixel) as much pixel as in merged_SP

            elbow_index, elbow_value = find_elbow(
                sorted(np.unique(DO_label_distribution))
            )  # Adjust the s parameter as needed
            mask_condition = pred_mask[merged_DO_label] < elbow_value
            merged_SP[mask_condition & (merged_SP == 1)] = routes.IGNORE_INDEX

            # save merged SP #
            save_path = f"./merged_SP/{frame_id}"
            if not os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
            torch.save(merged_SP, save_path + f"/{patch_nb}.pt")

    ### add all the labeled single patches in new_selected_patches ###
    for frame_id, patches in new_selected_patches.items():
        video_id, nb = frame_id.split("/")
        nb = FRAME_KEYWORD + nb
        frame_id = f"{video_id}/{nb}"
        if frame_id not in merged_nodes_per_frame:
            continue
        DO_label_dict = merged_DO_label_per_frame[frame_id]
        for root_patch in patches:
            if root_patch in merged_nodes_per_frame[frame_id]:
                for patch_id in merged_nodes_per_frame[frame_id][root_patch]:
                    new_selected_patches[frame_id].append(patch_id)
                    DO_label_dict[patch_id] = DO_label_dict[root_patch]
        merged_DO_label_per_frame[frame_id] = DO_label_dict

    for frame_id, DO_label_dict in merged_DO_label_per_frame.items():
        new_selected_patches[frame_id + "/DO_label"] = DO_label_dict

    return new_labeled, new_selected_patches


def oracle_query():
    # pets v0
    if config["DATASET"] == "pets":
        new_labeled = {
            "72690ef572": ["00005"],
            "7d0ffa68a4": ["00035", "00075", "00060"],
            "de4ddbccb1": ["00120", "00090", "00030"],
            "bc4f71372d": ["00050", "00095"],
            "3e7d2aeb07": ["00050"],
            "3a3bf84b13": ["00030", "00040"],
            "eaec65cfa7": ["00065", "00145"],
            "c37b17a4a9": ["00095", "00080"],
            "9bce4583a2": ["00090"],
            "cd47a23e31": ["00015"],
            "14fd28ae99": ["00135", "00140"],
            "2117fa0c14": ["00175", "00120", "00005"],
            "6b9c43c25a": ["00040"],
            "df7626172f": ["00050"],
            "2a18873352": ["00010"],
            "dd9fe6c6ac": ["00050", "00045", "00015"],
            "b1d7c03927": ["00090"],
            "bda224cb25": ["00055"],
            "8b518ee936": ["00030"],
            "0c817cc390": ["00105"],
            "c3509e728d": ["00030"],
            "cef824a1e1": ["00055"],
            "5fd3da9f68": ["00005"],
            "134d06dbf9": ["00005", "00075"],
            "bf961167a6": ["00010", "00020", "00040"],
            "caa8e97f81": ["00065"],
            "fbe541dd73": ["00065"],
            "5b1df237d2": ["00100", "00080"],
            "71e06dda39": ["00080"],
            "8dcfb878a8": ["00025"],
            "add21ee467": ["00045"],
            "a977126596": ["00005"],
            "04fbad476e": ["00110"],
        }
    # auris v11
    elif config["DATASET"] == "auris":
        new_labeled = {
            "10-03": ["24469", "24619", "24607"],
            "03-06": ["110419", "108819"],
            "03-01": ["56657"],
            "11-01": ["23807"],
            "15-04": ["41807"],
            "13-02": ["42019"],
            "02-00": ["22294"],
        }
    # intuitive v3
    # new_labeled = {'seq_4_': ['144',
    #           '140',
    #           '142',
    #           '145',
    #           '048'],
    #          'seq_5_': ['133'],
    #          'seq_14': ['007', '051', '046'],
    #          'seq_1_': ['039']}

    # intuitive v6
    elif config["DATASET"] == "intuitive":
        new_labeled = {
            "seq_10": ["036", "047", "055", "119", "054"],
            "seq_4_": [
                "068",
                "035",
                "110",
                "139",
                "147",
                "138",
                "142",
                "141",
                "144",
                "140",
                "145",
                "098",
                "072",
            ],
            "seq_12": ["098"],
            "seq_15": ["031", "003"],
            "seq_5_": ["012", "018", "041", "089", "016"],
            "seq_7_": ["030", "036", "000", "080", "025"],
            "seq_1_": ["147", "130", "100", "051", "075", "091", "094"],
            "seq_6_": ["085"],
            "seq_9_": ["135"],
            "seq_13": ["078"],
            "seq_3_": ["030", "090"],
            "seq_14": ["097", "076", "058", "110", "032", "147", "046"],
        }
    return new_labeled
