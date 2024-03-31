from routes import (
    FILE_TYPE,
)
import routes
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from collections import namedtuple
import torchvision.transforms.functional as F
import pickle
from config import config

class DataHandlerPascal(Dataset):
    def __init__(
        self,
        img_path,
        sequences,
        transform=None,
        label_transform=None,
        filter=None,
        labeled_set=True,
        curr_selected_patches=None,
        patch_number=None,
        patch_shape=None,
        dominant_labelling=False,
        merge_SP=False,
    ):
        super(DataHandlerPascal).__init__()
        self.img_path = img_path
        self.data_pool = []
        self.filter = filter
        if filter is not None:
            self.filter = {}
            for k, v in filter.items():
                for nb in v:
                    frame_id = '/'.join([k, nb])
                    self.filter[frame_id] = True

        self.nb_of_patches = 0
        # load labels and frames
        actual_patch_number = (
            (patch_number or 0) ** 2 if patch_shape == "rectangle" else patch_number
        )
        for frame_path in sequences:
            frame_id = "/".join(frame_path.split("/")[-2:])
            frame_id = frame_id[: -len(FILE_TYPE)]
            if self.filter is None:  # test, val, all train sets
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels").replace("jpg", "png"))
                )
            elif labeled_set and frame_id in self.filter:  # train set
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels").replace("jpg", "png"))
                )
                if curr_selected_patches is not None:
                    self.nb_of_patches += len(curr_selected_patches[frame_id])
            elif (
                not labeled_set and frame_id not in self.filter
            ):  # unlabeled set without patches
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels").replace("jpg", "png"))
                )
                if patch_number is not None:
                    self.nb_of_patches += actual_patch_number
            elif (  # unlabeled set with patches
                not labeled_set
                and curr_selected_patches is not None
                and len(curr_selected_patches[frame_id]) < actual_patch_number
            ):
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels").replace("jpg", "png"))
                )
                self.nb_of_patches += actual_patch_number - len(
                    curr_selected_patches[frame_id]
                )
        self.data_pool = np.array(self.data_pool)
        self.transform = transform
        self.label_transform = label_transform
        self.multi_class = True

        self.curr_selected_patches = curr_selected_patches
        self.patch_number = patch_number
        self.patch_shape = patch_shape
        self.return_patches = False
        self.labeled_set = labeled_set
        self.dominant_labelling = dominant_labelling
        self.merge_SP = merge_SP
        self.total_noise = 0

    def transform_x(self, img, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        img = self.transform(Image.fromarray(img))
        tensor_trans = T.ToTensor()
        if toTensor:
            img = tensor_trans(img)
            # img = F.normalize(
            #     img,
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # )
        else:
            img = np.array(img)
        return img

    def transform_y(self, label, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        label = self.label_transform(Image.fromarray(label))
        label = np.array(label)
        if toTensor:
            label = torch.Tensor(label)
            label = label.long()
            return label
        else:
            return label.astype("int")

    def open_path(self, img_path, label_path, name=None, toTensor=True):
        img = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))#.convert("RGB"))
        # label = new_lbl_convert(label)
        if (
            self.patch_number is not None
            and self.curr_selected_patches is not None
            and name is not None
            and self.return_patches == False
        ):
            if self.patch_shape == "rectangle" and self.labeled_set:
                label = self.label_patches(label, name)
            elif self.patch_shape == "superpixel" and self.labeled_set:
                label = self.label_patches_superpixel(label, label_path, name)
        rnd_state = (
            torch.random.get_rng_state()
        )  # get random state for consistent image and label transforms

        if self.return_patches:  # only used for simCLR embedding
            if self.patch_shape == "rectangle":
                img = self.extract_patches(img)
            elif self.patch_shape == "superpixel":
                img = self.extract_superpixel_patches(img, name)
            for i in range(len(img)):
                img[i] = self.transform_x(img[i], rnd_state, toTensor)
        else:
            img = self.transform_x(img, rnd_state, toTensor)
            label = self.transform_y(label, rnd_state, toTensor)

        return img, label

    def extract_patches(self, img):
        img = np.array(img)
        img_patches = []

        patch_size_x = img.shape[0] // self.patch_number
        patch_size_y = img.shape[1] // self.patch_number
        for i in range(self.patch_number):
            for j in range(self.patch_number):
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == self.patch_number - 1:
                    end_x = img.shape[0]
                end_y = start_y + patch_size_y
                if j == self.patch_number - 1:
                    end_y = img.shape[1]

                img_patch = img[start_x:end_x, start_y:end_y]
                img_patches.append(img_patch)

        return img_patches

    def extract_superpixel_patches(self, img, name):
        # name = video_id/frame keyword + frame_number
        img = np.array(img)
        img_patches = []

        superpixel_label = self.load_superpixel(name)
        patch_ids = np.unique(superpixel_label)

        for patch in patch_ids:
            img_patch = pad_superpixel(superpixel_label, img, patch, True, True)
            img_patches.append(img_patch)

        return img_patches

    def label_patches(self, label, name):
        name = (
            name.split("/")[0]
            + "/"
            + name.split("/")[1][len(routes.FRAME_KEYWORD_VOC) :]
        )
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:
            i, j = divmod(patch, self.patch_number)
            patch_size_x = label.shape[0] // self.patch_number
            patch_size_y = label.shape[1] // self.patch_number
            start_x = i * patch_size_x
            start_y = j * patch_size_y

            end_x = start_x + patch_size_x
            if i == self.patch_number - 1:
                end_x = label.shape[0]
            end_y = start_y + patch_size_y
            if j == self.patch_number - 1:
                end_y = label.shape[1]

            label[start_x:end_x, start_y:end_y] = copy_label[
                start_x:end_x, start_y:end_y
            ]
        return label

    def label_patches_superpixel(self, label, label_path, name):
        # name = frame_number
        superpixel_label = self.load_superpixel(name)
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:
            if self.dominant_labelling:
                dominant_label = None
                DO_label_dict = self.curr_selected_patches.get(name + '/DO_label', {})
                if patch in DO_label_dict and self.merge_SP:
                    dominant_label = DO_label_dict[patch]
                else:
                    frame_patch = copy_label[superpixel_label == patch]
                    n = 0
                    for l in np.unique(frame_patch):
                        if np.sum(frame_patch == l) > n:
                            n = np.sum(frame_patch == l)
                            dominant_label = l

                if dominant_label is not None:
                    self.total_noise += np.sum(frame_patch != dominant_label)
                    label[superpixel_label == patch] = dominant_label
                if self.merge_SP:
                    merged_SP = torch.load(f'./merged_SP/{name}/{patch}.pt')
                    label[merged_SP == 255] = 255
            else:
                label[superpixel_label == patch] = copy_label[superpixel_label == patch]
        return label

    def load_superpixel(self, name, transform=False):
        superpixel_path = (
            self.img_path.replace("images", f"superpixels_{self.patch_number}")
            + f"{name}.png"
        )
        superpixel_label = np.array(Image.open(superpixel_path))
        if config['MODEL_ARCH'] == 'vit' and transform:
            superpixel_label = self.transform_y(superpixel_label, torch.random.get_rng_state(), toTensor=False)
        return superpixel_label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path, label_path = self.data_pool[item]
        name = label_path[len(self.img_path) : -len(FILE_TYPE)]
        name = 'images/' + name

        img, label = self.open_path(img_path, label_path, name)
        return img, label, name

    def __len__(self):
        return len(self.data_pool)

class DataHandlerPascalMCAL(DataHandlerPascal):
    def __init__(
        self,
        img_path,
        sequences,
        transform=None,
        label_transform=None,
        filter=None,
        labeled_set=True,
        curr_selected_patches=None,
        patch_number=None,
        patch_shape=None,
        dominant_labelling=False,
        merge_SP=False,
    ):
        super().__init__(
            img_path,
            sequences,
            transform,
            label_transform,
            filter,
            labeled_set,
            curr_selected_patches,
            patch_number,
            patch_shape,
            dominant_labelling,
            merge_SP,
        )

        data_root = img_path[: -len("images/")]
        with open(f'{data_root}/multi_hot_cls_pascal_VOC.pkl', 'rb') as f:
            multi_hot_cls = pickle.load(f)
        self.multi_hot_cls = multi_hot_cls
        self.return_precise_label = False

    def open_path(self, img_path, label_path, name=None, toTensor=True):
        img = np.array(Image.open(img_path))

        label = torch.from_numpy(self.multi_hot_cls[name])
        superpixel_label = self.load_superpixel(name)

        rnd_state = (
            torch.random.get_rng_state()
        )  # get random state for consistent image and label transforms

        img = self.transform_x(img, rnd_state, toTensor)
        superpixel_label = self.transform_y(superpixel_label, rnd_state, toTensor)

        precise_label = 0
        if self.return_precise_label:
            precise_label = np.array(Image.open(label_path))
            precise_label = self.transform_y(precise_label, rnd_state, toTensor)

        return img, label, superpixel_label, precise_label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path, label_path = self.data_pool[item]
        name = label_path[len(self.img_path) : -len(FILE_TYPE)]
        name = 'images/' + name

        img, label, superpixel_label, precise_label = self.open_path(img_path, label_path, name)
        
        name2 = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_VOC):]
        preserving_labels = self.curr_selected_patches[name2]

        sp_mask = torch.from_numpy(np.isin(superpixel_label, preserving_labels))

        return {
            "images": img,
            "labels": label,
            "precise_labels": precise_label,
            "spx": superpixel_label,
            "spmask": sp_mask,
            "names": name,
        }
    def __len__(self):
        return len(self.data_pool)
