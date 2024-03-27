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

### https://github.com/YeLyuUT/UAVidToolKit

UAVID_CLASS_TO_NB = {
    "Clutter": 0,
    "Building": 1,
    "Road": 2,
    "Static_Car": 3,
    "Tree": 4,
    "Vegetation": 5,
    "Human": 6,
    "Moving_Car": 7,
}


class UAVidColorTransformer:
    def __init__(self):
        # color table.
        self.clr_tab = self.createColorTable()
        # id table.
        id_tab = {}
        for k, v in self.clr_tab.items():
            id_tab[k] = self.clr2id(v)
        self.id_tab = id_tab

    def createColorTable(self):
        clr_tab = {}
        clr_tab["Clutter"] = [0, 0, 0]
        clr_tab["Building"] = [128, 0, 0]
        clr_tab["Road"] = [128, 64, 128]
        clr_tab["Static_Car"] = [192, 0, 192]
        clr_tab["Tree"] = [0, 128, 0]
        clr_tab["Vegetation"] = [128, 128, 0]
        clr_tab["Human"] = [64, 64, 0]
        clr_tab["Moving_Car"] = [64, 0, 128]
        return clr_tab

    def colorTable(self):
        return self.clr_tab

    def clr2id(self, clr):
        return clr[0] + clr[1] * 255 + clr[2] * 255 * 255

    # transform to uint8 integer label
    def transform(self, label, dtype=np.int32):
        height, width = label.shape[:2]
        # default value is index of clutter.
        newLabel = np.zeros((height, width), dtype=dtype)
        id_label = label.astype(np.int64)
        id_label = (
            id_label[:, :, 0] + id_label[:, :, 1] * 255 + id_label[:, :, 2] * 255 * 255
        )
        for tid, key in enumerate(self.clr_tab.keys()):
            val = self.id_tab[key]
            mask = id_label == val
            newLabel[mask] = tid
        return newLabel

    # transform back to 3 channels uint8 label
    def inverse_transform(self, label):
        label_img = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
        values = list(self.clr_tab.values())
        for tid, val in enumerate(values):
            mask = label == tid
            label_img[mask] = val
        return label_img


class DataHandlerUAVID(Dataset):
    def __init__(
        self,
        label_path,
        img_path,
        sequences,
        transform=None,
        label_transform=None,
        filter=None,
        labeled_set=True,
        multi_class=False,
        curr_selected_patches=None,
        patch_number=None,
        patch_shape=None,
    ):
        super(DataHandlerUAVID).__init__()
        self.label_path = label_path
        self.data_pool = []
        self.filter = filter

        self.nb_of_patches = 0
        # load labels and frames
        actual_patch_number = (patch_number or 0) ** 2 if patch_shape == 'rectangle' else patch_number
        for seq in sequences:
            # glob is for getting the global path from root to the file
            label_files = sorted(os.listdir(self.label_path + seq))
            img_files = sorted(os.listdir(img_path + seq))
            assert len(label_files) > 0
            for f in label_files:
                if f not in img_files:
                    continue

                frame_nb = f[: -len(FILE_TYPE)]
                frame_id = seq + "/" + frame_nb
                if self.filter is None:  # test, val, all train sets
                    self.data_pool.append(
                        (img_path + seq + "/" + f, self.label_path + seq + "/" + f)
                    )
                elif labeled_set and frame_nb in self.filter[seq]: # train set
                    self.data_pool.append(
                        (img_path + seq + "/" + f, self.label_path + seq + "/" + f)
                    )
                    if curr_selected_patches is not None:
                        self.nb_of_patches += len(curr_selected_patches[frame_id])
                elif not labeled_set and frame_nb not in self.filter[seq]: # unlabeled set without patches
                    self.data_pool.append(
                        (img_path + seq + "/" + f, self.label_path + seq + "/" + f)
                    )
                    if patch_number is not None:
                        self.nb_of_patches += actual_patch_number
                elif ( # unlabeled set with patches
                    not labeled_set
                    and curr_selected_patches is not None
                    and len(curr_selected_patches[frame_id]) < actual_patch_number
                ):
                    self.data_pool.append(
                        (img_path + seq + "/" + f, self.label_path + seq + "/" + f)
                    )
                    self.nb_of_patches += actual_patch_number - len(
                        curr_selected_patches[frame_id]
                    )
        self.data_pool = np.array(self.data_pool)
        self.transform = transform
        self.label_transform = label_transform
        self.color_encoder = UAVidColorTransformer()
        self.multi_class = multi_class

        self.curr_selected_patches = curr_selected_patches
        self.patch_number = patch_number
        self.patch_shape = patch_shape
        self.return_patches = False
        self.labeled_set = labeled_set

    def transform_x(self, img, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        img = self.transform(Image.fromarray(img))
        tensor_trans = T.ToTensor()
        if toTensor:
            img = tensor_trans(img)
        else:
            img = np.array(img)
        return img

    def transform_y(self, label, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        label = self.label_transform(Image.fromarray(label))
        label = np.array(label)
        if toTensor:
            label = torch.Tensor(label)
            if not self.multi_class:
                label = (
                    label == UAVID_CLASS_TO_NB["Road"]
                )  # I chose Road because it appear on all the images
                label = label.float()
            else:
                label = label.long()
            return label
        else:
            label = label == UAVID_CLASS_TO_NB["Road"]
            return label.astype("float")

    def open_path(self, img_path, label_path, name=None, toTensor=True):
        img = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))

        label = self.color_encoder.transform(label, dtype=np.uint8)
        if self.patch_number is not None and self.curr_selected_patches is not None and name is not None and self.return_patches == False:
            if self.patch_shape == 'rectangle' and self.labeled_set:
                label = self.label_patches(label, name)
            elif self.patch_shape == 'superpixel' and self.labeled_set:
                label = self.label_patches_superpixel(label, label_path, name)
        rnd_state = (
            torch.random.get_rng_state()
        )  # get random state for consistent image and label transforms

        if self.return_patches:  # only used for simCLR embedding
            if self.patch_shape == 'rectangle':
                img = self.extract_patches(img)
            elif self.patch_shape == 'superpixel':
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

                img_patch = img[start_x: end_x, start_y: end_y]
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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_UAVID):]
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
            
            label[start_x: end_x, start_y: end_y] = copy_label[start_x: end_x, start_y: end_y]
        return label

    def label_patches_superpixel(self, label, label_path, name):
        # name = video_id/frame keyword + frame_number
        superpixel_label = np.array(Image.open(label_path.replace('labels', f'superpixels_{self.patch_number}')))
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_UAVID):]
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:            
            label[superpixel_label == patch] = copy_label[superpixel_label == patch]
        return label

    def load_superpixel(self, name):
        superpixel_path = self.label_path.replace('labels', f'superpixels_{self.patch_number}') + f'{name}.png'
        superpixel_label = np.array(Image.open(superpixel_path))
        return superpixel_label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path, label_path = self.data_pool[item]
        name = label_path[len(self.label_path) : -len(FILE_TYPE)]

        img, label = self.open_path(img_path, label_path, name)
        return img, label, name

    def __len__(self):
        return len(self.data_pool)
