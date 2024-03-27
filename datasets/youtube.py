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
import json
from config import img_size_youtube as img_size
from utils import pad_superpixel

PETS_CLASS_TO_NB = {
    'background': 0,
    'person': 1,
    'hand': 2,
    'dog': 3,
    'cat': 4,
    'rabbit': 5,
    'parrot': 6
}
YOUTUBE_NB_TO_CLASS = {v: k for k, v in PETS_CLASS_TO_NB.items()}

PETS_ORIGINAL_CLASS_TO_NB = {
    1: 1,
    4: 6,
    8: 3,
    11: 2,
    12: 5,
    14: 4,
}

class DataHandlerYoutube(Dataset):
    def __init__(
        self,
        data_path,
        img_trans=None,
        label_trans=None,
        img_path=routes.IMG_PATH_PARROTS,
        path=routes.DATAPATH_PARROTS,
        category="parrot",
        labeled_part=None,
        multi_class=False,
        curr_selected_patches=None,
        patch_number = None,
        patch_shape = None,
    ):
        self.data_path = data_path
        if len(data_path) > 0:
            self.data_pool = np.concatenate(list(data_path.values()), axis=0)
            if patch_number is not None and curr_selected_patches is not None and labeled_part is not None:
                self.data_pool = np.concatenate(list(data_path.values()) + list(labeled_part.values()), axis=0)
        else:
            self.data_pool = []
        self.img_trans = img_trans
        self.label_trans = label_trans
        self.img_path = img_path
        if category != 'youtube_VIS':
            self.nb_to_name = json.load(open(path + "nb_to_name.json", "r"))
        self.category = category
        self.multi_class = multi_class

        self.patch_number = patch_number
        patch_number = patch_number if patch_shape == 'superpixel' else (patch_number or 0) ** 2
        
        self.curr_selected_patches = curr_selected_patches
        self.patch_shape = patch_shape
        self.return_patches = False

        self.nb_of_patches = 0
        delete_inds = []
        if curr_selected_patches is not None:
            for i, (img_path, lab_path) in enumerate(self.data_pool):
                video_id = img_path.split('/')[-2]
                frame_nb = img_path.split('/')[-1].split('.')[0]
                frame_id = video_id + '/' + frame_nb[len(routes.FRAME_KEYWORD_YOUTUBE):]
                if frame_id in curr_selected_patches and labeled_part is None: # train set
                    self.nb_of_patches += len(curr_selected_patches[frame_id])
                elif frame_id in curr_selected_patches and labeled_part is not None: # unlabeled set with patches
                    unlabeled_patches_nb = patch_number - len(curr_selected_patches[frame_id])
                    self.nb_of_patches += unlabeled_patches_nb
                    if unlabeled_patches_nb == 0:
                        delete_inds.append(i)
                elif frame_id not in curr_selected_patches and labeled_part is not None: # unlabeled set without patches
                    self.nb_of_patches += patch_number

        self.data_pool = np.delete(self.data_pool, delete_inds, axis=0)
        self.labeled_part = labeled_part

    def transform_x(self, x, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        tensor_trans = T.ToTensor()
        # x = self.img_trans(Image.fromarray(x))
        x = self.img_trans(x)
        if toTensor:
            x = tensor_trans(x)
        else:
            x = np.array(x)
        return x
    
    def transform_y(self, y, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        # y = self.label_trans(Image.fromarray(y))
        y = self.label_trans(y)
        y = np.array(y)
        if toTensor:
            y = torch.Tensor(y)
            y = y.float() if not self.multi_class else y.long()
        else:
            y = y.astype("float")
        return y
        
    def open_path(self, img_path, mask_path, name=None, toTensor=True):
        x = Image.open(img_path)

        if self.category == "youtube_VIS":
            y = Image.open(mask_path.replace('masks', 'labels') + '.png')
        else:
            y = np.zeros((x.size[1], x.size[0]))
            if os.path.isdir(mask_path):
                masks = []
                for m in os.listdir(mask_path):
                    if m[0] == ".":
                        continue
                    if isinstance(m, bytes):
                        m = m.decode("utf-8")
                    masks.append(m)

                masks = sorted(masks)
                label = np.load(mask_path.replace("masks", "labels") + ".npy")
                if self.multi_class:
                    for i, instance in enumerate(label):
                        class_mask = np.array(Image.open(mask_path + '/' + masks[i]))
                        y[class_mask > 0] = PETS_ORIGINAL_CLASS_TO_NB[instance]
                else:
                    for i, instance in enumerate(label):
                        if self.nb_to_name[str(instance)] == self.category or self.nb_to_name[str(instance)] == self.category[:-1]:
                            y += np.array(Image.open(mask_path + "/" + masks[i]))
                
            if not self.multi_class:
                y[y > 0] = 1
            y = Image.fromarray(y.astype(np.uint8))

        # x = x.resize((img_size['IMG_SIZE2'], img_size['IMG_SIZE']))
        # y = y.resize((img_size['IMG_SIZE2'], img_size['IMG_SIZE']), Image.NEAREST)

        # x = np.array(x)
        # y = np.array(y)

        if self.patch_number is not None and self.curr_selected_patches is not None and name is not None and self.return_patches == False:
            if self.patch_shape == 'rectangle' and self.labeled_part is None:
                y = self.label_patches(y, name)
            elif self.patch_shape == 'superpixel' and self.labeled_part is None:
                y = self.label_patches_superpixel(y, name)
        #return img_left, label
        rnd_state = torch.random.get_rng_state()  # get random state for consistent image and label transforms
        if self.return_patches: # only used for simCLR embedding
            if self.patch_shape == 'rectangle':
                x = self.extract_patches(x)
            elif self.patch_shape == 'superpixel':
                x = self.extract_superpixel_patches(x, name)
            for i in range(len(x)):
                x[i] = self.transform_x(x[i], rnd_state, toTensor)
        else:
            x = self.transform_x(x, rnd_state, toTensor)
            y = self.transform_y(y, rnd_state, toTensor)

        return x, y

    def __getitem__(self, index):
        name = self.data_pool[index][0]
        name = name[len(self.img_path) : -len(FILE_TYPE)]
        x, y = self.open_path(self.data_pool[index][0], self.data_pool[index][1], name)

        return x, y, name

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
            img_patch = pad_superpixel(superpixel_label, img, patch)
            img_patches.append(img_patch)

        return img_patches
    
    def label_patches(self, label, name):
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_YOUTUBE):]
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

    def label_patches_superpixel(self, label, name):
        superpixel_label = self.load_superpixel(name)
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_YOUTUBE):]
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:            
            label[superpixel_label == patch] = copy_label[superpixel_label == patch]
        return label

    def load_superpixel(self, name):
        superpixel_path = self.img_path.replace('images', f'superpixels_{self.patch_number}') + f'{name}.png'
        superpixel_label = np.array(Image.open(superpixel_path))
        return superpixel_label

    def __len__(self):
        return len(self.data_pool)