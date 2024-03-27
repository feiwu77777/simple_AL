from routes import (
    FILE_TYPE,
)
import routes
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
import json
import glob
from config import img_size_intuitive as img_size
import pickle
from config import config

INSTRUMENT_CLASS = [1, 2, 3, 6, 7, 8, 9, 11]

intuitive_class_names = {0: 'background',
                1: 'shaft',
                2: 'clasper',
                3: 'wrist',
                4: 'kidney-parenchyma',
                5: 'covered-kidney',
                6: 'thread',
                7: 'clamps',
                8: 'needle',
                9: 'suction',
                10: 'intestine',
                11: 'US probe'}

class DataHandlerIntuitive(Dataset):
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
        patch_number=0,
        patch_shape=None,
        pixel_sampling=False,
        dominant_labelling=False,
    ):
        super(DataHandlerIntuitive).__init__()
        self.data_pool = []
        self.filter = filter

        self.lab_path = label_path
        # load labels and frames
        ind_keyword = len(self.lab_path + routes.CLASS_ID_TYPE_INTUITIVE)
        self.nb_of_patches = 0

        actual_patch_number = (patch_number or 0) ** 2 if patch_shape == 'rectangle' else patch_number
        for seq in sequences:
            files = sorted(glob.glob(os.path.join(self.lab_path, seq, "*.png")))
            img_files = os.listdir(os.path.join(img_path, seq))
            assert len(files) > 0
            for f in files:
                if f[ind_keyword:] not in img_files:
                    continue
                
                frame_nb = f[ind_keyword + len(routes.FRAME_KEYWORD_INTUITIVE) : -len(FILE_TYPE)]
                frame_id = seq + "/" + frame_nb
                if self.filter is None: # test, val, all train sets
                    self.data_pool.append((f.replace("labels", "images"), f))
                elif ( # train set
                    labeled_set
                    and frame_nb in self.filter[seq]
                ):
                    self.data_pool.append((f.replace("labels", "images"), f))
                    if curr_selected_patches is not None:
                        self.nb_of_patches += len(curr_selected_patches[frame_id])
                elif ( # unlabeled set without patches
                    not labeled_set
                    and frame_nb not in self.filter[seq]
                ):
                    self.data_pool.append((f.replace("labels", "images"), f))
                    if patch_number is not None:
                        self.nb_of_patches += actual_patch_number
                elif ( # unlabeled set with patches
                    not labeled_set
                    and curr_selected_patches is not None
                    and len(curr_selected_patches[frame_id]) < actual_patch_number # if all patches are selected, then the image is fully labeled
                ):
                    self.data_pool.append((f.replace("labels", "images"), f))
                    self.nb_of_patches += actual_patch_number - len(curr_selected_patches[frame_id])
                elif ( # unalabeled set with pixel sampling
                    not labeled_set
                    and pixel_sampling
                    and os.path.exists(f.replace('labels', 'partial_labels').replace('.png', '.pth'))
                ):
                    self.data_pool.append((f.replace("labels", "images"), f))

        self.data_pool = np.array(self.data_pool)
        # map rgb labes to unique class labels
        basepath = label_path[: -len("labels/")]
        with open(os.path.join(basepath, "labels.json"), "r") as f:
            label_look_up = json.load(f)
        # Get keys and values
        k = np.array([k["color"] for k in label_look_up])
        self.decode_v = np.array([k["classid"] for k in label_look_up])
        self.class_labels = [k["name"] for k in label_look_up]
        # Setup scale array for dimensionality reduction
        s = 256 ** np.arange(3)
        # Reduce k to 1D
        self.decode_k1D = k.dot(s)
        # Get sorted k1D and correspondingly re-arrange the values array
        self.decode_sidx = self.decode_k1D.argsort()
        self.transform = transform
        self.label_transform = label_transform
        self.multi_class = multi_class

        self.curr_selected_patches = curr_selected_patches
        self.patch_number = patch_number
        self.patch_shape = patch_shape
        self.return_patches = False

        self.pixel_sampling = pixel_sampling
        self.labeled_set = labeled_set
        self.dominant_labelling = dominant_labelling

        self.total_noise = 0

    def _rgbDecode(self, rgblabel):
        rgblabel = rgblabel.dot(256 ** np.arange(3))
        return self.decode_v[
            self.decode_sidx[
                np.searchsorted(self.decode_k1D, rgblabel, sorter=self.decode_sidx)
            ]
        ].astype(np.uint8)

    def transform_x(self, img_left, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        img_left = self.transform(Image.fromarray(img_left))
        tensor_trans = T.ToTensor()
        if toTensor:
            img_left = tensor_trans(img_left)
        else:
            img_left = np.array(img_left)
        return img_left
    
    def transform_y(self, label, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        label = self.label_transform(Image.fromarray(label))
        label = np.array(label)
        if toTensor:
            label = torch.Tensor(label)
            if not self.multi_class:
                label = torch.isin(label, torch.tensor(INSTRUMENT_CLASS))
                label = label.float()
            else:
                label = label.long()
            return label
        else:
            # label = np.isin(label, INSTRUMENT_CLASS)
            return label.astype("int")

    def open_path(self, left_path, label_path, name=None, toTensor=True):
        img_left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_RGB2BGR)
        label = self._rgbDecode(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_RGB2BGR))

        img_left = cv2.resize(img_left, (img_size['IMG_SIZE'], img_size['IMG_SIZE']))
        label = cv2.resize(label, (img_size['IMG_SIZE'], img_size['IMG_SIZE']), interpolation=cv2.INTER_NEAREST)

        # if self.pixel_sampling and os.path.exists(label_path.replace('labels', 'partial_labels').replace('.png', '.pth')):
        #     partial_label_ind = torch.load(label_path.replace('labels', 'partial_labels').replace('.png', '.pth'), map_location='cpu').numpy()
        #     partial_label = np.zeros_like(label)
        #     partial_label.fill(routes.IGNORE_INDEX)
        #     partial_label[partial_label_ind] = label[partial_label_ind]
        #     label = partial_label
        # elif self.pixel_sampling and not self.labeled_set:
        #     label = np.zeros_like(label)
        #     label.fill(routes.IGNORE_INDEX)

        if self.patch_number is not None and self.curr_selected_patches is not None and name is not None and self.return_patches == False:
            if self.patch_shape == 'rectangle' and self.labeled_set:
                label = self.label_patches(label, name)
            elif self.patch_shape == 'superpixel' and self.labeled_set:
                label = self.label_patches_superpixel(label, label_path, name)
        #return img_left, label
        rnd_state = (torch.random.get_rng_state())  # get random state for consistent image and label transforms
        if self.return_patches: # only used for simCLR embedding
            if self.patch_shape == 'rectangle':
                img_left = self.extract_rectangle_patches(img_left)
            elif self.patch_shape == 'superpixel':
                img_left = self.extract_superpixel_patches(img_left, name)
            for i in range(len(img_left)):
                img_left[i] = self.transform_x(img_left[i], rnd_state, toTensor)
        else:
            img_left = self.transform_x(img_left, rnd_state, toTensor)
            label = self.transform_y(label, rnd_state, toTensor)

        return img_left, label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        left_path, label_path = self.data_pool[item]
        name = label_path[len(self.lab_path) : -len(FILE_TYPE)]
        img_left, label = self.open_path(left_path, label_path, name)
        return img_left, label, name

    def extract_rectangle_patches(self, img):
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
        # name = video_id/frame keyword + frame_number
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_INTUITIVE):]
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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_INTUITIVE):]
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:
            if self.dominant_labelling:
                frame_patch = copy_label[superpixel_label == patch]
                n = 0
                dominant_label = None
                for l in np.unique(frame_patch):
                    if np.sum(frame_patch == l) > n:
                        n = np.sum(frame_patch == l)
                        dominant_label = l
                if dominant_label is not None:
                    self.total_noise += np.sum(frame_patch != dominant_label)
                    label[superpixel_label == patch] = dominant_label
            else:       
                label[superpixel_label == patch] = copy_label[superpixel_label == patch]
        return label

    def load_superpixel(self, name, transform=False):
        superpixel_path = self.lab_path.replace('labels', f'superpixels_{self.patch_number}') + f'{name}.png'
        superpixel_label = np.array(Image.open(superpixel_path))
        if config['MODEL_ARCH'] == 'vit' and transform:
            superpixel_label = self.transform_y(superpixel_label, torch.random.get_rng_state(), False)
        return superpixel_label

    def get_class_labels(self):
        l = {}
        for i, label in enumerate(self.class_labels):
            l[i] = label
        return l

    def __len__(self):
        return len(self.data_pool)


class DataHandlerIntuitiveMCAL(DataHandlerIntuitive):
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
        patch_number=0,
        patch_shape=None,
        pixel_sampling=False,
        dominant_labelling=False,
    ):
        super().__init__(
            label_path,
            img_path,
            sequences,
            transform,
            label_transform,
            filter,
            labeled_set,
            multi_class,
            curr_selected_patches,
            patch_number,
            patch_shape,
            pixel_sampling,
            dominant_labelling,
        )
        
        data_root = label_path[: -len("labels/")]
        with open(f'{data_root}/multi_hot_cls_intuitive.pkl', 'rb') as f:
            multi_hot_cls = pickle.load(f)
        self.multi_hot_cls = multi_hot_cls
        self.return_precise_label = False

    def open_path(self, left_path, label_path, name=None, toTensor=True):
        img_left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_RGB2BGR)
        img_left = cv2.resize(img_left, (img_size['IMG_SIZE'], img_size['IMG_SIZE']))

        label = torch.from_numpy(self.multi_hot_cls[name])

        superpixel_label = self.load_superpixel(name)
        #return img_left, label
        rnd_state = (torch.random.get_rng_state())  # get random state for consistent image and label transforms
        img_left = self.transform_x(img_left, rnd_state, toTensor)
        superpixel_label = self.transform_y(superpixel_label, rnd_state, toTensor)

        precise_label = 0
        if self.return_precise_label:
            precise_label = self._rgbDecode(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_RGB2BGR))
            precise_label = cv2.resize(precise_label, (img_size['IMG_SIZE'], img_size['IMG_SIZE']), interpolation=cv2.INTER_NEAREST)
            precise_label = self.label_patches_superpixel(precise_label, label_path, name)
            precise_label = self.transform_y(precise_label, rnd_state, toTensor)
        return img_left, label, superpixel_label, precise_label

    def __getitem__(self, item):
        left_path, label_path = self.data_pool[item]
        name = label_path[len(self.lab_path) : -len(FILE_TYPE)]
        
        img_left, label, superpixel_label, precise_label = self.open_path(left_path, label_path, name)

        name2 = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_INTUITIVE):]
        preserving_labels = self.curr_selected_patches[name2]

        sp_mask = torch.from_numpy(np.isin(superpixel_label, preserving_labels))

        return {
            "images": img_left,
            "labels": label,
            "precise_labels": precise_label,
            "spx": superpixel_label,
            "spmask": sp_mask,
            "names": name,
        }

    def __len__(self):
        return len(self.data_pool)


class DataHandlerIntuitiveMCALPseudoLabel(DataHandlerIntuitive):
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
        patch_number=0,
        patch_shape=None,
        pixel_sampling=False,
        dominant_labelling=False,
    ):
        super().__init__(
            label_path,
            img_path,
            sequences,
            transform,
            label_transform,
            filter,
            labeled_set,
            multi_class,
            curr_selected_patches,
            patch_number,
            patch_shape,
            pixel_sampling,
            dominant_labelling,
        )

        data_root = label_path[: -len("labels/")]
        with open(f'{data_root}/multi_hot_cls_intuitive.pkl', 'rb') as f:
            multi_hot_cls = pickle.load(f)
        self.multi_hot_cls = multi_hot_cls

    def open_path(self, left_path, label_path, name=None, toTensor=True):
        img_left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_RGB2BGR)
        img_left = cv2.resize(img_left, (img_size['IMG_SIZE'], img_size['IMG_SIZE']))

        label_precise = self._rgbDecode(cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_RGB2BGR))
        label_precise = cv2.resize(label_precise, (img_size['IMG_SIZE'], img_size['IMG_SIZE']), interpolation=cv2.INTER_NEAREST)
        label_precise = torch.from_numpy(label_precise)

        label = torch.from_numpy(self.multi_hot_cls[name])

        superpixel_label = self.load_superpixel(name)
        superpixel_label = torch.from_numpy(superpixel_label).long()
        img_left = self.transform(Image.fromarray(img_left))

        return img_left, label, label_precise, superpixel_label

    def __getitem__(self, item):
        left_path, label_path = self.data_pool[item]
        name = label_path[len(self.lab_path) : -len(FILE_TYPE)]
        img_left, label, label_precise, superpixel_label = self.open_path(left_path, label_path, name)

        name2 = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_INTUITIVE):]
        preserving_labels = torch.tensor(self.curr_selected_patches[name2] if name2 in self.curr_selected_patches else [])

        valid_preserving_labels = preserving_labels[label[preserving_labels].sum(dim=1) != 0]
        sp_mask = torch.isin(superpixel_label, valid_preserving_labels)

        return {
            "image_list": img_left,
            "labels": label_precise,
            "target": label,
            "spx": superpixel_label,
            "spmask": sp_mask,
            "names": name,
        }

    def __len__(self):
        return len(self.data_pool)
    

class DataHandlerIntuitiveMCALStage2(DataHandlerIntuitive):
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
        patch_number=0,
        patch_shape=None,
        pixel_sampling=False,
        dominant_labelling=False,
    ):
        super().__init__(
            label_path,
            img_path,
            sequences,
            transform,
            label_transform,
            filter,
            labeled_set,
            multi_class,
            curr_selected_patches,
            patch_number,
            patch_shape,
            pixel_sampling,
            dominant_labelling,
        )

        self.pseudo_label_path = './pseudo_labels/'

    def open_path(self, left_path, label_path, name=None, toTensor=True):
        img_left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_RGB2BGR)
        img_left = cv2.resize(img_left, (img_size['IMG_SIZE'], img_size['IMG_SIZE']))

        label = np.array(Image.open(self.pseudo_label_path + name + '.png'))
        
        rnd_state = (torch.random.get_rng_state())  # get random state for consistent image and label transforms
        img_left = self.transform_x(img_left, rnd_state, toTensor)
        label = self.transform_y(label, rnd_state, toTensor)

        return img_left, label

    def __len__(self):
        return len(self.data_pool)