from routes import (
    FILE_TYPE,
)
import routes
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import pickle
from config import config

AURIS_LABEL_TO_COLORS = {
    "background": [0, 0, 0],
    "REBUS Sheath": [0, 255, 0],
    "needle clear sheath": [255, 0, 0],
    "forceps sheath": [125, 0, 255],
    "needle tip": [255, 125, 0],
    "needle blue sheath": [255, 0, 125],
    # 'needle brush': [255, 125, 125],
    "forceps clamp": [0, 0, 255],
    "REBUS Probe": [255, 255, 0],
    # 'forceps blue': [0, 125, 255]
}

AURIS_LABEL_TO_NUMBERS = {
    "background": 0,
    "REBUS Sheath": 1,
    "needle clear sheath": 2,
    "forceps sheath": 3,
    "needle tip": 4,
    "needle blue sheath": 5,
    # 'needle brush': 6,
    "forceps clamp": 6,
    "REBUS Probe": 7,
    #'forceps blue': 9
}

# # for auris v4
# AURIS_LABEL_TO_NUMBERS = {
#     "background": 0,
#     "REBUS Sheath": 1,
#     "needle clear sheath": 2,
#     "forceps sheath": 3,
#     "needle tip": 4,
#     "needle blue sheath": 5,
#     'needle brush': 6,
#     "forceps clamp": 7,
#     "REBUS Probe": 8,
#     #'forceps blue': 9
# }


class DataHandler(Dataset):
    def __init__(
        self,
        data_path,
        img_trans=None,
        label_trans=None,
        lab_path=routes.LAB_PATH,
        labeled_part=None,
        multi_class=False,
        curr_selected_patches=None,
        patch_number = None,
        patch_shape = None,
        dominant_labelling=False
    ):
        self.data_path = data_path
        if len(data_path) > 0: # this condition and the chose_labeled function caused the error in auris v16 random DO where the length of unlabled set became 0
            self.data_pool = np.concatenate(list(data_path.values()), axis=0)
            if patch_number is not None and curr_selected_patches is not None and labeled_part is not None:
                self.data_pool = np.concatenate(list(data_path.values()) + list(labeled_part.values()), axis=0)
        else:
            self.data_pool = []
        self.img_trans = img_trans
        self.label_trans = label_trans
        self.lab_path = lab_path
        self.multi_class = multi_class
        
        self.curr_selected_patches = curr_selected_patches
        self.patch_number = patch_number
        patch_number = patch_number if patch_shape == 'superpixel' else (patch_number or 0) ** 2
        self.return_patches = False
        
        self.nb_of_patches = 0
        delete_inds = []
        if curr_selected_patches is not None:
            for i, (img_path, lab_path) in enumerate(self.data_pool):
                video_id = img_path.split('/')[-2]
                frame_nb = img_path.split('/')[-1].split('.')[0]
                frame_id = video_id + '/' + frame_nb[len(routes.FRAME_KEYWORD):]
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
        self.patch_shape = patch_shape
        self.dominant_labelling = dominant_labelling
        self.total_noise = 0

    def transform_x(self, x, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        tensor_trans = T.ToTensor()
        x = self.img_trans(Image.fromarray(x))
        if toTensor:
            x = tensor_trans(x)
        else:
            x = np.array(x)
        return x
    
    def transform_y(self, y, rnd_state, toTensor=True):
        torch.random.set_rng_state(rnd_state)
        y = self.label_trans(Image.fromarray(y))
        y = np.array(y)

        if toTensor:
            y = torch.Tensor(y)
            y = y.float() if not self.multi_class else y.long()
        else:
            y = y.astype("float") if not self.multi_class else y.astype("int")
        return y
      
    def open_path(self, img_path, mask_path, name=None, toTensor=True):
        x = Image.open(img_path)
        y = Image.open(mask_path)
        if not self.multi_class:
            y = y.convert("L")
            y = y != 0

        x = np.array(x)
        y = np.array(y)

        if self.multi_class:
            seg_map = np.zeros(y.shape[:2], dtype=np.int32)
            for label_name, label_index in AURIS_LABEL_TO_NUMBERS.items():
                color = AURIS_LABEL_TO_COLORS[label_name]
                seg_map[(y == color).all(axis=-1)] = label_index
            y = seg_map
            
        if self.patch_number is not None and self.curr_selected_patches is not None and name is not None and self.return_patches == False:
            if self.patch_shape == 'rectangle' and self.labeled_part is None:
                y = self.label_patches(y, name)
            elif self.patch_shape == 'superpixel' and self.labeled_part is None:
                y = self.label_patches_superpixel(y, name)

        rnd_state = torch.random.get_rng_state()
        if self.return_patches:
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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD):]
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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD):]
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
                    noise = np.sum(frame_patch != dominant_label)
                    self.total_noise += noise
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
    
    def __getitem__(self, index):
        name = self.data_pool[index][1]
        name = name[len(self.lab_path): -len(FILE_TYPE)]
        x, y = self.open_path(self.data_pool[index][0], self.data_pool[index][1], name)

        return x, y, name

    def __len__(self):
        return len(self.data_pool)
    
class DataHandlerMCAL(DataHandler):
    def __init__(
        self,
        data_path,
        img_trans=None,
        label_trans=None,
        lab_path=routes.LAB_PATH,
        labeled_part=None,
        multi_class=False,
        curr_selected_patches=None,
        patch_number = None,
        patch_shape = None,
        dominant_labelling=False
    ):
        super().__init__(
            data_path,
            img_trans,
            label_trans,
            lab_path,
            labeled_part,
            multi_class,
            curr_selected_patches,
            patch_number,
            patch_shape,
            dominant_labelling
        )
        
        data_root = lab_path[: -len("labels/")]
        with open(f'{data_root}/multi_hot_cls_auris.pkl', 'rb') as f:
            multi_hot_cls = pickle.load(f)
        self.multi_hot_cls = multi_hot_cls
        self.return_precise_label = False

    def open_path(self, img_path, mask_path, name=None, toTensor=True):
        x = Image.open(img_path)
        x = np.array(x)
        
        label = torch.from_numpy(self.multi_hot_cls[name])
        superpixel_label = self.load_superpixel(name)

        rnd_state = torch.random.get_rng_state()
        x = self.transform_x(x, rnd_state, toTensor)
        superpixel_label = self.transform_y(superpixel_label, rnd_state, toTensor)

        precise_label = 0
        if self.return_precise_label:
            precise_label = Image.open(mask_path)
            precise_label = np.array(precise_label)
            seg_map = np.zeros(precise_label.shape[:2], dtype=np.int32)
            for label_name, label_index in AURIS_LABEL_TO_NUMBERS.items():
                color = AURIS_LABEL_TO_COLORS[label_name]
                seg_map[(precise_label == color).all(axis=-1)] = label_index
            precise_label = seg_map
            precise_label = self.transform_y(precise_label, rnd_state, toTensor)
        return x, label, superpixel_label, precise_label
   
    def __getitem__(self, index):
        name = self.data_pool[index][1]
        name = name[len(self.lab_path): -len(FILE_TYPE)]
        
        x, y, superpixel_label, precise_label = self.open_path(self.data_pool[index][0], self.data_pool[index][1], name)

        name2 = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD):]
        preserving_labels = self.curr_selected_patches[name2]

        sp_mask = torch.from_numpy(np.isin(superpixel_label, preserving_labels))

        return {
            "images": x,
            "labels": y,
            "precise_labels": precise_label,
            "spx": superpixel_label,
            "spmask": sp_mask,
            "names": name,
        } 

    def __len__(self):
        return len(self.data_pool)