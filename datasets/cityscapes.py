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
from utils import pad_superpixel
from collections import namedtuple
import pickle
from config import config

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]


def new_lbl_convert(label):
    color_to_trainId = {
        l.color: l.trainId for l in labels if not l.ignoreInEval and l.trainId != 255
    }
    label = np.array(label)
    h, w, _ = label.shape
    label_convert = 255 * np.ones((h, w), dtype=np.uint8)

    for color, trainId in color_to_trainId.items():
        mask = np.all(label == np.array(color).reshape(1, 1, 3), axis=-1)
        label_convert[mask] = trainId

    return label_convert

def convert_label_to_color(label):
    trainId_to_color = {
        l.trainId: l.color for l in labels if not l.ignoreInEval and l.trainId != 255
    }
    trainId_to_color[255] = (255, 255, 255)
    label = np.array(label)
    h, w = label.shape
    label_convert = np.zeros((h, w, 3), dtype=np.uint8)

    for trainId, color in trainId_to_color.items():
        mask = label == trainId
        label_convert[mask] = color

    return label_convert

class DataHandlerCityscapes(Dataset):
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
    ):
        super(DataHandlerCityscapes).__init__()
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
                    (frame_path, frame_path.replace("images", "labels"))
                )
            elif labeled_set and frame_id in self.filter:  # train set
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels"))
                )
                if curr_selected_patches is not None:
                    self.nb_of_patches += len(curr_selected_patches[frame_id])
            elif (
                not labeled_set and frame_id not in self.filter
            ):  # unlabeled set without patches
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels"))
                )
                if patch_number is not None:
                    self.nb_of_patches += actual_patch_number
            elif (  # unlabeled set with patches
                not labeled_set
                and curr_selected_patches is not None
                and len(curr_selected_patches[frame_id]) < actual_patch_number
            ):
                self.data_pool.append(
                    (frame_path, frame_path.replace("images", "labels"))
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

        self.total_noise = 0

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
            + name.split("/")[1][len(routes.FRAME_KEYWORD_CITY) :]
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
        # name = video_id/frame keyword + frame_number
        superpixel_label = np.array(
            Image.open(label_path.replace("labels", f"superpixels_{self.patch_number}"))
        )
        name = (
            name.split("/")[0]
            + "/"
            + name.split("/")[1][len(routes.FRAME_KEYWORD_CITY) :]
        )
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:
            if self.dominant_labelling and len(self.curr_selected_patches[name]) < self.patch_number:
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
        superpixel_path = (
            self.img_path.replace("images", f"superpixels_{self.patch_number}")
            + f"{name}.png"
        )
        superpixel_label = np.array(Image.open(superpixel_path))
        if config['MODEL_ARCH'] == 'vit' and transform:
            superpixel_label = self.transform_y(superpixel_label, torch.random.get_rng_state(), False)
        return superpixel_label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path, label_path = self.data_pool[item]
        name = label_path[len(self.img_path) : -len(FILE_TYPE)]

        img, label = self.open_path(img_path, label_path, name)
        return img, label, name

    def __len__(self):
        return len(self.data_pool)


class DataHandlerCityscapesMCAL(DataHandlerCityscapes):
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
        )

        data_root = img_path[: -len("images/")]
        with open(f'{data_root}/multi_hot_cls_cityscapes.pkl', 'rb') as f:
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
            precise_label = np.array(Image.open(label_path))#.convert("RGB"))
            precise_label = self.transform_y(precise_label, rnd_state, toTensor)

        return img, label, superpixel_label, precise_label

    def __getitem__(self, item):
        # ToDo check if dataloading and label decoding is a bottleneck, if yes perform pre-loading
        # set seed for consistent DA
        img_path, label_path = self.data_pool[item]
        name = label_path[len(self.img_path) : -len(FILE_TYPE)]

        img, label, superpixel_label, precise_label = self.open_path(img_path, label_path, name)
        
        name2 = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_CITY):]
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