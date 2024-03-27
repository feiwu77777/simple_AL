from routes import (
    FILE_TYPE,
)
import routes
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from PIL import Image
import json
from torchvision import transforms as T

class A2D2ColorTransformer:
  def __init__(self, color_table):
    # color table.
    self.clr_tab = color_table
    # id table.
    id_tab = {}
    for k, v in self.clr_tab.items():
        id_tab[k] = self.clr2id(v)
    self.id_tab = id_tab

  def colorTable(self):
    return self.clr_tab
   
  def clr2id(self, clr):
    return clr[0]+clr[1]*255+clr[2]*255*255

  #transform to uint8 integer label
  def transform(self,label, dtype=np.int32):
    height,width = label.shape[:2]
    # default value is index of clutter.
    newLabel = np.zeros((height, width), dtype=dtype)
    id_label = label.astype(np.int64)
    id_label = id_label[:,:,0]+id_label[:,:,1]*255+id_label[:,:,2]*255*255
    for tid,key in enumerate(self.clr_tab.keys()):
      val = self.id_tab[key]
      mask = (id_label == val)
      newLabel[mask] = tid
    return newLabel

  #transform back to 3 channels uint8 label
  def inverse_transform(self, label):
    label_img = np.zeros(shape=(label.shape[0], label.shape[1],3),dtype=np.uint8)
    values = list(self.clr_tab.values())
    for tid,val in enumerate(values):
      mask = (label==tid)
      label_img[mask] = val
    return label_img

def hex_to_rgb(hex_color):
    # Remove the hash (#) at the front of the string if it exists
    hex_color = hex_color.lstrip('#')

    # Convert the string to a tuple of integers
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def lbl_convert(label):
    DD = {
        0: ['Speed bumper',
            'Zebra crossing',
            'RD restricted area',
            'Drivable cobblestone',
            'Slow drive area',
            'RD normal street'],
        1: ['Car 1', 'Car 2', 'Car 3', 'Car 4'],
        2: ['Bicycle 1', 'Bicycle 2', 'Bicycle 3', 'Bicycle 4'],
        3: ['Pedestrian 1', 'Pedestrian 2', 'Pedestrian 3'],
        4: ['Truck 1', 'Truck 2', 'Truck 3'],
        5: ['Traffic signal 1', 'Traffic signal 2', 'Traffic signal 3'],
        6: ['Traffic sign 1', 'Traffic sign 2', 'Traffic sign 3'],
        7: ['Non-drivable street', 'Sidewalk'],
        8: ['Nature object'],
        9: ['Sky'],
        10: ['Buildings'],
        11: ['Small vehicles 1', 'Small vehicles 2', 'Small vehicles 3'],
        12: ['Road blocks', 'Grid structure'],
        13: ['Utility vehicle 1', 'Utility vehicle 2'],
        14: ['Sidebars', 'Poles', 'Signal corpus'],
        15: ['Solid line', 'Painted driv. instr.', 'Dashed line'],
        16: ['Traffic guide obj.'],
        17: ['Curbstone'],
        18: ['Irrelevant signs',
        'Tractor',
        'Obstacles / trash',
        'Animals',
        'Electronic traffic',
        'Parking area',
        'Ego car',
        'Blurred area',
        'Rain dirt']
    }

    a2d2_color_seg = {
            "#ff0000": 1, "#c80000": 1, "#960000": 1, "#800000": 1, "#b65906": 2, "#963204": 2,
            "#5a1e01": 2, "#5a1e1e": 2, "#cc99ff": 3, "#bd499b": 3, "#ef59bf": 3, "#ff8000": 4,
            "#c88000": 4, "#968000": 4, "#00ff00": 11, "#00c800": 11, "#009600": 11, "#0080ff": 5,
            "#1e1c9e": 5, "#3c1c64": 5, "#00ffff": 6, "#1edcdc": 6, "#3c9dc7": 6, "#ffff00": 13,
            "#ffffc8": 13, "#e96400": 14, "#6e6e00": 0, "#808000": 17, "#ffc125": 15, "#400040": 18,
            "#b97a57": 12, "#000064": 18, "#8b636c": 7, "#d23273": 0, "#ff0080": 18, "#fff68f": 14,
            "#960096": 0, "#ccff99": 18, "#eea2ad": 12, "#212cb1": 14, "#b432b4": 0, "#ff46b9": 18,
            "#eee9bf": 0, "#93fdc2": 8, "#9696c8": 18, "#b496c8": 7, "#48d1cc": 18, "#c87dd2": 15,
            "#9f79ee": 16, "#8000ff": 15, "#ff00ff": 0, "#87ceff": 9, "#f1e6ff": 10, "#60458f": 18,
            "#352e52": 18
        }
    
    label = np.array(label)
    label_convert = np.zeros_like(label, dtype=np.uint8)

    for key, value in a2d2_color_seg.items():
        index = np.all(label == hex_to_rgb(key), axis=-1)
        label_convert[index] = value

    #print(label_convert.shape) 

    return cv2.cvtColor(label_convert, cv2.COLOR_RGB2GRAY)

class DataHandlerA2D2(Dataset):
    def __init__(
        self,
        data_path,
        img_trans=None,
        label_trans=None,
        lab_path=None,
        labeled_part=None,
        multi_class=False,
        curr_selected_patches=None,
        patch_number=None,
        patch_shape=None,
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
                frame_id = video_id + '/' + frame_nb
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
        # class_index_to_color, merged_class_list, class_to_nb = self.prepare_color_conversion('/'.join(lab_path.split('/')[:-2]))

        # self.color_converter = A2D2ColorTransformer(class_index_to_color)
        # self.merged_class_list = merged_class_list
        # self.class_to_nb = class_to_nb

    def prepare_color_conversion(self, data_path):
        with open(data_path + '/class_list.json', 'r') as f:
            class_list = json.load(f)

        ### uno ###
        class_index_to_color = {}
        for color, name in class_list.items():
            rgb = tuple(int(color[i+1:i+3], 16) for i in (0, 2, 4))
            class_index_to_color[name] = rgb

        ### due ###
        merged_class_list = {}
        for i, (k, v) in enumerate(class_list.items()):
            splited = v.split(' ')
            if splited[-1].isdigit():
                v = ' '.join(splited[:-1])
            else:
                v = ' '.join(splited)
            merged_class_list[i] = v
        
        class_to_nb = {}
        curr_nb = 0
        for k, v in merged_class_list.items():
            if v not in class_to_nb:
                class_to_nb[v] = curr_nb
                curr_nb += 1

        return class_index_to_color, merged_class_list, class_to_nb

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

        if not self.multi_class:
            y = y != 0

        if toTensor:
            y = torch.Tensor(y)
            y = y.float() if not self.multi_class else y.long()
        else:
            y = y.astype("float") if not self.multi_class else y.astype("int")
        return y

    def open_path(self, img_path, mask_path, name=None, toTensor=True):
        x = cv2.imread(img_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = cv2.imread(mask_path)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = lbl_convert(y)
        # y = self.color_converter.transform(y, dtype=np.uint8)
        # for k, v in self.merged_class_list.items():
        #     y[y == k] = self.class_to_nb[v]

        # if self.load_patches:
        #     x, y = self.extract_patches(x, y)

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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_A2D2):]
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
        name = name.split("/")[0] + "/" + name.split("/")[1][len(routes.FRAME_KEYWORD_A2D2):]
        label = np.array(label)

        copy_label = np.array(label)
        label[label > -1] = routes.IGNORE_INDEX
        for patch in self.curr_selected_patches[name]:            
            label[superpixel_label == patch] = copy_label[superpixel_label == patch]
        return label

    def load_superpixel(self, name):
        superpixel_path = self.lab_path.replace('labels', f'superpixels_{self.patch_number}') + f'{name}.png'
        superpixel_label = np.array(Image.open(superpixel_path))
        return superpixel_label
    
    def __getitem__(self, index):
        name = self.data_pool[index][1]
        name = name[len(self.lab_path) : -len(FILE_TYPE)]
        x, y = self.open_path(self.data_pool[index][0], self.data_pool[index][1], name)

        return x, y, name

    def __len__(self):
        return len(self.data_pool)