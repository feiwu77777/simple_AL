from routes import (
    FILE_TYPE,
)
import routes
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

class VideoDataHandler(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.video_ids = sorted(list(self.data_path.keys()))

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_frames = []
        for i in range(len(self.data_path[video_id])):
            img_path, _ = self.data_path[video_id][i]

            x = Image.open(img_path)
            x = torch.tensor(np.array(x)).permute((2, 0, 1))
            video_frames.append(x)
        return torch.stack(video_frames), video_id

    def __len__(self):
        return len(self.video_ids)


class OFDataHandler(Dataset):
    def __init__(self, data_path, label_path=routes.LAB_PATH):
        self.data_path = data_path
        self.video_ids = sorted(list(self.data_path.keys()))
        self.label_path = label_path

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_frames1 = []
        video_frames2 = []

        video_names1 = []
        video_names2 = []
        for i in range(len(self.data_path[video_id]) - 1):
            img_path1, _ = self.data_path[video_id][i]
            img_path2, _ = self.data_path[video_id][i + 1]

            x1 = Image.open(img_path1)
            x2 = Image.open(img_path2)

            x1 = torch.tensor(np.array(x1)).permute((2, 0, 1))
            x2 = torch.tensor(np.array(x2)).permute((2, 0, 1))

            video_frames1.append(x1)
            video_frames2.append(x2)

            video_names1.append(img_path1[len(self.label_path) : -len(FILE_TYPE)])
            video_names2.append(img_path2[len(self.label_path) : -len(FILE_TYPE)])
        return (
            torch.stack(video_frames1),
            torch.stack(video_frames2),
            video_names1,
            video_names2,
            video_id,
        )

    def __len__(self):
        return len(self.video_ids)


class DataHandlerBYOL(Dataset):
    def __init__(self, data_path, transform=None, label_path=routes.LAB_PATH, eval=False):
        self.data_path = data_path
        self.data_pool = np.concatenate(list(data_path.values()), axis=0)
        self.transform = transform
        self.label_path = label_path
        self.eval = eval

        self.normalize_op = T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
        self.tensor_op = T.ToTensor()

    def __getitem__(self, index):
        img_path, lab_path = self.data_pool[index]
        x = Image.open(img_path)
        x = self.tensor_op(x)
        x = self.normalize_op(x)

        name = lab_path[len(self.label_path) : -len(FILE_TYPE)]
        return x, np.zeros((10, 10)), name

    def __len__(self):
        return len(self.data_pool)