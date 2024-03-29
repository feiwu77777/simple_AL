import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as vision_F
import torch.nn as nn
import torch.nn.functional as F
from config import config
import routes
from miou import MeanIoU


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validate_v2(model, dataloader, criterion, metric='DICE', num_classes=1, dataset_name='auris'):
    #print("\n--------Validating---------\n")
    model.eval()
    valid_loss = 0.0
    score = 0
    counter = 0

    iou_helper = MeanIoU(config['N_LABEL'], routes.IGNORE_INDEX)
    iou_helper._before_epoch()
    with torch.no_grad():
        for i, (image, masks, names) in enumerate(dataloader):
            counter += 1
            image, masks = image.to(DEVICE), masks.to(DEVICE)

            outputs = model(image)
            _, preds = torch.max(outputs, 1)  # Get class predictions from probabilities
            loss = criterion(outputs, masks)
            valid_loss += loss.item()

            output_dict = {
                'outputs': preds,
                'targets': masks
            }
            iou_helper._after_step(output_dict)

    ious = iou_helper._after_epoch()
    start = 0
    if not config['DATASET'] in routes.CLASS_0_DATASETS:
        start = 1
    miou = np.mean(ious[start:])
    all_scores = {i: v for i, v in enumerate(ious)}

    valid_loss = valid_loss / counter
    return valid_loss, miou, all_scores


class RandomScale(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, interpolation, scale_limit=100):
        self.size = size
        self.interpolation = interpolation
        self.scale_limit = scale_limit

    def __call__(self, image):
        # resize image to a random size between (x1, x2) and (x1+100, x2+100)
        # without changing the aspect ratio using a single scale
        scale = int(self.scale_limit * torch.rand(1).item())
        size = (self.size[0] + scale, self.size[1] + scale)
        image = vision_F.resize(image, size, self.interpolation)

        # then a random crop of (x1, x2) will be obtained from this image
        return image


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, ignore_index=routes.IGNORE_INDEX, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()