### https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py

from email.policy import strict
import torch
from collections import OrderedDict
from typing import Optional, Dict
from urllib import request
import os

from torch import nn, Tensor
from torch.nn import functional as F
from routes import PRETRAINED_PATH
import torchvision.models as resnet


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module,
                 aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(
            x,
            size=input_shape,
            mode="nearest",
        )  # align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(
                x,
                size=input_shape,
                mode="nearest",
            )  #align_corners=False)
            result["aux"] = x

        return result['out']

    def feat_forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(
            x,
            size=input_shape,
            mode="nearest",
        )  # align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(
                x,
                size=input_shape,
                mode="nearest",
            )  #align_corners=False)
            result["aux"] = x

        return features['out'], result['out']

def load_state_dict_from_url(model_url):
    local_file = PRETRAINED_PATH + 'deeplabv3_resnet50_coco-cd0a2569.pth'
    if not os.path.isfile(local_file):
        request.urlretrieve(model_url, local_file)
    state = torch.load(local_file)
    _ = state.pop('aux_classifier.4.weight', None)
    _ = state.pop('aux_classifier.4.bias', None)
    _ = state.pop('classifier.4.weight', None)
    _ = state.pop('classifier.4.bias', None)
    return state


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str],
                  progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(model_url)
    model.load_state_dict(state_dict, strict=False)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str,
                                                             str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


__all__ = [
    "DeepLabV3",
    "deeplabv3_resnet50",
]

model_urls = {
    "deeplabv3_resnet50_coco":
    "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
}


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(
            x,
            size=size,
            mode="nearest",
        )
        #align_corners=False)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 atrous_rates,
                 out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels,
                      1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(backbone, num_classes: int, aux):
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """
    if pretrained:
        # aux_loss = True
        pretrained_backbone = False

    backbone = resnet.resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if pretrained:
        arch = "deeplabv3_resnet50_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model