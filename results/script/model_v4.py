import yaml
from config import img_size_auris, img_size_intuitive, img_size_cityscapes, img_size_pascalVOC
from ViT_model.factory import create_segmenter
import torch
from routes import PRINT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config():
    return yaml.load(
        open("ViT_model/config.yml", "r"), Loader=yaml.FullLoader
    )

def create_ViT_model(config, pretrained_path=None, dropout=0, drop_path=0.1, notebook=False):
    cfg_vit = load_config()

    backbone = config['VIT_BACKBONE']
    decoder = config['VIT_DECODER']
    if notebook:
        print('backbone: ', backbone)
    else:
        with open(PRINT_PATH, 'a') as f:
            f.write('backbone: ' + backbone + '\n')

    model_cfg = cfg_vit["model"][backbone]
    decoder_cfg = cfg_vit["decoder"]["mask_transformer"]

    if config['DATASET'] == 'auris':
        im_size = img_size_auris['IMG_SIZE']
        im_size += 4
    elif config['DATASET'] == 'intuitive':
        im_size = img_size_intuitive['IMG_SIZE']
    elif config['DATASET'] == 'pascal_VOC':
        im_size = img_size_pascalVOC['IMG_SIZE']
        im_size -= 1
    elif config['DATASET'] == 'cityscapes':
        im_size = img_size_cityscapes['IMG_SIZE']
        im_size -= 1

    model_cfg["image_size"] = (im_size, im_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg
    model_cfg['n_cls'] = config['N_LABEL']

    model = create_segmenter(model_cfg, notebook=notebook)

    if pretrained_path is not None:
        data = torch.load(pretrained_path, map_location=DEVICE)
        checkpoint = data["model"]

        curr_state = model.state_dict()
        filtered_checkpoint = {}
        for k, v in checkpoint.items():
            if v.shape != curr_state[k].shape:
                if notebook:
                    print(f"Skipping {k} due to shape mismatch")
                else:
                    with open(PRINT_PATH, 'a') as f:
                        f.write(f"Skipping {k} due to shape mismatch\n")
            # if 'mask_norm' not in k and 'cls_emb' not in k:
            else:
                filtered_checkpoint[k] = v
        model.load_state_dict(filtered_checkpoint, strict=False)
    return model