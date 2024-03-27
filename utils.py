import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_score
import collections
import torch
import random
import os
from torchvision.models import resnet34, resnet50
from torch.utils.data import DataLoader
import torch.nn as nn
from config import config
import routes
from routes import PRINT_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_random(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    # torch.use_deterministic_algorithms(True) # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resume_dataloader(model, loader, epochs):
    model.train()
    with torch.no_grad():
        for epoch in range(epochs):
            for i, (x, _, _) in enumerate(loader):
                if i == 0:
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"=== epoch {epoch + 1}, first sample: {x[0, 0, 120, 120]}\n"
                        )
                x = x.to(DEVICE)
                outputs = model(x)


def get_score(y_true, y_pred, metric='DICE', multi_class=False):
    all_dices = {}
    if metric == 'DICE':
        if not multi_class:
            score = dice_coef(y_true, y_pred, smooth=1)
        else:
            # print('unique values in y_true:', np.unique(y_true))
            # print('unique values in y_pred:', np.unique(y_pred))
            classes_present = np.union1d(np.unique(y_true), np.unique(y_pred))
            classes_present = classes_present[classes_present != 0]

            all_dices = {c: dice_coef(y_true == c, y_pred == c) for c in classes_present}
            score = np.mean(list(all_dices.values()))
        
        return score, all_dices, all_dices

    elif metric == 'IoU':
        classes_present = np.union1d(np.unique(y_true), np.unique(y_pred))
        if not config['DATASET'] in routes.CLASS_0_DATASETS:
            classes_present = classes_present[classes_present != 0]

        all_unions = {}
        all_inters = {}
        all_ious = {}
        for c in classes_present:
            iou, inter, union = iou_metric(y_true == c, y_pred == c)
            all_unions[c] = union
            all_inters[c] = inter
            all_ious[c] = iou
        score = np.mean(list(all_ious.values()))
        return score, all_inters, all_unions
    elif metric == 'AUROC':
        y_true = y_true > 0.5
        if np.sum(y_true) == 0:
            return 1
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        score = auc_score(fpr, tpr)
    elif metric == 'BCE':
        smooth = 1e-7
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        score = -np.sum(y_true * np.log(y_pred + smooth) +
                        (1 - y_true) * np.log(1 - y_pred + smooth))
    return score


def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1.0
    else:
        return (2. * intersection + smooth) / (union + smooth)


def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score, intersection, union


def get_similarity_score(new_labeled, curr_labeled, ML_preds, all_frames):

    all_sims = collections.defaultdict(int)
    for k, v in all_frames.items():
        labeled_frames = new_labeled[k] + curr_labeled[k]
        unlabeled_frames = [nb for nb in v if nb not in labeled_frames]

        for nb in unlabeled_frames:
            mask1 = np.load(ML_preds[k + nb])
            for nb2 in labeled_frames:
                mask2 = np.load(ML_preds[k + nb2])
                sim = 1 - pixelChangeV2(mask1, mask2)
                if sim > all_sims[k + nb]:
                    all_sims[k + nb] = sim

    return all_sims


def resnet_embedding(dataloader, reduce_FM=True):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model = resnet34(pretrained=True)
    model.eval()
    # chose the layer for the embedding
    model.layer2[3].register_forward_hook(get_activation('features'))
    # model.avgpool.register_forward_hook(get_activation('features'))
    embedding_resnets = {}
    with torch.no_grad():
        for images, y, names in dataloader:
            output = model(images)
            for i, n in enumerate(names):
                if reduce_FM:
                    embedding_resnets[n] = torch.mean(
                        activation['features'][i], axis=(1, 2))
                else:
                    embedding_resnets[n] = activation['features'][i]

    return embedding_resnets


def simCLR_embedding(dataloader, 
                    arg=None, 
                    weight_path='../pretrained_models/skateboard_simCLR/pretrained_epoch=10.pth.tar', 
                    patch_number=None,
                    cpu=False):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    if cpu:
        weight = torch.load(weight_path, map_location='cpu')
    else:
        weight = torch.load(weight_path)

    epoch = weight['epoch']
    arch = weight['arch']

    if arch == 'resnet34':
        model = resnet34(pretrained=False)
    elif arch == 'resnet50':
        model = resnet50(pretrained=False)
    model.avgpool.register_forward_hook(get_activation('avgpool'))

    with open(PRINT_PATH, "a") as f:
        f.write(f"-- loaded embedding model weight from epoch {epoch}\n")

    state_dict = weight['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    model = model.to(DEVICE)
    model.eval()

    if patch_number is not None:
        patch_embeddings = []
        with torch.no_grad():
            for patch_nb in range(patch_number):
                embeddings = {}
                for images, y, names in dataloader:
                    images = images[patch_nb]
                    images = images.to(DEVICE)
                    output = model(images)
                    for i, name in enumerate(names):
                        embeddings[name] = activation['avgpool'][i].squeeze().detach().cpu()
                patch_embeddings.append(embeddings)
        
        return patch_embeddings
    else:
        embeddings = {}
        with torch.no_grad():
            for images, y, names in dataloader:
                images = images.to(DEVICE)
                
                output = model(images)

                for i, name in enumerate(names):
                    embeddings[name] = activation['avgpool'][i].squeeze().detach().cpu()

        return embeddings


def euc_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2)**2))


def average_center(embeddings):
    embeddings = np.stack(embeddings)
    return np.mean(embeddings, axis=0)


def median_center(embeddings):
    embeddings = np.stack(embeddings)
    return np.median(embeddings, axis=0)


def center_diff(centers, prev_centers):
    diff = 0
    for i, center in enumerate(centers):
        diff_ = euc_distance(torch.tensor(center),
                             torch.tensor(prev_centers[i]))
        # print(f'cluster {i}', diff_)
        diff += diff_
    return diff


def cosine_similarity(Ii, Ij):
    return np.dot(Ii, Ij) / (np.linalg.norm(Ii) * np.linalg.norm(Ij))


def max_cosine_similarity(Sa, Ix):
    max_sim = -np.inf
    for Ii in Sa:
        sim = cosine_similarity(Ii, Ix)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def representativeness(Sa, Su):
    return sum(max_cosine_similarity(Sa, Ix) for Ix in Su)


def compute_entropy_v2(model, dataset, smooth=1e-7, patch_number=None, curr_selected_patches=None):
    ML_entropy = {}
    ML_class_entropy = {}
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img) # shape = (batch_size, n_class, h, w)

            all_proba = torch.softmax(pred, dim=1).cpu() # shape = (batch_size, n_class, h, w)

            class_entropy = []
            for c in range(config['N_LABEL']):
                proba_pos = all_proba[:, c, :, :] # shape = (batch_size, h, w)
                proba_neg = 1 - proba_pos # shape = (batch_size, h, w)
                all_proba_bin = torch.stack([proba_neg, proba_pos], dim=1) # shape = (batch_size, 2, h, w)
                log_proba_bin = torch.log(all_proba_bin + smooth) # shape = (batch_size, 2, h, w)
                entropy = - (all_proba_bin * log_proba_bin).sum(1) # shape = (batch_size, h, w)
                class_entropy.append(entropy)
            class_entropy = torch.stack(class_entropy, dim=1) # shape = (batch_size, n_class, h, w)

            for i, name in enumerate(names):
                if 'frame' in name:
                    name2 = '/'.join([name.split('/')[-2], name.split('/')[-1][len('frame'):]])
                else:
                    name2 = name
                selected_patches = curr_selected_patches[name2]
                superpixel_lab = dataset.load_superpixel(name, transform=True)

                all_patch_class_entropy = []
                for patch_id in range(patch_number):
                    patch_class_entropy = class_entropy[i, :, superpixel_lab == patch_id] # shape = (n_class, n_pixel)
                    if patch_class_entropy.shape[1] == 0 or patch_id in selected_patches:
                        patch_class_entropy = np.zeros(config['N_LABEL'])
                    else:
                        patch_class_entropy = np.max(patch_class_entropy.numpy(), axis=1) # shape = (n_class) CHANGE MAX TO MEAN?
                    all_patch_class_entropy.append(patch_class_entropy)

                ML_class_entropy[name] = np.stack(all_patch_class_entropy, axis=0) # shape = (patch_number**2, n_class)

    return ML_entropy, ML_class_entropy


def balance_classes(img_per_class, total_additional_imgs=10, n_class=12, start=0):
    # make copy of img_per_class
    img_dict = img_per_class.copy()
    for class_id in range(start, n_class):
        if class_id not in img_dict:
            img_dict[class_id] = 0

    added_imgs = {}
    while True:
        least_class, img_count = sorted(img_dict.items(), key=lambda x: x[1])[0]
            
        # If we've added all the images we can, stop
        if total_additional_imgs <= 0:
            return added_imgs, img_dict
            
        # Add an image to the current class
        img_dict[least_class] += 1
        total_additional_imgs -= 1
        
        # If the class is already in the dictionary, increment the count
        if least_class in added_imgs:
            added_imgs[least_class] += 1
        # Otherwise, add the class to the dictionary with a count of 1
        else:
            added_imgs[least_class] = 1


def gini(array):
    # source https://github.com/oliviaguest/gini
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten().astype(np.float64) #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def compute_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))