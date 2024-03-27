import numpy as np
from VAE_model import VAE
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_score
import collections
import torch
import random
import os
from torchvision.models import resnet34, resnet50
from torch.utils.data import DataLoader
import torch.nn as nn
# from byol_pytorch import BYOL
from config import config
import routes
# from kneed import KneeLocator
import matplotlib.pyplot as plt
from routes import PRINT_PATH
import cv2

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

def pixelChange(x1, x2):
    x1 = x1[:, :, 0] / np.max(x1[:, :, 0])
    x2 = x2[:, :, 0] / np.max(x2[:, :, 0])

    res = np.sum(np.around(x1, 1) != np.around(x2, 1)) / (x1.shape[0] *
                                                          x1.shape[1])
    return res


def mutual_information_2d(x, y, sigma=1, normalized=False):
    bins = (256, 256)
    EPS = np.finfo(float).eps
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) /
              np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi


def addSample(new_labeled, curr_labeled, unlabeled_dataset, new_path, class_ID,
              IMG_PATH, thresh):
    x_new, _ = unlabeled_dataset.open_path(new_path, new_path, toTensor=False)
    if class_ID not in new_labeled and class_ID not in curr_labeled:
        return True
    v1 = new_labeled[class_ID]
    v2 = curr_labeled[class_ID]
    for nb in v1:
        img_path = IMG_PATH + class_ID + nb + ".jpg"
        x, _ = unlabeled_dataset.open_path(img_path, img_path, toTensor=False)

        diff = mutual_information_2d(x.ravel(), x_new.ravel())
        if diff > thresh:
            return False
    for nb in v2:
        img_path = IMG_PATH + class_ID + nb + ".jpg"
        x, _ = unlabeled_dataset.open_path(img_path, img_path, toTensor=False)

        diff = mutual_information_2d(x.ravel(), x_new.ravel())
        if diff > thresh:
            return False
    return True


def pixelChangeV2(x1, x2):
    assert (np.min(x1) >= 0)
    assert (np.max(x1) <= 1)
    assert (np.min(x2) >= 0)
    assert (np.max(x2) <= 1)

    res = np.sum(np.around(x1, 1) != np.around(x2, 1)) / (x1.shape[0] *
                                                          x1.shape[1])
    return res


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


# def AE_embeddings(data_path, vae_net):

#     dataset = VAE_DataHandler(data_path)
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
#     embeddings = []
#     names = []
#     with torch.no_grad():
#         for i, (image, name) in enumerate(dataloader):
#             recon_data, embedding, log_var = vae_net(image.to(DEVICE))
#             # embedding = torch.zeros((8, 512, 3, 3))
#             embedding = torch.mean(embedding, axis=(2, 3))
#             embeddings.append(embedding)
#             names.extend(name)
#     embeddings = torch.cat(embeddings, axis=0)

#     return embeddings, names


def embedding_similarity(labeled_frames,
                         unlabeled_frames,
                         weight_path='../pretrained_models/vae/all_video.pt'):

    vae_net = VAE(channel_in=3, ch=64).to(DEVICE)
    save_file = torch.load(weight_path)
    vae_net.load_state_dict(save_file['model_state_dict'])
    vae_net.eval()

    labeled_embeddings, _ = AE_embeddings(labeled_frames, vae_net)
    unlabeled_embeddings, names = AE_embeddings(unlabeled_frames, vae_net)

    distances = torch.cdist(unlabeled_embeddings, labeled_embeddings)
    distances = torch.mean(distances, axis=1)

    return distances, names


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

def simCLR_projection_embedding(dataloader, 
                    arg=None, 
                    weight_path='../pretrained_models/skateboard_simCLR/pretrained_epoch=10.pth.tar', 
                    notebook=False,
                    out_dim=128):
        
    model = resnet50(pretrained=False)
    dim_mlp = model.fc.in_features
    model.fc = nn.Linear(dim_mlp, out_dim)
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                        nn.ReLU(), model.fc)

    if notebook:
        weight = torch.load(weight_path, map_location='cpu')
    else:
        weight = torch.load(weight_path)

    epoch = weight['epoch']
    if notebook:
        print(f"-- loaded embedding model weight from epoch {epoch} iter\n")
    else:
        with open(PRINT_PATH, "a") as f:
            f.write(f"-- loaded embedding model weight from epoch {epoch} iter\n")

    state_dict = weight['state_dict']
    for k in list(state_dict.keys()):
        state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for images, y, names in dataloader:
            images = images.to(DEVICE)
            
            output = model(images)

            for i, name in enumerate(names):
                embeddings[name] = output[i].detach().cpu()

    return embeddings

def BYOL_embedding(dataloader, 
                    weight_path='..',
                    notebook=False):

    net = resnet50(pretrained=False)
    learner = BYOL(
        net,
        image_size = 220,
        hidden_layer = 'avgpool',
        use_momentum = True,
    )
    online_encoder = learner.online_encoder

    model_weights = torch.load(weight_path, map_location=torch.device('cuda:0'))
    epoch = model_weights['epoch']
    if notebook:
        print(f"-- loaded embedding model weight from epoch {epoch} iter\n")
    else:
        with open(PRINT_PATH, "a") as f:
            f.write(f"-- loaded embedding model weight from epoch {epoch} iter\n")

    encoder_weight = model_weights['state_dict']

    encoder_net_dict = {}
    encoder_projector_dict = {}
    for key, value in encoder_weight.items():
        if 'projector' in key:
            encoder_projector_dict[key[len('projector.'):]] = value
        elif 'net':
            encoder_net_dict[key[len('net.'):]] = value

    online_encoder.net.load_state_dict(encoder_net_dict)
    online_encoder.projector.load_state_dict(encoder_projector_dict)
    online_encoder = online_encoder.to(DEVICE)
    online_encoder.eval()

    embeddings = {}
    proj_embeddings = {}
    with torch.no_grad():
        for images, y, names in dataloader:
            images = images.to(DEVICE)
            
            proj, emb = online_encoder(images, True)

            for i, name in enumerate(names):
                embeddings[name] = emb[i].detach().cpu()
                proj_embeddings[name] = proj[i].detach().cpu()

    return proj_embeddings, embeddings

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

def compute_entropy(model, dataset, smooth=1e-7, patch_number=None, patch_shape=None, save_predicted_masks=False):
    ML_entropy = {}
    ML_class_entropy = {}
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
    with torch.no_grad():
        for img, label, names in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img) # shape = (batch_size, n_class, h, w)
            predicted_masks = torch.argmax(pred, dim=1).cpu() # shape = (batch_size, h, w)

            if not dataset.multi_class:
                proba_pos = torch.sigmoid(pred).cpu().squeeze(1)
                proba_neg = 1 - proba_pos
                all_proba = torch.stack((proba_pos, proba_neg), axis=1)
            else:
                all_proba = torch.softmax(pred, dim=1).cpu() # shape = (batch_size, n_class, h, w)

                class_entropy = []
                for c in range(config['N_LABEL']):
                    proba_pos = all_proba[:, c, :, :] # shape = (batch_size, h, w)
                    proba_neg = 1 - proba_pos # shape = (batch_size, h, w)
                    all_proba_bin = torch.stack([proba_neg, proba_pos], dim=1) # shape = (batch_size, 2, h, w)
                    log_proba_bin = torch.log(all_proba_bin + smooth) # shape = (batch_size, 2, h, w)
                    entropy = - (all_proba_bin * log_proba_bin).sum(1) # shape = (batch_size, h, w)
                    class_entropy.append(entropy)
                class_entropy = torch.stack(class_entropy, dim=0) # shape = (n_class, batch_size, h, w)

            log_proba = torch.log(all_proba + smooth)
            global_entropy = (all_proba * log_proba).sum(1) # shape = (batch_size, h, w)

            for i, name in enumerate(names):
                if save_predicted_masks:
                    video_id, frame_nb = name.split('/')
                    save_path = f'./ML_preds/{video_id}'
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path, exist_ok=True)
                    torch.save(predicted_masks[i], save_path + f'/{frame_nb}.pt')

                # calculate ML entropy
                if patch_number is None:
                    entropy_value = torch.mean(global_entropy[i]).item()
                    ML_entropy[name] = -entropy_value
                    # calculate ML class entropy
                    if dataset.multi_class:
                        ML_class_entropy[name] = np.max(class_entropy[:, i, :, :].numpy(), axis=(1,2)) # shape = (n_class) CHANGE MAX TO MEAN?

                elif patch_shape == 'rectangle':
                    patch_size_x = class_entropy.shape[2] // patch_number
                    patch_size_y = class_entropy.shape[3] // patch_number

                    all_patch_class_entropy = []
                    all_patch_entropy = []
                    for i2 in range(patch_number):
                        for j2 in range(patch_number):
                            start_x = i2 * patch_size_x
                            start_y = j2 * patch_size_y

                            end_x = start_x + patch_size_x
                            if i2 == patch_number - 1:
                                end_x = class_entropy.shape[2]
                            end_y = start_y + patch_size_y
                            if j2 == patch_number - 1:
                                end_y = class_entropy.shape[3]

                            if dataset.multi_class:
                                patch_class_entropy = class_entropy[:, i, start_x: end_x, start_y: end_y]
                                patch_class_entropy = np.mean(patch_class_entropy.numpy(), axis=(1,2)) # shape = (n_class) # change max to mean?
                                all_patch_class_entropy.append(patch_class_entropy)

                            patch_entropy = global_entropy[i, start_x: end_x, start_y: end_y]
                            patch_entropy = -torch.mean(patch_entropy).item()
                            all_patch_entropy.append(patch_entropy)
                    
                    ML_entropy[name] = np.stack(all_patch_entropy, axis=0) # shape = (patch_number**2)
                    if dataset.multi_class:
                        ML_class_entropy[name] = np.stack(all_patch_class_entropy, axis=0) # shape = (patch_number**2, n_class)
                
                elif patch_shape == 'superpixel':
                    superpixel_lab = dataset.load_superpixel(name, transform=True)

                    all_patch_class_entropy = []
                    all_patch_entropy = []
                    for patch_id in np.unique(superpixel_lab):
                        if dataset.multi_class:
                            patch_class_entropy = class_entropy[:, i, superpixel_lab == patch_id] # shape = (n_class, n_pixel)
                            patch_class_entropy = np.max(patch_class_entropy.numpy(), axis=1) # shape = (n_class) CHANGE MAX TO MEAN?
                            all_patch_class_entropy.append(patch_class_entropy)
                        
                        patch_entropy = global_entropy[i, superpixel_lab == patch_id] # shape = (n_pixel)
                        patch_entropy = -torch.mean(patch_entropy).item()
                        all_patch_entropy.append(patch_entropy)
                    
                    ML_entropy[name] = np.stack(all_patch_entropy, axis=0) # shape = (patch_number**2)
                    if dataset.multi_class:
                        ML_class_entropy[name] = np.stack(all_patch_class_entropy, axis=0) # shape = (patch_number**2, n_class)

    return ML_entropy, ML_class_entropy


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

def balance_pixels(pixel_per_class, total_additional_pixel=10, n_class=12, start=0):
    # make copy of img_per_class
    pixel_dict = pixel_per_class.copy()
    for class_id in range(start, n_class):
        if class_id not in pixel_dict:
            pixel_dict[class_id] = 0

    added_pixels = {}
    while True:
        least_class, img_count = sorted(pixel_dict.items(), key=lambda x: x[1])[0]
            
        # If we've added all the images we can, stop
        if total_additional_pixel <= 0:
            return added_pixels, pixel_dict
            
        # Add an image to the current class
        pixel_dict[least_class] += 1
        total_additional_pixel -= 1
        
        # If the class is already in the dictionary, increment the count
        if least_class in added_pixels:
            added_pixels[least_class] += 1
        # Otherwise, add the class to the dictionary with a count of 1
        else:
            added_pixels[least_class] = 1

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

def get_img_perc_ubelix(
    config,
    train_dataset,
    train_class_dices=None,
    val_class_dices=None,
    test_class_dices=None,
    frame_keyword="frame",
    notebook=False,
    print_=True
):
    train_class_dices = train_class_dices or {}
    val_class_dices = val_class_dices or {}
    test_class_dices = test_class_dices or {}

    if notebook and print_:
        print(f"-- TOTAL IMG -- {len(train_dataset)}\n")
    elif os.path.exists(PRINT_PATH):
        with open(PRINT_PATH, "a") as f:
            f.write("\n")
            f.write(f"-- TOTAL IMG -- {len(train_dataset)}\n")

    img_per_class = {}
    pixel_per_class = {}
    class_per_img = {}

    if train_dataset.patch_number is None:
        for i in range(len(train_dataset)):
            x, y, frame_id = train_dataset[i]
            for c in range(config['N_LABEL']):
                if torch.sum(y == c).item() > 0:
                    img_per_class[c] = img_per_class.get(c, 0) + 1
                    pixel_per_class[c] = pixel_per_class.get(c, 0) + torch.sum(y == c).item()
                    class_per_img[frame_id] = class_per_img.get(frame_id, []) + [c]
    
    elif config['PATCH_SHAPE'] == 'rectangle':
        total_pixels = 0
        patch_number = train_dataset.patch_number
        curr_selected_patches = train_dataset.curr_selected_patches
        for i in range(len(train_dataset)):
            x, y, frame_id = train_dataset[i]
            n = '/'.join([frame_id.split('/')[-2], frame_id.split('/')[-1][len(frame_keyword):]])
            patch_ids = curr_selected_patches[n]
            for patch_id in patch_ids:
                i, j = divmod(patch_id, patch_number)
                patch_size_x = y.shape[0] // patch_number
                patch_size_y = y.shape[1] // patch_number
                start_x = i * patch_size_x
                start_y = j * patch_size_y

                end_x = start_x + patch_size_x
                if i == patch_number - 1:
                    end_x = y.shape[0]
                end_y = start_y + patch_size_y
                if j == patch_number - 1:
                    end_y = y.shape[1]
                
                patch_y = y[start_x:end_x, start_y:end_y]
                total_pixels += patch_y.shape[0] * patch_y.shape[1]
                for c in range(config['N_LABEL']):
                    if torch.sum(patch_y == c).item() > 0:
                        img_per_class[c] = img_per_class.get(c, 0) + 1
                        pixel_per_class[c] = pixel_per_class.get(c, 0) + torch.sum(patch_y == c).item()
                        class_per_img[frame_id + f'/{patch_id}'] = class_per_img.get(frame_id + f'/{patch_id}', []) + [c]
    
    elif config['PATCH_SHAPE'] == 'superpixel' and not config['MULTI_CLASS_LABELING']:
        total_pixels = 0
        curr_selected_patches = train_dataset.curr_selected_patches
        for i in range(len(train_dataset)):
            x, y, frame_id = train_dataset[i]
            n = '/'.join([frame_id.split('/')[-2], frame_id.split('/')[-1][len(frame_keyword):]])
            patch_ids = curr_selected_patches[n]

            superpixel_lab = train_dataset.load_superpixel(frame_id, transform=True)
            for patch_id in patch_ids:                
                patch_y = y[superpixel_lab == patch_id]
                total_pixels += len(patch_y)
                for c in range(config['N_LABEL']):
                    if torch.sum(patch_y == c).item() > 0:
                        img_per_class[c] = img_per_class.get(c, 0) + 1
                        pixel_per_class[c] = pixel_per_class.get(c, 0) + torch.sum(patch_y == c).item()
                        class_per_img[frame_id + f'/{patch_id}'] = class_per_img.get(frame_id + f'/{patch_id}', []) + [c]

    elif config['PATCH_SHAPE'] == 'superpixel' and config['MULTI_CLASS_LABELING']:
        total_pixels = 0
        curr_selected_patches = train_dataset.curr_selected_patches
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            y = data['labels']
            frame_id = data['names']
            n = '/'.join([frame_id.split('/')[-2], frame_id.split('/')[-1][len(frame_keyword):]])
            patch_ids = curr_selected_patches[n]
            for c in range(config['N_LABEL']):
                img_per_class[c] = img_per_class.get(c, 0) + torch.sum(y[patch_ids, c]).item()

    # sort img_per class
    img_per_class = {k: v for k, v in sorted(img_per_class.items(), key=lambda x: x[0])}

    start = 1
    total_background_pixel = pixel_per_class.get(0, 0)
    if config['DATASET'] in routes.CLASS_0_DATASETS:
        start = 0
        total_background_pixel = 0

    total_foreground_pixel = 1e-7
    for k in range(start, config['N_LABEL']):
        total_foreground_pixel += pixel_per_class.get(k, 0)

    if train_dataset.patch_number is None and not config['MULTI_CLASS_LABELING']:
        assert y.shape[0] * y.shape[1] * len(train_dataset) == total_background_pixel + total_foreground_pixel
    else:
        pass 
        # not working here because intuitive image are resized from rect to square
        # and the border of patches are changed a bit and there are one column or one row that have 255 values per patch
        # assert total_pixels == total_background_pixel + total_foreground_pixel

    img_gini_coef = 1 - gini(np.array([img_per_class.get(k, 0) for k in range(start, config['N_LABEL'])]))
    pixel_gini_coef = 0
    if not config['MULTI_CLASS_LABELING']:
        pixel_per_class_perc = {}
        for k in range(start, config['N_LABEL']):
            pixel_per_class_perc[k] = pixel_per_class.get(k, 0) / total_foreground_pixel
        
        pixel_gini_coef = 1 - gini(np.array(list(pixel_per_class_perc.values())))
    
    if notebook and print_:
        for k in range(start, config['N_LABEL']):
            print(f'{k}: {img_per_class.get(k, 0)}  -- {pixel_per_class.get(k, 0) / total_foreground_pixel: 0.03f} -- {train_class_dices.get(k, -1):0.02f} -- {val_class_dices.get(k, -1):0.02f} -- {test_class_dices.get(k, -1):0.02f}\n')

        print(
            f"mIoU: -- {np.mean(list(train_class_dices.values())):0.04f} -- {np.mean(list(val_class_dices.values())):0.04f} -- {np.mean(list(test_class_dices.values())):0.04f}\n"
            # f"Total Foreground Pixel: {total_foreground_pixel}\n"
            f"Pixel Gini Coef: {pixel_gini_coef}, Img Gini Coef: {img_gini_coef}\n"
            f"\n"
        )
    elif os.path.exists(PRINT_PATH):
        with open(PRINT_PATH, 'a') as f:
            for k in range(start, config['N_LABEL']):
                f.write(f'{k}: {img_per_class.get(k, 0)}  -- {pixel_per_class.get(k, 0) / total_foreground_pixel: 0.03f} -- {train_class_dices.get(k, -1):0.02f} -- {val_class_dices.get(k, -1):0.02f} -- {test_class_dices.get(k, -1):0.02f}\n')

            f.write(
                f"mIoU: -- {np.mean(list(train_class_dices.values())):0.04f} -- {np.mean(list(val_class_dices.values())):0.04f} -- {np.mean(list(test_class_dices.values())):0.04f}\n"
                # f"Total Foreground Pixel: {total_foreground_pixel}\n"
                f"Pixel Gini Coef: {pixel_gini_coef}, Img Gini Coef: {img_gini_coef}\n"
                f"\n"
            )

    img_distributions = {
        'class_per_img': class_per_img,
        'img_per_class': img_per_class,
        'pixel_per_class': pixel_per_class,
        'pixel_gini_coef': pixel_gini_coef,
        'img_gini_coef': img_gini_coef,
    }
    return img_distributions

def get_clusters_x_class(k_means_centers_name, fixed_cluster, ML_class_entropy, added_imgs, start=1):
    class_entropy_per_cluster = {}
    for cluster, frames in k_means_centers_name.items():
        if cluster in fixed_cluster:
            continue
        avg_class_entropy = []
        for frame in frames:
            avg_class_entropy.append(ML_class_entropy[frame])
        avg_class_entropy = np.stack(avg_class_entropy)
        avg_class_entropy = np.mean(avg_class_entropy, axis=0) # changed from mean to max
        class_entropy_per_cluster[cluster] = avg_class_entropy

    tuples = []
    # Iterate through the array
    for cluster, row in class_entropy_per_cluster.items():
        for i in range(start, len(row)):
            # Store the value and its indexes in a tuple, then append it to the list
            tuples.append((row[i], cluster, i))
    tuples.sort(reverse=True)

    clusters_x_class = {}
    for entropy, cluster, class_ in tuples:
        remaining = added_imgs.get(class_, 0)
        if remaining > 0 and cluster not in clusters_x_class:
            clusters_x_class[cluster] = class_
            added_imgs[class_] -= 1

    return clusters_x_class

def get_clusters_x_class_video(all_k_means_centers_name, all_fixed_cluster, ML_class_entropy, added_imgs, start=1):
    class_entropy_per_video = {}
    for video_id, k_means_centers_name in all_k_means_centers_name.items():
        fixed_cluster = all_fixed_cluster.get(video_id, {})

        for cluster, frames in k_means_centers_name.items():
            if cluster in fixed_cluster:
                continue
            avg_class_entropy = []
            for frame in frames:
                avg_class_entropy.append(ML_class_entropy[f'{video_id}/{frame}'])
            avg_class_entropy = np.stack(avg_class_entropy)
            avg_class_entropy = np.mean(avg_class_entropy, axis=0) # changed from mean to max
            assert video_id not in class_entropy_per_video
            class_entropy_per_video[video_id] = avg_class_entropy # shape = (n_class)

    tuples = []
    # Iterate through the array
    for video_id, row in class_entropy_per_video.items():
        for i in range(start, len(row)):
            # Store the value and its indexes in a tuple, then append it to the list
            tuples.append((row[i], video_id, i))
    tuples.sort(reverse=True)

    video_x_class = {}
    for entropy, video_id, class_ in tuples:
        remaining = added_imgs.get(class_, 0)
        if remaining > 0 and video_id not in video_x_class:
            video_x_class[video_id] = class_
            added_imgs[class_] -= 1

    return video_x_class


def get_videos_x_class(frames_per_video, ML_class_entropy, added_imgs, start=1):
    class_entropy_per_video = {}
    for video, frames in frames_per_video.items():
        avg_class_entropy = []
        for frame in frames:
            avg_class_entropy.append(ML_class_entropy[f'{video}/{frame}'])
        avg_class_entropy = np.stack(avg_class_entropy)
        avg_class_entropy = np.mean(avg_class_entropy, axis=0) # changed from mean to max
        class_entropy_per_video[video] = avg_class_entropy

    tuples = []
    # Iterate through the array
    for video, row in class_entropy_per_video.items():
        for i in range(start, len(row)):
            # Store the value and its indexes in a tuple, then append it to the list
            tuples.append((row[i], video, i))
    tuples.sort(reverse=True)

    videos_x_class = {}
    for entropy, video, class_ in tuples:
        remaining = added_imgs.get(class_, 0)
        if remaining > 0 and video not in videos_x_class:
            videos_x_class[video] = class_
            added_imgs[class_] -= 1

    return videos_x_class


def get_patch_clusters_x_class(all_k_means_centers_name, all_fixed_cluster, ML_class_entropy, added_imgs, dataset_name='auris'):
    all_class_entropy_per_cluster = []
    for patch_id, k_means_centers_name in enumerate(all_k_means_centers_name):
        fixed_cluster = all_fixed_cluster[patch_id]
    
        class_entropy_per_cluster = {}
        for cluster, frames in k_means_centers_name.items():
            if cluster in fixed_cluster:
                continue
            avg_class_entropy = []
            for frame in frames:
                avg_class_entropy.append(ML_class_entropy[frame][patch_id])
            avg_class_entropy = np.stack(avg_class_entropy)
            avg_class_entropy = np.mean(avg_class_entropy, axis=0)
            class_entropy_per_cluster[cluster] = avg_class_entropy

        all_class_entropy_per_cluster.append(class_entropy_per_cluster)

    tuples = []
    start = 1
    if dataset_name in routes.CLASS_0_DATASETS:
        start = 0
    for patch_id, class_entropy_per_cluster in enumerate(all_class_entropy_per_cluster):
        for cluster, row in class_entropy_per_cluster.items():
            for class_nb in range(start, len(row)):
                tuples.append((row[class_nb], patch_id, cluster, class_nb))
    tuples.sort(reverse=True)

    all_clusters_x_class = [{} for i in range(len(all_k_means_centers_name))]
    for entropy, patch_id, cluster, class_ in tuples:
        remaining = added_imgs.get(class_, 0)
        clusters_x_class = all_clusters_x_class[patch_id]
        if remaining > 0 and cluster not in clusters_x_class:
            clusters_x_class[cluster] = class_
            added_imgs[class_] -= 1
        all_clusters_x_class[patch_id] = clusters_x_class

    return all_clusters_x_class

class SpatialPurity(nn.Module):

    def __init__(self, in_channels=19, padding_mode='zeros', size=3):
        super(SpatialPurity, self).__init__()
        assert size % 2 == 1, "error size"
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                              stride=1, padding=int(size / 2), bias=False, padding_mode=padding_mode,
                              groups=in_channels)
        a = torch.ones((size, size), dtype=torch.float32)
        a = a.unsqueeze(dim=0).unsqueeze(dim=0)
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, x):
        summary = self.conv(x)
        # summary: (b, 19, h, w)
        count = torch.sum(summary, dim=1, keepdim=True)
        # count: (b, 1, h, w)
        dist = summary / count
        # dist: (b, 19, h, w)
        spatial_purity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True)
        # (b, 1, h, w), normally b = 1, (1, 1, h, w)
        return spatial_purity
    
def pad_superpixel(label, img, patch_id, fill_blank=False, center=False):
    image = np.array(img)
    patch = label == patch_id
    lower_y = np.where(np.any(patch, axis=1))[0][0]
    upper_y = np.where(np.any(patch, axis=1))[0][-1]

    lower_x = np.where(np.any(patch, axis=0))[0][0]
    upper_x = np.where(np.any(patch, axis=0))[0][-1]

    if not fill_blank:
        image[label != patch_id] = 0
    image = image[lower_y: upper_y + 1, lower_x: upper_x + 1]

    h, w = image.shape[:2]
    if h > w:
        if lower_x == 0 and not center:
            image = np.pad(image, ((0, 0), (0, h - w), (0, 0)), 'constant', constant_values=0)
        elif upper_x == label.shape[1] - 1 and not center:
            image = np.pad(image, ((0, 0), (h - w, 0), (0, 0)), 'constant', constant_values=0)
        else:
            image = np.pad(image, ((0, 0), ((h - w) // 2, (h - w) // 2), (0, 0)), 'constant', constant_values=0)
    else:
        if lower_y == 0 and not center:
            image = np.pad(image, ((0, w - h), (0, 0), (0, 0)), 'constant', constant_values=0)
        elif upper_y == label.shape[0] - 1 and not center:
            image = np.pad(image, ((w - h, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            image = np.pad(image, (((w - h) // 2, (w - h) // 2), (0, 0), (0, 0)), 'constant', constant_values=0)

    return image


def find_elbow(data, s=1.0, plot=False):
    x = range(1, len(data) + 1)
    y = data

    kn = KneeLocator(x, y, curve='concave', direction='increasing', S=s)
    elbow_point = kn.knee

    if plot:
        plt.figure(figsize=(8,6))
        plt.plot(x, y, 'b')  # data curve
        if elbow_point is not None:  # check if a knee was found
            plt.plot(elbow_point, y[elbow_point-1], 'ro')  # elbow point
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('The Elbow Method showing the optimal knee')
        plt.grid(True)
        plt.show()

    return elbow_point, y[elbow_point-1] if elbow_point is not None else None

def is_adjacent(ind1, ind2, superpixel_lab):
    patch1 = np.zeros_like(superpixel_lab)
    patch2 = np.zeros_like(superpixel_lab)
    patch1[superpixel_lab == ind1] = 1
    patch2[superpixel_lab == ind2] = 1

    # expand patch1 by 1 pixel on all directions
    kernel = np.ones((3, 3), np.uint8)
    patch1 = cv2.dilate(patch1.astype(np.uint8), kernel, iterations=1)

    intersection = np.logical_and(patch1, patch2)
    return np.sum(intersection) > 0

def compute_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))