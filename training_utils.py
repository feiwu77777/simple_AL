import os
import collections
import torch
from utils import get_score
from routes import CLASS_ID_TYPE, PRINT_PATH, SAVE_MASK_PATH
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as vision_F
from torchvision.transforms import InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import config
import routes
from torch_scatter import scatter, scatter_max
from skimage.morphology import binary_dilation
from miou import MeanIoU


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validate(model, dataloader, criterion, metric='DICE', num_classes=1, dataset_name='auris'):
    #print("\n--------Validating---------\n")
    model.eval()
    valid_loss = 0.0
    score = 0
    counter = 0
    score_counter = 0

    all_inters = collections.defaultdict(float)
    all_unions = collections.defaultdict(float)
    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            counter += 1
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            if num_classes > 1:
                loss = criterion(outputs, mask)
                _, pred = torch.max(outputs, 1)  # Get class predictions from probabilities
            else:
                loss = criterion(outputs.squeeze(1), mask)
                pred = torch.sigmoid(outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                value, inters, unions = get_score(mask[j].detach().cpu().numpy(),
                                            pred[j].detach().cpu().numpy(),
                                            metric=metric,
                                            multi_class=num_classes > 1)
                score += value
                for c, union in unions.items():
                    all_unions[c] += union
                    all_inters[c] += inters[c]
            # with open(PRINT_PATH, "a") as f:
            #     f.write(
            #         f"=== val images batch {i}: {image[0, 0, 120, 120]}\n"
            #         f"=== val output batch {i}: {outputs[0, 0, 120, 120]}\n"
            #         f"=== val score  batch {i}: {score}\n"
            #     )
            valid_loss += loss.item()

    #### save image
    image = Image.fromarray((image[0].permute((1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8))
    if not num_classes > 1:
        label = Image.fromarray((mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
        pred = Image.fromarray((pred[0].detach().cpu().numpy() * 255).astype(np.uint8))
    else:
        label = mask[0].detach().cpu().numpy()
        label = label / label.max()
        pred = pred[0].detach().cpu().numpy()
        pred = pred / pred.max()
        label = Image.fromarray((label * 255).astype(np.uint8))
        pred = Image.fromarray((pred * 255).astype(np.uint8))
    image.save(f"results/preview_test_image.png")
    label.save(f"results/preview_test_label.png")
    pred.save(f"results/preview_test_pred.png")
    #### 

    valid_loss = valid_loss / counter
    score = score / score_counter

    all_scores = {}
    for c, union in all_unions.items():
        all_scores[c] = all_inters[c] / union
    
    if num_classes > 1: # and dataset_name != 'intuitive':
        score = np.mean(list(all_scores.values()))
    return valid_loss, score, all_scores


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


def train_validate(model,
                      dataset,
                      SEED,
                      n_round,
                      eval_metric='DICE',
                      sampling_metric='AUROC',
                      save=False,
                      class_id_type=CLASS_ID_TYPE,
                      num_classes=1,
                      dataset_name='auris'):
    #print("\n--------Validating---------\n")
    model.eval()
    score = 0
    counter = 0
    score_counter = 0

    score_per_class = collections.defaultdict(float)
    counter_per_class = collections.defaultdict(int)

    ML_preds = {}
    ML_scores = {}

    all_inters = collections.defaultdict(float)
    all_unions = collections.defaultdict(float)

    dataloader = DataLoader(dataset, batch_size=config["EVAL_BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            counter += 1
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            
            if num_classes > 1:
                _, pred = torch.max(outputs, 1)
            else:
                pred = torch.sigmoid(outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                value, inters, unions = get_score(mask[j].detach().cpu().numpy(),
                                        pred[j].detach().cpu().numpy(),
                                        metric=eval_metric,
                                        multi_class=num_classes > 1)
                
                score += value
                for c, union in unions.items():
                    all_unions[c] += union
                    all_inters[c] += inters[c]

                class_ID = names[j][:len(class_id_type)]
                score_per_class[class_ID] += value
                counter_per_class[class_ID] += 1

                if save:
                    save_path = SAVE_MASK_PATH + names[j] + '.npy'
                    if not os.path.exists(SAVE_MASK_PATH):
                        os.mkdir(SAVE_MASK_PATH)
                    if not os.path.exists(SAVE_MASK_PATH + class_ID):
                        os.mkdir(SAVE_MASK_PATH + class_ID)
                    np.save(save_path, pred[j].detach().cpu().numpy())
                    ML_preds[names[j]] = save_path

                    res = get_score(mask[j].detach().cpu().numpy(),
                                    pred[j].detach().cpu().numpy(),
                                    metric=sampling_metric)
                    ML_scores[names[j]] = res
    
    ### save image
    image = Image.fromarray((image[0].permute((1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8))
    if num_classes == 1:
        label = Image.fromarray((mask[0].detach().cpu().numpy() * 255).astype(np.uint8))
        pred = Image.fromarray((pred[0].detach().cpu().numpy() * 255).astype(np.uint8))
    else:
        label = mask[0].detach().cpu().numpy()
        label = label / label.max()
        pred = pred[0].detach().cpu().numpy()
        pred = pred / pred.max()
        label = Image.fromarray((label * 255).astype(np.uint8))
        pred = Image.fromarray((pred * 255).astype(np.uint8))
    image.save(f"results/preview_train_image.png")
    label.save(f"results/preview_train_label.png")
    pred.save(f"results/preview_train_pred.png")
    ###

    if save:
        with open(f'results/MLvsGT_scores_SEED={SEED}_round={n_round}.json',
                  'w') as f:
            json.dump(ML_scores, f)
        with open(f'results/ML_preds_SEED={SEED}_round={n_round}.json',
                  'w') as f:
            json.dump(ML_preds, f)

    for k, v in score_per_class.items():
        score_per_class[k] = v / counter_per_class[k]
    score = score / score_counter
    score_per_class['all'] = score

    all_scores = {}
    for c, union in all_unions.items():
        all_scores[c] = all_inters[c] / union
    
    if num_classes > 1: # and dataset_name != 'intuitive':
        score = np.mean(list(all_scores.values()))
        score_per_class['all'] = score

    return score_per_class, ML_preds, all_scores


def validate_copy(model, copy_model, dataloader, metric='DICE'):
    #print("\n--------Validating---------\n")
    model.eval()
    copy_model.eval()
    score = 0
    copy_score = 0
    score_counter = 0

    with torch.no_grad():
        for i, (image, mask, names) in enumerate(dataloader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            outputs = model(image)
            copy_outputs = copy_model(image)
            pred = torch.sigmoid(outputs).squeeze(1)
            copy_pred = torch.sigmoid(copy_outputs).squeeze(1)

            for j in range(len(mask)):
                score_counter += 1
                score += get_score(mask[j].detach().cpu().numpy(),
                                   pred[j].detach().cpu().numpy(),
                                   metric=metric)
                copy_score += get_score(mask[j].detach().cpu().numpy(),
                                        copy_pred[j].detach().cpu().numpy(),
                                        metric=metric)

    score = score / score_counter
    copy_score = copy_score / score_counter
    return score, copy_score

def fit(model,
        copy_model,
        running_coef,
        dataloader,
        optimizer,
        criterion,
        print_first=False,
        metric='DICE'):
    #print('-------------Training---------------')
    model.train()
    train_running_loss = 0.0
    score = 0
    counter = 0
    score_counter = 0

    # num of batches
    for i, (image, mask, names) in enumerate(dataloader):
        if print_first and i == 0:
            with open(PRINT_PATH, "a") as f:
                f.write(f"=== first sample: {image[0, 0, 120, 120]}\n")
        counter += 1
        image, mask = image.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs.squeeze(1), mask)
        pred = torch.sigmoid(outputs).squeeze(1)
        for j in range(len(pred)):
            score_counter += 1
            score += get_score(mask[j].detach().cpu().numpy(),
                               pred[j].detach().cpu().numpy(),
                               metric=metric)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        copy_state_dict = copy_model.state_dict()
        for k, v in model.state_dict().items():
            copy_state_dict[k] = running_coef * copy_state_dict[k] + (
                1 - running_coef) * v
        copy_model.load_state_dict(copy_state_dict)
    train_loss = train_running_loss / counter
    score = score / score_counter
    return train_loss, score

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

def max_scatter(src, index, dim, dim_size=None):
    ''' 
        src.shape = HW * C
        index.shape = HW * 1
    '''

    if dim_size is None:
        dim_size = index.max().item() + 1
    # Initialize the output tensor with negative infinity values for each class
    out = torch.full((dim_size, src.size(1)), float('-inf'), device=src.device)
    # out_inds = torch.full((dim_size, src.size(1)), -1, dtype=torch.long, device=src.device)
    # Scatter max operation
    for ind in range(dim_size):
        bin_indexes = index == ind # shape = HW x 1
        values = src[bin_indexes.squeeze()] # shape = N x C
        if len(values) > 0:
            max_values, max_inds = torch.max(values, dim=dim)
            out[ind] = max_values
            # out_inds[ind] = max_inds
    # out.scatter_reduce_(dim, index, src, reduce='amax', include_self=False)
    
    # Replace '-inf' with zeros to mimic the behavior of scatter_add_ which starts with zero
    out[out == float('-inf')] = 0
    # out_inds[out_inds == -1] = 0 # the max index seems to be random in the case of torch_scatter.scatter_max
    
    return out#, out_inds

### GroupMultiLabelCE_onlymulti
class GroupMultiLabelCE(nn.Module):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__()
        self.args = args
        self.num_class = num_class
        self.num_superpixel = num_superpixel
        self.eps = 1e-8
        self.temp = temperature
        self.reduction = reduction

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs: NxCxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets[..., :-1], dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
                ### self.num_superpixel x C : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
            trg_sup_mxpool = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            
            out_sup_mxpool = out_sup_mxpool[empty_trg_mask[i]]
            trg_sup_mxpool = trg_sup_mxpool[empty_trg_mask[i]]

            top_one_preds = out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering

            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

            loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class GroupMultiLabelCE_(GroupMultiLabelCE):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature, reduction)

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs: NxCxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets, dim=2).bool() ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
                ### self.num_superpixel x C : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
            trg_sup_mxpool = targets[i] ### self.num_superpixel x C: multi-hot annotation
            
            out_sup_mxpool = out_sup_mxpool[empty_trg_mask[i]]
            trg_sup_mxpool = trg_sup_mxpool[empty_trg_mask[i]]

            top_one_preds = out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering

            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

            loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError

class GroupMultiLabelCE_onlymulti(GroupMultiLabelCE_):
    def __init__(self, args, num_class, num_superpixel, temperature=1.0, reduction='mean'):
        super().__init__(args, num_class, num_superpixel, temperature, reduction)


    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs: NxCxHxW
            targets: N x self.num_superpixel x C+1
            superpixels: NxHxW
            spmasks: NxHxW
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel, and apply CE loss​.
            '''
        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs / self.temp, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        empty_trg_mask = torch.any(targets, dim=2).bool() ### N x self.num_superpixel
        is_trg_multi = (1 < targets.sum(dim=2)) ### N x self.num_superpixel
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''

            ### filtered outputs
            valid_mask = spmasks[i]
            if not torch.any(valid_mask):
                continue
            multi_mask = is_trg_multi[i][superpixels[i].squeeze(dim=1)[spmasks[i]]].detach()
            valid_mask = spmasks[i].clone()
            valid_mask[spmasks[i]] = multi_mask
            if not torch.any(valid_mask):
                continue

            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            # out_sup_mxpool = scatter(valid_output, valid_superpixel, dim=0, reduce='max', dim_size=self.num_superpixel)
            out_sup_mxpool = max_scatter(valid_output, valid_superpixel, dim=0, dim_size=self.num_superpixel)
                ### self.num_superpixel x C : sp 영역 내 class 별 max predicted prob, invalid superpixel 은 모두 0 으로 채워짐.
            # with open(PRINT_PATH, "a") as f:
            #             f.write(
            #                 f"scatter diff: {torch.sum(out_sup_mxpool - out_sup_mxpool_old)}\n"
            #             )

            trg_sup_mxpool = targets[i] ### self.num_superpixel x C: multi-hot annotation
            
            out_sup_mxpool = out_sup_mxpool[empty_trg_mask[i]]
            trg_sup_mxpool = trg_sup_mxpool[empty_trg_mask[i]]

            top_one_preds = out_sup_mxpool * trg_sup_mxpool ### self.num_superpixel x C: 존재하는 multi-hot 으로 filtering

            top_one_preds_nonzero = top_one_preds[top_one_preds.nonzero(as_tuple=True)] ### 해당 value indexing
            num_valid += top_one_preds_nonzero.shape[0] ### valid pixel 개수 측정

            loss += -torch.log(top_one_preds_nonzero + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            raise NotImplementedError
        
## OnehotCEMultihotChoice
class MultiChoiceCE(nn.Module):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.eps = 1e-8
        self.temp = temperature

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        loss = 0
        num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            ### filtered outputs
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask):
                continue
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i][..., :-1] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)].detach() ### HW' x C : pixel-wise multi-hot annotation
            
            ### filter out empty target
            empty_trg_mask = torch.any(trg_pixel, dim=1).bool() ### HW'
            valid_output = valid_output[empty_trg_mask]
            trg_pixel = trg_pixel[empty_trg_mask]
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)
            num_valid += pos_pred.shape[0]
            loss += -torch.log(pos_pred + self.eps).sum()

        if self.reduction == 'mean':
            return loss / num_valid
        elif self.reduction == 'none':
            return loss, num_valid
        else:
            NotImplementedError

class OnehotCEMultihotChoice(MultiChoiceCE):
    def __init__(self, num_class, temperature=1.0, reduction='mean'):
        super().__init__(num_class, temperature, reduction)
        assert(self.reduction == 'mean')

    def forward(self, inputs, targets, superpixels, spmasks):
        ''' inputs:  N x C x H x W
            targets: N x self.num_superpiexl x C+1
            superpixels: N x H x W
            spmasks: N x H x W
        '''

        N, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        outputs = F.softmax(inputs / self.temp, dim=2) ### N x HW x C
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW: binary mask indicating current selected spxs
        oh_loss = 0
        oh_num_valid = 1
        mh_loss = 0
        mh_num_valid = 1

        for i in range(N):
            '''
            outputs[i] ### HW x C
            superpixels[i] ### HW x 1
            spmasks[i] ### HW x 1
            '''
            r''' skip this image if valid superpixel is not included '''
            valid_mask = spmasks[i] ### HW
            if not torch.any(valid_mask): continue ### empty image

            r''' calculate pixel-wise (CE, MC) loss jointly'''
            valid_output = outputs[i][valid_mask] ### HW' x C : class-wise prediction 중 valid 한 영역
            valid_superpixel = superpixels[i][valid_mask] ### HW' x 1 : superpixel id 중 valid 한 ID

            trg_sup = targets[i] ### self.num_superpixel x C: multi-hot annotation
            trg_pixel = trg_sup[valid_superpixel.squeeze(dim=1)] ### HW' x C : pixel-wise multi-hot annotation
            
            pos_pred = (valid_output * trg_pixel).sum(dim=1)

            r''' ce loss on one-hot spx '''
            onehot_trg = (1 == trg_pixel.sum(dim=1))
            if torch.any(onehot_trg):
                oh_pos_pred = pos_pred[onehot_trg]
                oh_loss += -torch.log(oh_pos_pred + self.eps).sum()
                oh_num_valid += oh_pos_pred.shape[0]

            r''' mc loss on multi-hot spx '''
            # multihot_trg = torch.logical_not(onehot_trg)
            multihot_trg = (1 < trg_pixel.sum(dim=1))
            if torch.any(multihot_trg):
                # assert(torch.all(multihot_trg == (1 < trg_pixel.sum(dim=1))))
                mh_pos_pred = pos_pred[multihot_trg]
                mh_loss += -torch.log(mh_pos_pred + self.eps).sum()
                mh_num_valid += mh_pos_pred.shape[0]

        return oh_loss / oh_num_valid, mh_loss / mh_num_valid
    

def pseudoLabels_generation_process(train_loader, model, SEED, n_round, kernel):
    model.eval()

    save_dir = f'./pseudo_labels/SEED={SEED}/round={n_round}'
    os.makedirs(name=save_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            labels = batch['labels'].to(DEVICE, dtype=torch.long)
            im_size = labels.shape[-2:]
            fname = batch['names'][0]
            feats = []
            outputs = []
            for msidx, img in enumerate(batch['image_list'][0]):
                img = img.to(DEVICE, dtype=torch.float32)
                feat, output = model.feat_forward(img[None, ...])
                if (len(batch['image_list'][0]) - 1 ) // 2 < msidx:
                    feat = transforms.RandomHorizontalFlip(p=1.0)(feat)
                    output = transforms.RandomHorizontalFlip(p=1.0)(output)

                feat = vision_F.resize(feat, im_size, InterpolationMode.BILINEAR)
                output = vision_F.resize(output, im_size, InterpolationMode.BILINEAR)

                feats.append(feat[0])
                outputs.append(output[0])

            feats = torch.stack(feats).mean(dim=0)
            outputs = torch.stack(outputs).mean(dim=0)

            feats = F.normalize(feats, dim=0) ### renormalize feat vector
            
            feats = feats[None, ...] ### add batch dimension
            outputs = outputs[None, ...]

            # images = batch['image_list'].to(self.device, dtype=torch.float32)
            # feats, outputs = self.net.feat_forward(images)

            r''' NN based pseudo label acquisition '''
            superpixels = batch['spx'].to(DEVICE)
            spmasks = batch['spmask'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            nn_pseudo_label = pseudo_label_generation(labels, feats, outputs, targets, spmasks, superpixels, kernel)
            ### ㄴ N x H x W

            r''' Save pseudo labels '''
            # if iteration == 38: import pdb; pdb.set_trace()
            # lbl_id = fname.split('/')[-1].split('.')[0]
            class_name, frame_nb = fname.split('/')
            if not os.path.exists(f"{save_dir}/{class_name}"):
                os.makedirs(f"{save_dir}/{class_name}")
            plbl_save = nn_pseudo_label[0].cpu().numpy().astype('uint8')
            pil_plbl_save = Image.fromarray(plbl_save)
            assert(list(im_size[::-1]) == list(pil_plbl_save.size))
            pil_plbl_save = vision_F.resize(pil_plbl_save, im_size, InterpolationMode.NEAREST)
            pil_plbl_save.save("{}/{}.png".format(save_dir, fname))

# def scatter_max(
#     src: torch.Tensor, index: torch.Tensor, dim: int = -1,
#     out = None,
#     dim_size = None):
#     return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)

def pseudo_label_generation(labels, feats, inputs, targets, spmasks, superpixels, kernel):
        r'''
        Args::
            feats: N x Channel x H x W
            inputs: N x C x H x W
            targets: N x self.num_superpiexl x C
            spmasks: N x H x W
            superpixels: N x H x W
            superpixel_smalls: N x H x W
            
            Apply max operation over predicted probabilities for each multi-hot label within the superpixel,
            and highlight selected top-1 pixel with its corresponding labels
            
        return::
            pseudo_label (torch.Tensor): pseudo label map to be evaluated
                                         N x H x W
            '''

        N, C, H, W = inputs.shape
        outputs = F.softmax(inputs, dim=1) ### N x C x H x W
        outputs = outputs.permute(0,2,3,1).reshape(N, -1, C) ### N x HW x C
        N, Ch, H, W = feats.shape
        feats = feats.permute(0,2,3,1).reshape(N, -1, Ch) ### N x HW x Ch
        superpixels_orig = superpixels.cpu().numpy()
        superpixels = superpixels.reshape(N, -1, 1) ### N x HW x 1
        spmasks = spmasks.reshape(N, -1) ### N x HW
        is_trg_multihot = (1 < targets.sum(dim=2)) ### N x self.num_superpixel

        nn_plbl = torch.ones_like(labels) * 255 ### N x H x W
        nn_plbl = nn_plbl.reshape(N, -1)

        for i in range(N):
            '''
            outputs[i] : HW x C
            feats[i] : HW x Ch
            superpixels[i] : HW x 1
            superpixel_smalls[i] : HW x 1
            targets[i] : self.num_superpiexl x C
            spmasks[i] : HW
            '''

            r''' filtered outputs '''
            if not torch.any(spmasks[i]): continue
            validall_superpixel = superpixels[i][spmasks[i]]
            # validall_trg_pixel = targets[i][validall_superpixel.squeeze(dim=1)]

            # multi_mask = is_trg_multihot[i][validall_superpixel.squeeze(dim=1)].detach()
            # valid_mask = spmasks[i].clone()
            # valid_mask[spmasks[i]] = multi_mask
            # if not torch.any(valid_mask): continue

            valid_mask = spmasks[i]

            valid_output = outputs[i][valid_mask] ### HW' x C
            valid_feat = feats[i][valid_mask] ### HW' x Ch
            vpx_superpixel = superpixels[i][valid_mask] ### HW' x 1
            multi_hot_target = targets[i] ### self.num_superpixel x C

            r''' get max pixel for each class within superpixel '''
            _, vdx_sup_mxpool = scatter_max(valid_output, vpx_superpixel, dim=0, dim_size=config['PATCH_NUMBER'])
            # _, vdx_sup_mxpool2 = max_scatter(valid_output, vpx_superpixel, dim=0, dim_size=config['PATCH_NUMBER'])
            # with open(PRINT_PATH, "a") as f:
            #     f.write(
            #         f"scatter ind diff: {torch.sum(vdx_sup_mxpool - vdx_sup_mxpool2)}\n"
            #         f"ind1: {vdx_sup_mxpool}\n"
            #         f"ind2: {vdx_sup_mxpool2}\n"
            #     )
            ### ㄴ self.num_superpixel x C: 각 (superpixel, class) pair 의 max 값을 가지는 index
           
            r''' filter invalid superpixels '''
            is_spx_valid = vdx_sup_mxpool[:,0] < valid_output.shape[0]
            ### ㄴ vpx_superpixel 에 포함되지 않은 superpixel id 에 대해서는 max index 가
            ### valid_output index 최대값 (==크기)로 잡힘. 이 값을 통해 쓸모없는 spx filtering            
            vdx_vsup_mxpool = vdx_sup_mxpool[is_spx_valid]
            ### ㄴ nvalidseg x C : index of max pixel for each class (for valid spx)
            multihot_vspx = multi_hot_target[is_spx_valid]
            ### ㄴ nvalidseg x C : multi-hot label (for valid spx)

            r''' Index conversion (valid pixel -> pixel) '''
            validex_to_pixdex = valid_mask.nonzero().squeeze(dim=1)
            ### ㄴ translate valid_pixel -> pixel space
            proto_vspx, proto_clsdex = multihot_vspx.nonzero(as_tuple=True)
            ### ㄴ valid superpixel index && valid class index
            top1_vdx = vdx_vsup_mxpool[proto_vspx, proto_clsdex]
            ### ㄴ vdx_sup_mxpool 중에서 valid 한 superpixel 과 target 에서의 valid index
            # top1_pdx = validex_to_pixdex[top1_vdx]
            # ### ㄴ max index 들을 pixel space 로 변환

            r''' Inner product between prototype features & superpixel features '''
            prototypes = valid_feat[top1_vdx]
            ### ㄴ nproto x Ch
            similarity = torch.mm(prototypes, valid_feat.T)
            ### ㄴ nproto x nvalid_pixels: 각 prototype 과 모든 valid pixel feature 사이의 유사도
            
            vspdex_to_spdex = is_spx_valid.nonzero(as_tuple=True)[0]
            # proto_spx = vspdex_to_spdex[proto_vspx] ### to calcualte equal operation in all index
            # multispx = validall_superpixel[multi_mask].squeeze(dim=1)

            # is_similarity_valid = torch.eq(proto_spx[..., None], multispx[None, ...])

            r''' Nearest prototype selection '''
            mxproto_sim, idx_mxproto_pxl = scatter_max(similarity, proto_vspx, dim=0)
            # mxproto_sim2, idx_mxproto_pxl2 = max_scatter(similarity, proto_vspx, dim=0)
            # with open(PRINT_PATH, "a") as f:
            #     f.write(
            #         f"scatter ind diff: {torch.sum(idx_mxproto_pxl - idx_mxproto_pxl2)}\n"
            #         f"scatter sim diff: {torch.sum(mxproto_sim - mxproto_sim2)}\n"
            #     )

            ### ㄴ nvalidspx x nvalid_pixels: pixel 별 가장 유사한 prototype 과의 similarity
            ### ㄴ nvalidspx x nvalid_pixels: pixel 별 가장 유사한 prototype id

            r''' Assign pseudo label of according prototype
            - idx_mxproto_pxl 중에서 각 pixel 이 해당하는 superpixel superpixel 의 값을 얻기
            - 이를 위해 우선 (superpixel -> valid superpixel)로 index conversion 을 만듦
            - pixel 별 superpixel id 를 pixel 별 valid superpixel id 로 변환 (=nearest_vspdex)
            - 각 valid superpixel 의 label 로 pseudo label assign (=plbl_vdx)
            - pseudo label map 의 해당 pixel 에 valid pixel 별 pseudo label 할당 (nn_plbl)
            '''
            spdex_to_vspdex = torch.ones_like(is_spx_valid) * -1
            vspx_ids, proto_counts = torch.unique(proto_vspx, return_counts=True)
            # import pdb; pdb.set_trace()
            # vspx_ids = proto_vspx
            # proto_counts = torch.tensor(vspx_ids.size())
            # spdex_to_vspdex[is_spx_valid] = vspx_ids
            try:
                spdex_to_vspdex[is_spx_valid] = vspx_ids
            except:
                print(is_spx_valid)
            vspdex_superpixel = spdex_to_vspdex[vpx_superpixel.squeeze(dim=1)]
            ### ㄴ 여기 vpx_superpixel 의 id value 는 superpixel 의 id 이다.
            nn_local_cls = idx_mxproto_pxl.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]
            nn_local_similarity = mxproto_sim.T[torch.arange(vspdex_superpixel.shape[0]), vspdex_superpixel]

            r''' Prototype similarity value & neighborhood spx id acquisition'''
            trg_vsup_median_sim = torch.zeros_like(multihot_vspx).float()
            spx_neighbor_ids = {}
            offset = 0
            for vspx in range(vspx_ids.shape[0]):
                r''' Get similarity threshold value for each prototype
                - Get index value of max similarity value for each superpixel
                - For each prototype within superpixel, calculate median simialrity threshold
                '''
                indices = torch.masked_select(nn_local_cls, (vspdex_superpixel == vspx))
                similarity = torch.masked_select(nn_local_similarity, (vspdex_superpixel == vspx))
                for proto_ids in range(proto_counts[vspx]):
                    proto_ids_ = proto_ids + offset
                    if torch.any(indices==proto_ids_):
                        similarity_threshold = torch.median(similarity[indices==proto_ids_])
                    else:
                        similarity_threshold = 1.0
                    trg_vsup_median_sim[vspx, proto_clsdex[proto_ids_]] = similarity_threshold
                offset += proto_counts[vspx]

                r''' Get ids of surrounding superpixels
                - Get binary mask of current superpixel id --> Dilation -> Id collection
                '''
                spx_id = vspdex_to_spdex[vspx].item()
                spx_id_binmap = (superpixels_orig[i] == spx_id)
                spx_id_binmap_dilate = binary_dilation(spx_id_binmap, kernel)
                spx_map_tensor = torch.from_numpy(superpixels_orig[i])
                dilated_mask = torch.from_numpy(spx_id_binmap_dilate)
                selected = torch.masked_select(spx_map_tensor, dilated_mask)
                spx_neighbor_ids[spx_id] = torch.unique(selected).cuda()

                r''' TODO: Get ids of surrounding superpixels (larger superpixel)
                - Get id from larger superpixel -> select maximum index
                '''

            r''' TODO: 인접한 selected superpixel 예외 처리 '''

            r''' Similarity calculation & pseudo label assignment ''' 
            # spx_neighbor_ids = {i:j for i,j in spx_neighbor_ids.items()}
            for vspx in range(vspx_ids.shape[0]):
                r''' Get similarity betwen prototype and sourrounding regions
                - prototypes within superpixel indexing
                - sourrouding feature filtering
                - similarity calculation
                '''
                spx_id = vspdex_to_spdex[vspx].item()
                curr_spx_prototypes = prototypes[proto_vspx == vspx]
                surr_spx_mask = torch.isin(superpixels[i], spx_neighbor_ids[spx_id]).squeeze(dim=1)
                surr_feature = feats[i][surr_spx_mask] ### HW' x Ch
                curr_spx_similarity = torch.mm(curr_spx_prototypes, surr_feature.T)

                r''' Pseduo label generation from similarity and assign them into plbl map
                - prototype argmax
                - prototype index -> pseudo label index
                - Thresholding with prototype-wise threshold
                - (Skip) Exclude within superpixel indices from filtering
                - nn_plbl saving: surr_spx_mask index -> pixel index 
                '''
                prototype_cls = proto_clsdex[proto_vspx == vspx]
                plbl_prototype_id = curr_spx_similarity.argmax(dim=0)
                plbl_unfiltered = prototype_cls[plbl_prototype_id]
                similarity_threshold = trg_vsup_median_sim[vspx, prototype_cls]
                is_plbl_valid = torch.any((similarity_threshold[..., None] < curr_spx_similarity), dim=0)
                # is_plbl_valid = torch.ones_like(is_plbl_valid).bool() ### TODO: Debug  for similarity

                surrounding_index_to_pixel_index = surr_spx_mask.nonzero(as_tuple=True)[0]
                filtered_pixel_index = surrounding_index_to_pixel_index[is_plbl_valid]
                plbl_filtered = plbl_unfiltered[is_plbl_valid]
                nn_plbl[i, filtered_pixel_index] = plbl_filtered

                # TODO: if not args.within_spx_filtering:

            plbl_vdx = proto_clsdex[nn_local_cls]
            nn_plbl[i, validex_to_pixdex] = plbl_vdx

        nn_plbl = nn_plbl.reshape(N, H, W)
        
        return nn_plbl


class TestTimeAugmentation:
    r'''
    Usage:
        tta = TestTimeAugmentation()
        image = Image.open('path/to/your/image.jpg')
        augmented_images = tta(image)
    '''

    def __init__(self):
        self.rescale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

    def __call__(self, image):
        original_width, original_height = image.size
        augmented_images = []

        for flip in [False, True]:
            for factor in self.rescale_factors:
                transform = transforms.Compose([
                    transforms.Resize((int(factor * original_height), int(factor * original_width))),
                    transforms.RandomHorizontalFlip(p=1.0) if flip else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225]),
                ])

                augmented_image = transform(image)
                augmented_images.append(augmented_image)

        return augmented_images