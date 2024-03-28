import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import copy
import json
import pickle
import time
from datetime import timedelta

from utils import set_random, compute_entropy_v2
from training_utils import FocalLoss, validate_v2
from model import UNet
from model_v2 import deeplabv3_resnet50
from model_v3 import MC_DeepLab
from model_v4 import create_ViT_model
from create_dataset import (
    get_datasets,
    get_image_curr_labeled,
)
from routes import (
    CONTINUE_PATH,
    CONTINUE_FOLDER,
    PRINT_PATH,
    IGNORE_INDEX
)

#from other_queries import (
#    GT_query,
#    GTxSim_query,
#    density_entropy_query,
#    density_classEntropyV2_query,
#    density_query,
#    entropy_query,
#    BvSB_patch_query,
#    class_entropy_query,
#    class_entropy_patch_query,
#    class_entropy_ML_pred_patch_query,
#    class_entropy_video_query,
#    random_query,
#    coreset_query,
#    coreset_entropy_query,
#    COWAL_center_query,
#    k_means_entropy_query,
#    COWAL_entropy_query,
#    COWAL_entropy_video_query,
#    COWAL_entropy_patch_query,
#    MC_dropout_query,
#    BALD_query,
#    suggestive_annotation_query,
#   suggestive_annotation_patch_query,
#    VAAL_query,
#    oracle_query,
#    COWAL_classEntropy_query,
#    COWAL_classEntropy_video_query,
#    COWAL_classEntropy_v2_query,
#    COWAL_classEntropy_v2_video_query,
#    COWAL_classEntropy_v2_1_query,
#    COWAL_classEntropy_v2_2_query,
#    COWAL_classEntropy_v2_2_video_query,
#    COWAL_classEntropy_patch_query,
#    COWAL_classEntropy_patch_v2_query,
#    RIPU_PA_query,
#    entropy_patch_query,
#    revisiting_query,
#    revisiting_v2_query,
##    revisiting_adaptiveSP_query,
#    CBAL_query,
#    CBAL_v2_query,
#    pixelBal_query,
#    pixelBal_v2_query,
#    BvSB_patch_v2_query,
#)
from config import config

def query_new(
    copy_model,
    curr_labeled,
    train_dataset,
    unlabeled_dataset,
    all_train_dataset,
    ML_preds,
    SAMPLING,
    NUM_QUERY,
    n_round,
    SEED,
    step=1,
    metric="DICE",
    nb_iter=1,
    patch_number=None,
):
    if SAMPLING not in ["COWAL_entropy", 
                        'revisiting',
                        'revisiting_v2', 
                        'COWAL_entropy_video', 
                        'BvSB_patch',
                        'BvSB_patch_v2',
                        'class_entropy_ML_pred_patch', 
                        'revisiting_adaptiveSP', 
                        'pixelBal', 
                        'pixelBal_v2',
                        'CBAL',
                        'CBAL_v2',
                        'random']:
        ML_entropy, ML_class_entropy = compute_entropy_v2(copy_model, 
                                                       all_train_dataset, 
                                                       patch_number=patch_number, 
                                                       curr_selected_patches=train_dataset.curr_selected_patches)
        torch.save(ML_entropy, f"results/ML_entropy_SEED={SEED}_round={n_round}.pt")
        torch.save(ML_class_entropy, f"results/ML_class_entropy_SEED={SEED}_round={n_round}.pt")

        unlabeled_ML_entropy = {}
        unlabeled_ML_class_entropy = {}
        for path, label_path in unlabeled_dataset.data_pool:
            frame_id = path.split("/")[-2] + "/" + path.split("/")[-1].split(".")[0]
            unlabeled_ML_entropy[frame_id] = ML_entropy.get(frame_id, 0)
            unlabeled_ML_class_entropy[frame_id] = ML_class_entropy[frame_id]

    new_selected_patches = None
    if SAMPLING == "random":
        new_labeled, new_selected_patches = random_query(
            unlabeled_dataset, 
            num_query=NUM_QUERY, 
            patch_number=patch_number, 
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "density":
        new_labeled = density_query(
            train_dataset,
            unlabeled_dataset,
            num_query=NUM_QUERY,
        )
    elif SAMPLING == "entropy":
        new_labeled = entropy_query(
            unlabeled_ML_entropy,
            num_query=NUM_QUERY,
        )
    elif SAMPLING == "entropy_patch":
        new_labeled, new_selected_patches = entropy_patch_query(
            unlabeled_ML_entropy,
            num_query=NUM_QUERY,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "BvSB_patch":
        new_labeled, new_selected_patches = BvSB_patch_query(
            model=copy_model,
            dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "BvSB_patch_v2":
        new_labeled, new_selected_patches = BvSB_patch_v2_query(
            model=copy_model,
            unlabeled_dataset=unlabeled_dataset,
            train_dataset=train_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "class_entropy":
        new_labeled = class_entropy_query(
            unlabeled_ML_class_entropy, 
            train_dataset, 
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
        )
    elif SAMPLING == "class_entropy_patch":
        new_labeled, new_selected_patches = class_entropy_patch_query(
            unlabeled_ML_class_entropy, 
            train_dataset, 
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "class_entropy_ML_pred_patch":
        new_labeled, new_selected_patches = class_entropy_ML_pred_patch_query(
            copy_model,
            all_train_dataset, 
            train_dataset, 
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "CBAL":
        new_labeled, new_selected_patches = CBAL_query(
               train_dataset, 
               unlabeled_dataset, 
               copy_model, 
               NUM_QUERY,
               n_round, 
               SEED,
               patch_number=patch_number,
               smooth=1e-7
        )
    elif SAMPLING == "CBAL_v2":
        new_labeled, new_selected_patches = CBAL_v2_query(
               train_dataset, 
               unlabeled_dataset, 
               copy_model, 
               NUM_QUERY,
               n_round, 
               SEED,
               patch_number=patch_number,
               smooth=1e-7
        )
    elif SAMPLING == "class_entropy_video":
        new_labeled = class_entropy_video_query(
            unlabeled_ML_class_entropy,
            curr_labeled,
            train_dataset,
            all_train_dataset, 
            num_query=NUM_QUERY,
        )
    elif SAMPLING == "density-entropy":
        new_labeled = density_entropy_query(
            unlabeled_ML_entropy,
            curr_labeled,
            num_query=NUM_QUERY,
        )
    elif SAMPLING == "density_classEntropy":
        new_labeled = density_classEntropyV2_query(
            unlabeled_ML_class_entropy,
            curr_labeled,
            num_query=NUM_QUERY,
        )
    elif SAMPLING == "OF":
        new_labeled = OF_query(
            curr_labeled,
            train_dataset,
            unlabeled_dataset,
            ML_preds,
            num_query=NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            step=step,
            metric=metric,
        )
    elif SAMPLING == "coreset":
        new_labeled = coreset_query(
            all_train_dataset,
            train_dataset,
            copy_model,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            use_sphere=config["USE_SPHERE"],
        )
    elif SAMPLING == "coreset_entropy":
        new_labeled = coreset_entropy_query(
            unlabeled_ML_entropy,
            all_train_dataset,
            train_dataset,
            copy_model,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            use_sphere=config["USE_SPHERE"],
        )
    elif SAMPLING == "density-OF":
        new_labeled = density_OF_query(
            ML_preds,
            curr_labeled,
            train_dataset,
            unlabeled_dataset,
            num_query=NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            step=step,
            metric=metric,
        )
    elif SAMPLING == "GT_query":
        new_labeled = GT_query(NUM_QUERY, n_round=n_round, SEED=SEED)
    elif SAMPLING == "GTxSim_query":
        new_labeled = GTxSim_query(
            train_dataset, unlabeled_dataset, NUM_QUERY, n_round=n_round, SEED=SEED
        )
    elif SAMPLING == "RAFT_query":
        new_labeled = RAFT_query(
            curr_labeled,
            train_dataset,
            unlabeled_dataset,
            ML_preds,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            step=step,
            metric=metric,
        )
    elif SAMPLING == "RAFTxSim_query":
        new_labeled = RAFTxSim_query(
            curr_labeled,
            train_dataset,
            unlabeled_dataset,
            ML_preds,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            step=step,
            metric=metric,
        )
    elif SAMPLING == "MC_dropout":
        new_labeled = MC_dropout_query(
            copy_model,
            unlabeled_dataset,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            smooth=1e-12,
        )
    elif SAMPLING == "VAAL":
        new_labeled = VAAL_query(
            train_dataset,
            unlabeled_dataset,
            NUM_QUERY,
            n_round=n_round,
            num_iter=nb_iter,
        )
    elif SAMPLING == "BALD":
        new_labeled = BALD_query(
            copy_model,
            unlabeled_dataset,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            smooth=1e-12,
        )
    elif SAMPLING == "suggestive_annotation":
        new_labeled = suggestive_annotation_query(
            unlabeled_ML_entropy,
            unlabeled_dataset,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            smooth=1e-12,
        )
    elif SAMPLING == "suggestive_annotation_patch":
        new_labeled, new_selected_patches = suggestive_annotation_patch_query(
            unlabeled_ML_entropy,
            unlabeled_dataset,
            NUM_QUERY,
            n_round=n_round,
            SEED=SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            smooth=1e-12,
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "k_means_entropy":
        new_labeled = k_means_entropy_query(
            copy_model,
            unlabeled_dataset,
            NUM_QUERY,
            config["NB_K_MEANS_CLUSTER"],
            n_round,
            SEED,
            smooth=1e-7,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
        )
    elif SAMPLING == "COWAL_center":
        new_labeled = COWAL_center_query(
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            use_kmedian=config["USE_KMEDIAN"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_entropy":
        new_labeled = COWAL_entropy_query(
            copy_model,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            smooth=1e-7,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_entropy_video":
        new_labeled = COWAL_entropy_video_query(
            copy_model,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            smooth=1e-7,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_entropy_patch":
        new_labeled, new_selected_patches = COWAL_entropy_patch_query(
            unlabeled_ML_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            smooth=1e-7,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "COWAL_classEntropy":
        new_labeled = COWAL_classEntropy_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_video":
        new_labeled = COWAL_classEntropy_video_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_v2":
        new_labeled = COWAL_classEntropy_v2_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_v2_video":
        new_labeled = COWAL_classEntropy_v2_video_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_v2_1":
        new_labeled = COWAL_classEntropy_v2_1_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_v2_2":
        new_labeled = COWAL_classEntropy_v2_2_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_v2_2_video":
        new_labeled = COWAL_classEntropy_v2_2_video_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
        )
    elif SAMPLING == "COWAL_classEntropy_patch":
        new_labeled, new_selected_patches = COWAL_classEntropy_patch_query(
            unlabeled_ML_class_entropy,
            train_dataset,
            all_train_dataset,
            NUM_QUERY,
            n_round,
            SEED,
            embedding_method=config["EMBEDDING_METHOD"],
            weight_path=config["PRETRAINED_WEIGHT_PATH"],
            sphere=config["USE_SPHERE"],
            hung_matching=config["HUNG_MATCHING"],
            patch_number=patch_number,
            patch_shape=config['PATCH_SHAPE']
        )
    elif SAMPLING == "RIPU_PA":
        new_labeled = RIPU_PA_query(
            model=copy_model,
            dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            n_round=n_round,
        )
    elif SAMPLING == "revisiting":
        new_labeled, new_selected_patches = revisiting_query(
            model=copy_model,
            dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
        )
    elif SAMPLING == "revisiting_v2":
        new_labeled, new_selected_patches = revisiting_v2_query(
            model=copy_model,
            unlabeled_dataset=unlabeled_dataset,
            train_dataset=train_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
        )
    elif SAMPLING == "revisiting_adaptiveSP":
        new_labeled, new_selected_patches = revisiting_adaptiveSP_query(
            model=copy_model,
            dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
        )
    elif SAMPLING == "pixelBal":
        new_labeled, new_selected_patches = pixelBal_query(
            model=copy_model,
            dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
        )
    elif SAMPLING == "pixelBal_v2":
        new_labeled, new_selected_patches = pixelBal_v2_query(
            model=copy_model,
            unlabeled_dataset=unlabeled_dataset,
            num_query=NUM_QUERY,
            train_dataset=train_dataset,
            SEED=SEED,
            n_round=n_round,
            patch_number=patch_number,
        )
    elif SAMPLING == "oracle":
        new_labeled = oracle_query()
    return new_labeled, new_selected_patches


def main():
    if os.path.isfile(PRINT_PATH):
        os.remove(PRINT_PATH)

    if not os.path.exists(CONTINUE_FOLDER):
        os.mkdir(CONTINUE_FOLDER)

    with open(PRINT_PATH, "a") as f:
        f.write(f"========== config ============\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    if config["CONTINUE_TRAIN"]:
        checkpoint = torch.load(CONTINUE_PATH)
        config["START_SEED"] = checkpoint["seed"]

    for SEED in range(config["START_SEED"], config["TOTAL_SEEDS"]):
        with open(PRINT_PATH, "a") as f:
            f.write(f"========== SEED: {SEED} ============\n")
        set_random(SEED)

        ### SPLIT TRAIN SET ###
        curr_labeled, curr_selected_patches, train_data, val_data, test_data, TRAIN_SEQ, VAL_SEQ, TEST_SEQ = get_image_curr_labeled(config, SEED)

        ################### define patience iter ###################
        if config['N_LABEL'] > 1 and config['DATASET'] not in ['auris', 'intuitive', 'cityscapes', 'pascal_VOC'] and not config['BALANCED_INIT']:                                                 
            config['PATIENCE_ITER'] = 2 * (config['INIT_NUM_VIDEO'] + (config['NUM_ROUND'] - 1) * config['NUM_QUERY'])
        # # elif config['N_LABEL'] > 1 and config['DATASET'] != 'auris' and config['BALANCED_INIT']:
        #     config['PATIENCE_ITER'] = 2 * ((config['N_LABEL'] - 1) * 2 + (config['NUM_ROUND'] - 1) * config['NUM_QUERY'])
        config["PATIENCE_ITER"] = config["PATIENCE_ITER"] // config["BATCH_SIZE"]

        if config['DATASET'] == 'auris': # auris v11
            auris_patience_iter = [132, 90, 130, 130, 130, 127, 142, 113, 113, 113]
            config['PATIENCE_ITER'] = auris_patience_iter[SEED]
        ##################### 
            
        start_round = 0
        test_scores = []

        if config["CONTINUE_TRAIN"]:
            start_round = checkpoint["round"]
            curr_labeled = checkpoint["curr_labeled"]
            curr_selected_patches = checkpoint["curr_selected_patches"]

        for n_round in range(start_round, config["NUM_ROUND"]):
            with open(PRINT_PATH, "a") as f:
                f.write(f"======== ROUND: {n_round}, SEED: {SEED} ========\n")
            set_random(SEED)  # to have the same model parameters

            ### SET UP DATASET ###
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
                all_train_dataset,
                train_dataset_noAug,
                unlabeled_dataset,
            ) = get_datasets(
                config,
                curr_labeled,
                curr_selected_patches=curr_selected_patches,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                TRAIN_SEQ=TRAIN_SEQ,
                VAL_SEQ=VAL_SEQ,
                TEST_SEQ=TEST_SEQ,
            )
            if config["NUM_ROUND"] == 1:
                config["PATIENCE_ITER"] = len(train_dataloader)
            
            ### DEFINE MODEL ###
            if True:
                if config['MODEL_ARCH'] == 'deeplabv3_resnet101':
                    model = MC_DeepLab(
                        num_classes=config["N_LABEL"],
                        backbone="resnet101",
                        output_stride=16,
                        sync_bn=False,
                        pretrained=True,
                    ).to(config["DEVICE"])
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"MC DeepLab 101, model param: {next(model.parameters())[0, 0, 0, 0]}\n"\
                            + f"patience iter: {config['PATIENCE_ITER']}\n"
                        )
                elif config['MODEL_ARCH'] == 'deeplabv3_resnet50':
                    model = deeplabv3_resnet50(
                        pretrained=True, num_classes=config["N_LABEL"]
                    ).to(config["DEVICE"])
                    # model_weights = torch.load(config['SEGMENTATION_MODEL_WEIGHT_PATH'])
                    # model_weights = model_weights['state_dict']
                    # log = model.load_state_dict(model_weights, strict=False)
                    log = ''
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"model param: {next(model.parameters())[0, 0, 0, 0]}\n"\
                            + f"{log}\n"\
                            + f"patience iter: {config['PATIENCE_ITER']}\n"
                        )
                elif config['MODEL_ARCH'] == 'unet':
                    model = UNet(n_channels=3, n_classes=config['N_LABEL'], bilinear=False).to(config["DEVICE"])
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"model param: {next(model.parameters())[0, 0, 0, 0]}\n"\
                            + f"patience iter: {config['PATIENCE_ITER']}\n"
                        )
                elif config['MODEL_ARCH'] == 'vit':
                    model = create_ViT_model(config, pretrained_path='../pretrained_models/vit/Seg_S_Mask_16.pth')
                    model = model.to(config["DEVICE"])
                    with open(PRINT_PATH, "a") as f:
                        f.write(
                            f"VIT model param: {next(model.parameters())[0, 0, 0]}\n"\
                            + f"patience iter: {config['PATIENCE_ITER']}\n"
                        )
                if n_round > 0:# and config['MODEL_ARCH'] != 'vit':
                    model.load_state_dict(torch.load(CONTINUE_FOLDER + f"best_val.pth"))
                    with open(PRINT_PATH, "a") as f:
                        f.write(f"Loaded: best_val.pth\n")

                copy_model = copy.deepcopy(model).to(config["DEVICE"])
                copy_model.eval()
                optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
                criterion = nn.BCEWithLogitsLoss() if config['N_LABEL'] == 1 else FocalLoss()

            start_epoch = 0
            best_val_score = 0
            best_val_class_dices = {}
            patience = 0
            curr_iter = 0
            nb_iter = 0

            ### resume training ###
            if config["CONTINUE_TRAIN"]:
                best_val_score = checkpoint["best_val_score"]
                best_val_class_dices = checkpoint["best_val_class_dices"]
                start_epoch = checkpoint["epoch"]
                patience = checkpoint["patience"]
                curr_iter = checkpoint["curr_iter"]
                nb_iter = checkpoint["nb_iter"]

                model.load_state_dict(checkpoint["state_dict"])
                copy_model.load_state_dict(checkpoint["copy_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"----- model loaded at epoch {start_epoch}-----\n"
                        + f"----- curr iter is {curr_iter}-----\n"
                        f"----- total iter is {nb_iter}-----\n"
                        f"----- patience is {patience}-----\n"
                        f"----- best val score is {best_val_score}-----\n"
                    )
                config["CONTINUE_TRAIN"] = False
                checkpoint = None

            ### TRAIN THE MODEL ###
            val_scores = []
            shuffle_seed = (11 * SEED + n_round) * config["EPOCHS"]

            print_first_sample = True
            for epoch in range(start_epoch, config["EPOCHS"]):
                model.train()
                break_ = False
                # a unique seed for every triplet of (SEED, n_round, epoch)
                set_random(shuffle_seed + epoch)
                # num of batches
                for i, (image, mask, names) in enumerate(train_dataloader):

                    if patience == config["MAX_PATIENCE"]:
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"=== MAX PATIENCE REACHED, best val score: {best_val_score}\n"
                            )
                        break_ = True
                        break

                    if print_first_sample and i == 0:
                        print_first_sample = False
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"=== first sample at epoch {epoch + 1}: {image[0, 0, 120, 120]}\n"
                            )

                    image, mask = image.to(config["DEVICE"]), mask.to(config["DEVICE"])

                    optimizer.zero_grad()
                    outputs = model(image)
                    if config["N_LABEL"] == 1:
                        outputs = outputs.squeeze(1)
                        mask = mask.squeeze(1)
                    loss = criterion(outputs, mask)
                    loss.backward()
                    optimizer.step()

                    copy_state_dict = copy_model.state_dict()
                    for k, v in model.state_dict().items():
                        copy_state_dict[k] = (
                            config["RUNNING_COEF"] * copy_state_dict[k]
                            + (1 - config["RUNNING_COEF"]) * v
                        )
                    copy_model.load_state_dict(copy_state_dict)

                    nb_iter += 1
                    curr_iter += 1

                    if curr_iter >= config["PATIENCE_ITER"]:
                        curr_iter = 0
                        val_loss, val_score, val_class_dices = validate_v2(
                            copy_model,
                            val_dataloader,
                            criterion,
                            metric=config['EVAL_METRIC'],
                            num_classes=config["N_LABEL"],
                            dataset_name=config["DATASET"],
                        )
                        val_scores.append(val_score)
                        np.save(
                            f"results/val_scores_SEED={SEED}_round={n_round}.npy",
                            val_scores,
                        )
                        with open(PRINT_PATH, "a") as f:
                            f.write(
                                f"----- Epoch {epoch + 1}, {nb_iter} iter:  Val {config['EVAL_METRIC']} is {val_score}, train loss is {loss.item()}, val loss is {val_loss}\n"
                            )

                        if val_score > best_val_score:
                            with open(PRINT_PATH, "a") as f:
                                f.write(
                                    f"Best Val changed from {best_val_score} to {val_score}\n"
                                )
                            best_val_score = val_score
                            best_val_class_dices = val_class_dices
                            patience = 0

                            torch.save(
                                copy_model.state_dict(),
                                CONTINUE_FOLDER + f"best_val.pth",
                            )

                        elif val_score <= best_val_score:
                            patience += 1
                            with open(PRINT_PATH, "a") as f:
                                f.write(f"Patience: {patience}\n")

                        ### SAVE MODELS ###
                        state = {
                            "round": n_round,
                            "seed": SEED,
                            "epoch": epoch + 1,
                            "patience": patience,
                            "curr_iter": curr_iter,
                            "nb_iter": nb_iter,
                            "curr_labeled": curr_labeled,
                            "curr_selected_patches": curr_selected_patches,
                            "best_val_score": best_val_score,
                            "best_val_class_dices": best_val_class_dices,
                            "state_dict": model.state_dict(),
                            "copy_state_dict": copy_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                        torch.save(state, CONTINUE_PATH)

                if break_:
                    break

            ### test evaluation with the best val model
            if True:
                copy_model.load_state_dict(
                    torch.load(CONTINUE_FOLDER + f"best_val.pth")
                )

                ### test evaluation ###
                _, test_score, test_class_dices = validate_v2(
                    copy_model,
                    test_dataloader,
                    criterion,
                    metric=config['EVAL_METRIC'],
                    num_classes=config["N_LABEL"],
                    dataset_name=config["DATASET"],
                )

                if config["N_LABEL"] > 1:
                    with open(
                        f"results/test_class_dices_SEED={SEED}_round={n_round}.pickle",
                        "wb",
                    ) as f:
                        pickle.dump(test_class_dices, f)

                test_scores.append(test_score)
                np.save(f"results/test_score_SEED={SEED}.npy", test_scores)

                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"=== END OF ROUND {n_round}, test score: {test_score}\n"
                    )

            ### QUERY NEW LABELS ###
            if config["NUM_ROUND"] > 1:
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"====== unlabeled dataset length: {len(unlabeled_dataset)}\n"
                    )
                
                with open(
                    f"results/curr_labeled_SEED={SEED}_round={n_round}.json", "w"
                ) as f:
                    json.dump(curr_labeled, f)
                with open(
                    f"results/curr_selected_patches_SEED={SEED}_round={n_round}.json", "w"
                ) as f:
                    json.dump(curr_selected_patches, f)

                new_labeled, new_selected_patches = query_new(
                    copy_model,
                    curr_labeled,
                    train_dataset_noAug,
                    unlabeled_dataset,
                    all_train_dataset,
                    ML_preds=None,
                    SAMPLING=config["SAMPLING"],
                    NUM_QUERY=config["NUM_QUERY"],
                    n_round=n_round,
                    SEED=SEED,
                    step=config["STEP_SIZE"],
                    metric=config["SAMPLING_METRIC"],
                    nb_iter=nb_iter,
                    patch_number=config["PATCH_NUMBER"],
                )

                with open(
                    f"results/new_labeled_SEED={SEED}_round={n_round}.json", "w"
                ) as f:
                    json.dump(new_labeled, f)
                with open(
                    f"results/new_selected_patches_SEED={SEED}_round={n_round}.json", "w"
                ) as f:
                    json.dump(new_selected_patches, f)

                if config['PATCH_NUMBER'] is None and not config['PIXEL_SAMPLING']:
                    assert len(np.concatenate(list(new_labeled.values()))) == config['NUM_QUERY']

                count_dict = {k: len(v) for k, v in new_labeled.items()}
                example_frames = next(iter(new_labeled.values()))
                example_frames = sorted(example_frames, key=lambda x: int(x))
                if config['DATASET'] not in ['cityscapes', 'auris', 'intuitive']:
                    example_frames = example_frames[:10]
                with open(PRINT_PATH, "a") as f:
                    f.write(
                        f"-- number of video sampled: {len([k for k,v in count_dict.items() if v !=0])}\n"
                        + f"-- sampled frames: {example_frames}\n"
                        f"-- number of frames per video: {count_dict}\n"
                    )
                
                for k, v in new_labeled.items():
                    curr_labeled[k] = curr_labeled[k] + v
                    curr_labeled[k] = list(set(curr_labeled[k]))

                if config['PATCH_NUMBER'] is not None:
                    for k, v in new_selected_patches.items():
                        curr_selected_patches[k] = curr_selected_patches.get(k, []) + v
                        curr_selected_patches[k] = list(set(curr_selected_patches[k]))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    total_time_seconds = end_time - start_time
    total_time_str = str(timedelta(seconds=total_time_seconds))

    # Open the file in read mode and store lines
    with open(PRINT_PATH, "r") as f:
        lines = f.readlines()

    # Add the time information at the beginning
    lines.insert(0, f"Total time: {total_time_str}\n")

    # Write lines back to the file
    with open(PRINT_PATH, "w") as f:
        for line in lines:
            f.write(line)
