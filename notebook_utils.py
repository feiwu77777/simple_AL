import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
import json
from PIL import Image
import torch
import collections
from utils import average_center
import torch.nn.functional as F
import pickle

from utils import set_random, euc_distance
from create_dataset import divide_data_split_auris, divide_data_split_youtube
from routes import (
    CLASS_ID_TYPE,
    FRAME_KEYWORD,
    FRAME_KEYWORD_YOUTUBE,
    CLASS_ID_TYPE_YOUTUBE,
)
from config import config, img_size_youtube
from torchvision import transforms as T
import matplotlib.patches as patches
from torchvision.transforms import InterpolationMode as Inter

from sklearn.manifold import TSNE

from routes import IMG_PATH, LAB_PATH, FILE_TYPE, ZFILL_NB


def draw_sampled_frames(
    curr_labels,
    new_labels,
    exp_folder,
    name_to_nb,
    name="",
    img_path=IMG_PATH,
    frame_keyword=FRAME_KEYWORD,
    color="green",
    title=None,
    save_name=None,
    dataset_category="auris",
):
    void = " "
    figure(figsize=(15, 8), dpi=80)
    max_max_ind = 0
    for video_id, video_nb in name_to_nb.items():
        frames = os.listdir(f"{img_path}{video_id}/")
        frames = [f for f in frames if f != ".DS_Store"]
        frames = sorted(
            frames, key=lambda x: int(x[len(frame_keyword) : -len(FILE_TYPE)])
        )
        frames = [f[len(frame_keyword) : -len(FILE_TYPE)] for f in frames]
        # min_ind = int(frames[0][len(FRAME_KEYWORD):-len(FILE_TYPE)])
        # max_ind = int(frames[-1][len(FRAME_KEYWORD):-len(FILE_TYPE)])
        min_ind = 0
        max_ind = len(frames)
        if max_ind > max_max_ind:
            max_max_ind = max_ind
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in new_labels and len(new_labels[video_id]) > 0:
            for frame in new_labels[video_id]:
                ind = frames.index(frame)
                plt.plot([video_nb], [ind], marker="o", markersize=12, color=color)
        if video_id in curr_labels and len(curr_labels[video_id]) > 0:
            for frame in curr_labels[video_id]:
                ind = frames.index(frame)
                plt.plot([video_nb], [ind], marker="X", markersize=12, color="black")
        else:
            plt.plot([video_nb], [-10], marker="o", markersize=3, color="white")

    if title is None:
        title = name[len(exp_folder + "AL_") :]
    # make legend for color
    plt.plot(
        [video_nb],
        [-500],
        marker="o",
        markersize=12,
        color=color,
        label="newly sampled frames",
    )
    plt.plot(
        [video_nb],
        [-500],
        marker="X",
        markersize=12,
        color="black",
        label="already sampled frames",
    )
    plt.legend(fontsize=20)
    plt.ylim(-5, max_max_ind + 10)
    # plt.xticks(rotation=70)
    # plt.title(title, fontsize=20)
    # plt.ylabel('frame index', fontsize=20)
    # plt.xlabel('video number', fontsize=20)

    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=30)
    plt.ylabel("frame index", fontsize=30)
    plt.xlabel("video number", fontsize=30)


def draw_sampled_frames_compare(
    name1,
    name2,
    round_,
    SEED,
    exp_folder,
    name_to_nb,
    img_path=IMG_PATH,
    color1="orange",
    color2="magenta",
    label1=None,
    label2=None,
    save_name=None,
    title=None,
):
    new_labels1 = get_sampled_labels(name1 + "/", path2="new", round_=round_, seed=SEED)
    curr_labels1 = get_sampled_labels(
        name1 + "/", path2="curr", round_=round_, seed=SEED
    )
    new_labels2 = get_sampled_labels(name2 + "/", path2="new", round_=round_, seed=SEED)
    curr_labels2 = get_sampled_labels(
        name2 + "/", path2="curr", round_=round_, seed=SEED
    )

    void = " "
    for k, v in new_labels1.items():
        curr_labels1[k] = curr_labels1.get(k, []) + v

    for k, v in new_labels2.items():
        curr_labels2[k] = curr_labels2.get(k, []) + v

    figure(figsize=(15, 8), dpi=80)
    for video_id in name_to_nb.keys():
        frames = os.listdir(f"{img_path}{video_id}/")
        frames = [f for f in frames if f != ".DS_Store"]
        frames = sorted(
            frames, key=lambda x: int(x[len(FRAME_KEYWORD) : -len(FILE_TYPE)])
        )
        frames = [f[len(FRAME_KEYWORD) : -len(FILE_TYPE)] for f in frames]
        min_ind = 0
        max_ind = len(frames)
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in curr_labels1:
            for frame in curr_labels1[video_id]:
                ind = frames.index(frame)
                plt.plot([video_id], [ind], marker="o", markersize=16, color=color1)
        if video_id in curr_labels2:
            for frame in curr_labels2[video_id]:
                ind = frames.index(frame)
                plt.plot([video_id], [ind], marker="X", markersize=12, color=color2)
        elif video_id not in curr_labels1 and video_id not in curr_labels2:
            plt.plot([video_id], [0], marker="o", markersize=3, color="white")

    if label1 is None:
        label1 = name1[len(exp_folder + "AL_") :]
    if label2 is None:
        label2 = name2[len(exp_folder + "AL_") :]
    if title is None:
        f"Training frames at round {round_ + 1}"

    # make legend for color
    plt.plot([video_id], [-15], marker="o", markersize=16, color=color1, label=label1)
    plt.plot([video_id], [-15], marker="X", markersize=12, color=color2, label=label2)
    # legend fontsize
    plt.legend(fontsize=25)

    plt.ylim(-10)
    plt.xticks(rotation=30, fontsize=23)
    plt.yticks(fontsize=25)
    plt.title(title, fontsize=30)
    plt.ylabel("frame index", fontsize=30)
    if save_name is not None:
        plt.savefig(f"../../../MICCAI/{save_name}.pdf", bbox_inches="tight")
    plt.show()


def draw_sampled_frames_cluster(
    all_labels,
    new_labels,
    name_to_nb,
    fixed_cluster,
    title="",
    k_means_name_centers=None,
    cluster=0,
    img_path=IMG_PATH,
):
    void = " "

    if str(cluster) in fixed_cluster:
        title = title + " which is fixed"
    figure(figsize=(15, 8), dpi=80)
    for video_id in name_to_nb.keys():
        frames = sorted(os.listdir(f"{img_path}{video_id}/"))
        min_ind = int(frames[0][:-4])
        max_ind = int(frames[-1][:-4])
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        if video_id in all_labels:
            for frame in all_labels[video_id]:
                frame_id = video_id + "/" + frame
                frame_cluster = k_means_name_centers[frame_id]
                color = "green"
                if video_id in new_labels and frame in new_labels[video_id]:
                    color = "red"
                if frame_cluster == cluster:
                    if (
                        str(cluster) in fixed_cluster
                        and frame_id == fixed_cluster[str(cluster)]
                    ):
                        color = "black"
                    plt.plot(
                        [video_id],
                        [int(frame) - min_ind],
                        marker="o",
                        markersize=3,
                        color=color,
                    )
                else:
                    plt.plot([video_id], [-5], marker="o", markersize=3, color="white")
    plt.xticks(rotation=70)
    plt.title(title, fontsize=20)
    plt.ylabel("frame number", fontsize=20)
    plt.show()


# generate matplotlib marker and color pair
def get_marker_color_pair(color_seed=0, include_warm=False):
    markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
        "4",
        "3",
        "2",
        "1",
        "+",
        "x",
        "X",
    ]
    if include_warm:
        colors = ["r", "orange", "b", "g", "c", "m", "y", "k"]
    else:
        colors = ["b", "g", "c", "m", "y", "k"]
    marker_color_pair = []
    for marker in markers:
        for color in colors:
            marker_color_pair.append((marker, color))

    np.random.seed(color_seed)
    np.random.shuffle(marker_color_pair)
    return marker_color_pair


def draw_all_sampled_frames_cluster(
    all_labeled,
    center_representatives,
    curr_labeled,
    k_means_name_centers=None,
    color_seed=0,
    name_to_nb={},
    assignment_done=False,
    title=None,
):
    marker_color_pair = get_marker_color_pair(color_seed=color_seed)
    void = " "
    figure(figsize=(15, 8), dpi=80)
    for video_id, frames in all_labeled.items():
        min_ind = int(frames[0])
        max_ind = int(frames[-1])
        plt.vlines([void], 0, max_ind - min_ind)
        void += " "
        for frame in frames:
            frame_id = video_id + "/" + frame
            frame_cluster = k_means_name_centers[frame_id]
            marker, color = marker_color_pair[frame_cluster]
            markersize = 6
            if not assignment_done:
                if (
                    frame_cluster in center_representatives
                    and center_representatives[frame_cluster] == frame_id
                ):
                    color = "red"
                elif video_id in curr_labeled and frame in curr_labeled[video_id]:
                    color = "orange"
            elif assignment_done:
                if (
                    frame_cluster in center_representatives
                    and center_representatives[frame_cluster] == frame_id
                ):
                    color = "orange"
                elif video_id in curr_labeled and frame in curr_labeled[video_id]:
                    color = "red"

            if marker in ["2", "3", "4"] and color in ["red", "orange"]:
                markersize = markersize * 2
            plt.plot(
                [name_to_nb[video_id]],
                [int(frame) - min_ind],
                marker=marker,
                markersize=markersize,
                color=color,
            )

    # make legend for color
    # move the legend to the right
    # remove the line in the legend figure
    for cluster in np.unique(list(k_means_name_centers.values())):
        marker, color = marker_color_pair[cluster]
        plt.plot(
            [name_to_nb[video_id]],
            [-15],
            marker=marker,
            markersize=6,
            color=color,
            label=f"cluster {cluster}",
        )

    plt.plot([0], [-15], label="----------------")
    if assignment_done:
        plt.plot(
            [0],
            [-15],
            marker="o",
            markersize=8,
            color="orange",
            label="frame to be labeled",
        )
        plt.plot(
            [0], [-15], marker="o", markersize=8, color="red", label="fixed cluster"
        )
    else:
        plt.plot(
            [0], [-15], marker="o", markersize=8, color="orange", label="labeled frame"
        )
        plt.plot(
            [0], [-15], marker="o", markersize=8, color="red", label="cluster center"
        )
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        handletextpad=1.5,
        handlelength=0,
    )

    if title is None:
        title = "All clusters"
    plt.xticks(rotation=0)
    plt.title(title, fontsize=20)
    plt.ylabel("frame index", fontsize=20)
    plt.ylim(-10)
    return marker_color_pair


def get_sampled_labels(name, round_=0, seed=0, path2="new_labeled", file_type="json"):
    for folder in sorted(os.listdir(name)):
        if (
            folder.isnumeric()
            and path2 + f"_SEED={seed}_round={round_}.{file_type}"
            in os.listdir(name + folder)
        ):  
            if file_type == "json":
                with open(
                    name + f"{folder}/{path2}_SEED={seed}_round={round_}.json", "r"
                ) as f:
                    label = json.load(f)
                return label
            elif file_type == "pickle":
                with open(
                    name + f"{folder}/{path2}_SEED={seed}_round={round_}.pickle", "rb"
                ) as f:
                    label = pickle.load(f)
                return label
            elif file_type == 'pt':
                label = torch.load(name + f"{folder}/{path2}_SEED={seed}_round={round_}.pt")
                return label
    
    print(f"Could not find {path2}_SEED={seed}_round={round_}.{file_type} in {name}")
    return None


def get_embeddings(name, seed=0, embedding_name="embeddings"):
    for folder in sorted(os.listdir(name)):
        if folder.isnumeric():
            embedding_dir = "embeddings"
            if embedding_dir not in os.listdir(name + folder):
                embedding_dir = "checkpoints"
            if embedding_dir not in os.listdir(name + folder):
                continue
            saved_embeddings = os.listdir(name + folder + "/" + embedding_dir)
            for file in saved_embeddings:
                if f"{embedding_name}_SEED={seed}.pth" == file:
                    embeddings = torch.load(
                        name
                        + f"{folder}/{embedding_dir}/{embedding_name}_SEED={seed}.pth"
                    )
                    return embeddings


def plot_curr_training_images(
    name_to_nb,
    curr_labels,
    new_labels,
    nb_col=6,
    figsize=(15, 8),
    img_path=IMG_PATH,
    color="green",
    save_name=None,
):
    all_frame_paths = []
    for video_id in name_to_nb.keys():
        curr_frames = curr_labels.get(video_id, [])
        new_frames = new_labels.get(video_id, [])
        curr_frames = [(frame, "black") for frame in curr_frames]
        new_frames = [(frame, color) for frame in new_frames]
        frames = curr_frames + new_frames
        frames = sorted(frames, key=lambda x: int(x[0]))
        frame_paths = [
            (f"{img_path}{video_id}/{FRAME_KEYWORD}{frame}{FILE_TYPE}", color)
            for frame, color in frames
        ]
        all_frame_paths += frame_paths
    curr_col = 0
    curr_video = None
    header = ""
    for frame_path, color in all_frame_paths:
        if curr_col == 0:
            fig, axs = plt.subplots(1, nb_col, figsize=figsize)
            # remmove white space around image
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
        img = Image.open(frame_path)
        img = np.array(img)

        frame_id = frame_path[len(img_path) : -len(FILE_TYPE)]
        video_id = frame_id.split("/")[0]
        if curr_video != video_id:
            curr_video = video_id
            header = ""
        else:
            header = ""
        axs[curr_col].imshow(img)
        # remove white space around image
        axs[curr_col].margins(x=0)
        axs[curr_col].axis("off")

        frames = os.listdir(f"{img_path}{video_id}/")
        frames = [f for f in frames if f != ".DS_Store"]
        frames = sorted(
            frames, key=lambda x: int(x[len(FRAME_KEYWORD) : -len(FILE_TYPE)])
        )
        frames = [f[len(FRAME_KEYWORD) : -len(FILE_TYPE)] for f in frames]
        ind = frames.index(frame_id.split("/")[1][len(FRAME_KEYWORD) :])

        axs[curr_col].set_title(
            header + name_to_nb[video_id] + f" - frame {ind}", fontsize=8, color=color
        )
        curr_col += 1

        if curr_col == nb_col:
            plt.show()
            curr_col = 0

    if curr_col > 0 and curr_col < nb_col:
        for i in range(curr_col, nb_col):
            axs[i].imshow(np.zeros_like(img))
            axs[i].axis("off")
    if save_name:
        plt.savefig(f"../../../MICCAI/{save_name}.pdf", bbox_inches="tight")


def plot_curr_training_images_savefig(
    name_to_nb,
    curr_labels,
    new_labels,
    nb_col=3,
    figsize=(15, 8),
    img_path=IMG_PATH,
    color="green",
    save_name=None,
):
    all_frame_paths = []
    for video_id in name_to_nb.keys():
        new_frames = new_labels.get(video_id, [])
        new_frames = curr_labels.get(video_id, []) + new_frames
        frame_paths = [
            f"{img_path}{video_id}/{FRAME_KEYWORD}{frame}{FILE_TYPE}"
            for frame in new_frames
        ]
        frame_paths = sorted(
            frame_paths,
            key=lambda x: int(
                x[len(img_path + CLASS_ID_TYPE + FRAME_KEYWORD) : -len(FILE_TYPE)]
            ),
        )
        all_frame_paths += frame_paths
    curr_col = 0
    header = ""

    frames_id = ["43", "62", "80", "142", "158"]
    for frame_path in all_frame_paths:
        if frame_path[len(img_path) : len(img_path + CLASS_ID_TYPE) - 1] != "15-00":
            curr_col = 0
            continue
        if curr_col == 0:
            fig, axs = plt.subplots(1, nb_col, figsize=figsize)
            plt.subplots_adjust(wspace=0.0, hspace=0.05)
        img = Image.open(frame_path)
        img = np.array(img)

        frame_id = frame_path[len(img_path) : -len(FILE_TYPE)]
        video_id = frame_id.split("/")[0]
        axs[curr_col].imshow(img)
        # remove white space around image
        axs[curr_col].margins(x=0)
        axs[curr_col].axis("off")
        use_color = color
        if curr_col == 2:
            use_color = "black"
        axs[curr_col].set_title(
            header + f"frame {frames_id[curr_col]}", fontsize=10, color=use_color
        )
        curr_col += 1
        if curr_col == nb_col:
            if save_name is not None:
                plt.savefig(f"../../../MICCAI/{save_name}.pdf", bbox_inches="tight")
            plt.show()
            return


def vec_sim(u, v):
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def plot_most_similar_images(
    new_labels,
    curr_labels,
    embeddings,
    nb_of_similar_img=4,
    use_dist=True,
    img_path=IMG_PATH,
    figsize=(15, 6),
    plot=True,
):
    if use_dist:
        dist_func = torch.dist
    else:
        dist_func = vec_sim

    avg_score = []
    for new_video_id, new_frames in new_labels.items():
        for new_frame in new_frames:
            new_frame_id = new_video_id + "/frame" + new_frame
            new_frame_embedding = embeddings[new_frame_id]

            all_scores = []
            for curr_video_id, curr_frames in curr_labels.items():
                for curr_frame in curr_frames:
                    curr_frame_id = curr_video_id + "/frame" + curr_frame
                    curr_frame_embedding = embeddings[curr_frame_id]
                    all_scores.append(
                        [
                            curr_frame_id,
                            dist_func(new_frame_embedding, curr_frame_embedding).item(),
                        ]
                    )
            all_scores = sorted(
                all_scores, key=lambda x: x[1], reverse=use_dist == False
            )
            avg_score.append(all_scores[0][1])

            if plot:
                fig, axs = plt.subplots(1, nb_of_similar_img + 1, figsize=figsize)
                new_frame = np.array(Image.open(img_path + new_frame_id + ".png"))
                axs[0].imshow(new_frame)
                axs[0].set_title(
                    f"{new_frame_id}", fontsize=8, color="green", weight="bold"
                )
                axs[0].axis("off")
                curr_col = 1

                for curr_frame_id, sim_score in all_scores:
                    frame = np.array(Image.open(img_path + curr_frame_id + ".png"))
                    axs[curr_col].imshow(frame)
                    axs[curr_col].set_title(
                        f"{curr_frame_id} {round(sim_score, 3)}", fontsize=8
                    )
                    axs[curr_col].axis("off")
                    curr_col += 1
                    if curr_col == nb_of_similar_img + 1:
                        break
                plt.show()

    return avg_score


def display_cluster_img(
    cluster,
    embeddings,
    centers,
    k_means_name_centers,
    dataset,
    curr_labeled,
    center_representatives,
    name_to_nb,
    assigment_done=False,
    nb_col=10,
    ML_entropy={},
    figsize=(15, 8),
    img_path=IMG_PATH,
    lab_path=LAB_PATH,
    dataset_category="AURIS",
    frame_keyword="frame",
    patch=False
):

    filetype = ".png"
    if dataset_category == "pets":
        filetype = ".jpg"

    header = f"C{cluster}"
    k_means_centers_name = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_name[v].append(k)

    target_cluster = k_means_centers_name[cluster]
    nb_row = len(target_cluster) // nb_col
    if nb_row != len(target_cluster) / nb_col:
        nb_row += 1

    fig, axs = plt.subplots(nb_row, nb_col, figsize=figsize)
    # reduce blank space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    col = 0
    row = 0
    curr_video = None
    bold_header = False
    for frame_id in target_cluster:
        patch_id = None
        if patch:
            frame_id, patch_id = '/'.join(frame_id.split("/")[:2]), frame_id.split("/")[-1]
        
        video_id, frame = frame_id.split("/")
        frame = frame[len(frame_keyword) :]
        if curr_video != video_id:
            curr_video = video_id
            bold_header = True
        else:
            bold_header = False
        
        if patch:
            frame_embedding = embeddings[frame_id + "/" + patch_id]
        else:
            frame_embedding = embeddings[frame_id]

        dist_to_center = torch.dist(
            frame_embedding, torch.tensor(centers[cluster])
        ).item()

        patch_color = None
        if assigment_done:
            if (
                cluster in center_representatives
                and center_representatives[cluster] == f'{frame_id}/{patch_id}'
            ):
                patch_color = "orange"
            # elif video_id in curr_labeled and frame in curr_labeled[video_id]:
            #     patch_color = "red"
        else:
            if (
                cluster in center_representatives
                and center_representatives[cluster] == frame_id
            ):
                patch_color = "red"
            elif video_id in curr_labeled and frame in curr_labeled[video_id]:
                patch_color = "orange"

        frame_path = img_path + frame_id + filetype
        mask_path = lab_path + frame_id + filetype
        title = name_to_nb[video_id] + "/f" + str(int(frame))
        
        if patch:
            dataset.return_patches = True
            imgs, lab = dataset.open_path(frame_path, mask_path, name=frame_id)
            img = imgs[int(patch_id)]
            dataset.return_patches = False
        else:
            img, lab = dataset.open_path(frame_path, mask_path)
        
        axs[row, col].imshow(img.permute((1, 2, 0)))
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        if patch_color:
            rect = patches.Rectangle(
                (0, 0),
                img.permute((1, 2, 0)).shape[1],
                img.permute((1, 2, 0)).shape[0],
                linewidth=8,
                edgecolor=patch_color,
                facecolor="none",
            )
            axs[row, col].add_patch(rect)

        metric = ML_entropy[frame_id][int(patch_id)] if patch else ML_entropy[frame_id]
        if bold_header:
            axs[row, col].set_title(
                # f"{header} - {title} - {round(dist_to_center, 3)} - {patch_id}",
                f"{header} - {title} - {round(metric, 3)} - {patch_id}",
                fontsize=6,
                weight="bold",
            )
        else:
            axs[row, col].set_title(
                # f"{header} - {title} - {round(dist_to_center, 3)} - {patch_id}", fontsize=6
                f"{header} - {title} - {round(metric, 3)} - {patch_id}", fontsize=6
            )
        col += 1
        if col == nb_col:
            col = 0
            row += 1

    if col < nb_col and row < nb_row:
        for i in range(col, nb_col):
            axs[row, i].imshow(np.zeros_like(lab))
            axs[row, i].set_xticks([])
            axs[row, i].set_yticks([])


def display_cluster_img_savefig(
    cluster,
    embeddings,
    all_labeled,
    k_means_name_centers,
    dataset,
    fixed_cluster,
    new_labels,
    center_color="red",
    target_frame=None,
    frames_to_plot=None,
    nb_col=10,
    figsize=(15, 8),
    img_path=IMG_PATH,
    lab_path=LAB_PATH,
    KMEDIAN=False,
    sphere=False,
    save_name=None,
):

    k_means_centers_frames = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_frames[v].append(k)

    if KMEDIAN:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
        else:
            for frame_id in k_means_centers_frames[cluster]:
                video_id, frame_nb = frame_id.split("/")
                if video_id in new_labels and frame_nb in new_labels[video_id]:
                    cluster_center = frame_id
                    break

        cluster_center_embeddings = embeddings[cluster_center]

    else:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
            cluster_center_embeddings = embeddings[cluster_center]

        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
            cluster_center_embeddings = embeddings[cluster_center]

        else:
            clusters_embeddings = []
            for frame_id in k_means_centers_frames[cluster]:
                clusters_embeddings.append(embeddings[frame_id])
            cluster_center_embeddings = torch.tensor(
                average_center(torch.stack(clusters_embeddings))
            )
            if sphere:
                cluster_center_embeddings = F.normalize(
                    cluster_center_embeddings, dim=0
                )

    all_images_ids = []

    col = 0
    for k, frames in all_labeled.items():
        for frame in frames:
            frame_id = k + "/" + frame
            # if int(frame) < 125:
            #     continue
            if frames_to_plot is not None and int(frame) not in frames_to_plot:
                continue
            frame_cluster = k_means_name_centers[frame_id]
            frame_embedding = embeddings[frame_id]
            dist_to_center = torch.dist(
                frame_embedding, cluster_center_embeddings
            ).item()

            color = "black"
            if (
                str(cluster) in fixed_cluster
                and fixed_cluster[str(cluster)] == frame_id
            ):
                color = "blue"
            if k in new_labels and frame in new_labels[k]:
                color = center_color

            if target_frame is not None and target_frame == int(frame):
                color = "blue"
            if frame_cluster == cluster:
                if col == 0:
                    fig, axs = plt.subplots(1, nb_col, figsize=figsize)
                    # reduce blank space between subplots
                    fig.subplots_adjust(wspace=0, hspace=0)

                all_images_ids.append(frame_id)
                frame_path = img_path + frame_id + ".jpg"
                mask_path = lab_path + frame_id + ".png"
                img, lab = dataset.open_path(frame_path, mask_path)
                axs[col].imshow(img.permute((1, 2, 0)))
                axs[col].set_xticks([])
                axs[col].set_yticks([])
                axs[col].set_title(
                    f"{frame_id[-5:]} - {round(dist_to_center, 4)}",
                    color=color,
                    fontsize=8,
                )
                col += 1
            if col == nb_col:
                if save_name is not None:
                    plt.savefig(f"../../../{save_name}.pdf", bbox_inches="tight")
                return all_images_ids


def display_video_img(
    target_video_id,
    nb_to_name,
    all_labeled,
    k_means_name_centers,
    cluster_marker_color_pair,
    dataset,
    center_representatives,
    curr_labeled,
    embeddings,
    centers,
    ML_entropy=None,
    show_mask=False,
    assigment_done=False,
    nb_col=10,
    figsize=(15, 8),
    img_path=IMG_PATH,
    lab_path=LAB_PATH,
    dataset_category="AURIS",
    frame_keyword="frame",
):

    filetype = ".png"
    if dataset_category in ["pets", "skateboard", "dog", "parrots"]:
        filetype = ".jpg"
    for video_id, frames in all_labeled.items():
        if video_id != nb_to_name[target_video_id]:
            continue
        else:
            nb_row = len(frames) // nb_col
            if nb_row != len(frames) / nb_col:
                nb_row += 1
            break

    curr_cluster = None
    # figsize = (figsize[0], nb_row + 1)
    fig, axs = plt.subplots(nb_row, nb_col, figsize=figsize)
    # reduce blank space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    col = 0
    row = 0
    for i, frame in enumerate(frames):
        frame_id = video_id + f"/{frame_keyword}" + frame
        frame_cluster = k_means_name_centers[frame_id]
        if ML_entropy is not None:
            sampling_metric = round(ML_entropy.get(frame_id, 0), 3)
        else:
            frame_embedding = embeddings[frame_id]
            dist_to_center = torch.dist(
                frame_embedding, torch.tensor(centers[frame_cluster])
            ).item()
            sampling_metric = round(dist_to_center, 3)
        ## define header
        header = f"C{frame_cluster}"
        if frame_cluster != curr_cluster:
            bold_header = True
            curr_cluster = frame_cluster
        else:
            bold_header = False

        marker, color = cluster_marker_color_pair[frame_cluster]
        # frame_embedding = embeddings[frame_id]
        # dist_to_center = torch.dist(frame_embedding, cluster_center_embeddings).item()

        patch_color = None
        if assigment_done:
            if (
                frame_cluster in center_representatives
                and center_representatives[frame_cluster] == frame_id
            ):
                patch_color = "orange"
            elif video_id in curr_labeled and frame in curr_labeled[video_id]:
                patch_color = "red"
        else:
            if (
                frame_cluster in center_representatives
                and center_representatives[frame_cluster] == frame_id
            ):
                patch_color = "red"
            elif video_id in curr_labeled and frame in curr_labeled[video_id]:
                patch_color = "orange"

        # all_images_ids.append(frame_id)
        frame_path = img_path + frame_id + filetype
        mask_path = lab_path + frame_id + filetype
        title = target_video_id + "/f" + str(int(i))
        img, lab = dataset.open_path(frame_path, mask_path)
        axs[row, col].imshow(img.permute((1, 2, 0)))
        if show_mask:
            axs[row, col].imshow(lab, alpha=0.5)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        if patch_color:
            rect = patches.Rectangle(
                (0, 0),
                img.permute((1, 2, 0)).shape[1],
                img.permute((1, 2, 0)).shape[0],
                linewidth=8,
                edgecolor=patch_color,
                facecolor="none",
            )
            axs[row, col].add_patch(rect)
        # axs[col].set_title(f'{title} - {round(dist_to_center, 4)}', color=color, fontsize=8)
        if bold_header:
            axs[row, col].set_title(
                f"{header} - {title} - {sampling_metric}",
                color=color,
                fontsize=6,
                fontweight="bold",
            )
        else:
            axs[row, col].set_title(
                f"{header} - {title} - {sampling_metric}", color=color, fontsize=6
            )
        col += 1
        if col == nb_col:
            row += 1
            col = 0

    if col < nb_col and row < nb_row:
        for i in range(col, nb_col):
            axs[row, i].imshow(np.zeros_like(lab))
            axs[row, i].set_xticks([])
            axs[row, i].set_yticks([])


def display_video_img_simple(
    target_video_id,
    nb_to_name,
    all_labeled,
    dataset,
    nb_col=10,
    figsize=(15, 8),
    img_path=IMG_PATH,
    lab_path=LAB_PATH,
):

    for video_id, frames in all_labeled.items():
        if video_id != nb_to_name[target_video_id]:
            continue
        else:
            nb_row = len(frames) // nb_col
            if nb_row != len(frames) / nb_col:
                nb_row += 1
            break

    # figsize = (figsize[0], nb_row + 1)
    fig, axs = plt.subplots(nb_row, nb_col, figsize=figsize)
    # reduce blank space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    col = 0
    row = 0
    all_frames = {}
    for i, frame in enumerate(frames):
        frame_id = video_id + "/frame" + frame
        # all_images_ids.append(frame_id)
        frame_path = img_path + frame_id + ".png"
        mask_path = lab_path + frame_id + ".png"
        title = target_video_id + "/f" + str(int(i))
        img, lab = dataset.open_path(frame_path, mask_path)
        all_frames[title] = img
        axs[row, col].imshow(img.permute((1, 2, 0)))
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        axs[row, col].set_title(f"{title}", fontsize=8)
        col += 1
        if col == nb_col:
            row += 1
            col = 0

    if col < nb_col and row < nb_row:
        for i in range(col, nb_col):
            axs[row, i].imshow(np.zeros_like(lab))
            axs[row, i].set_xticks([])
            axs[row, i].set_yticks([])

    return all_frames


def image_to_cluster_distance(
    image_id_simple,
    cluster,
    embeddings,
    k_means_name_centers,
    fixed_cluster,
    nb_to_name,
    new_labels=None,
    assignment_done=True,
    KMEDIAN=False,
    sphere=False,
):

    image_id = (
        nb_to_name[image_id_simple.split("/")[0]]
        + "/"
        + image_id_simple.split("/")[1][1:].zfill(ZFILL_NB)
    )
    k_means_centers_frames = collections.defaultdict(list)
    for k, v in k_means_name_centers.items():
        k_means_centers_frames[v].append(k)

    if KMEDIAN:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
        elif cluster in fixed_cluster:
            cluster_center = fixed_cluster[cluster]
        else:
            for frame_id in k_means_centers_frames[cluster]:
                video_id, frame_nb = frame_id.split("/")
                if video_id in new_labels and frame_nb in new_labels[video_id]:
                    cluster_center = frame_id
                    break

        cluster_center_embeddings = embeddings[cluster_center]

    else:
        if str(cluster) in fixed_cluster:
            cluster_center = fixed_cluster[str(cluster)]
            cluster_center_embeddings = embeddings[cluster_center]

        elif cluster in fixed_cluster and assignment_done:
            cluster_center = fixed_cluster[cluster]
            cluster_center_embeddings = embeddings[cluster_center]
        else:
            clusters_embeddings = []
            for frame_id in k_means_centers_frames[cluster]:
                clusters_embeddings.append(embeddings[frame_id])
            cluster_center_embeddings = torch.tensor(
                average_center(torch.stack(clusters_embeddings))
            )
            if sphere:
                cluster_center_embeddings = F.normalize(
                    cluster_center_embeddings, dim=0
                )

    frame_embedding = embeddings[image_id]
    dist_to_center = torch.dist(frame_embedding, cluster_center_embeddings).item()
    return dist_to_center


def image_to_image_distance(image_id1, image_id2, embeddings, nb_to_name):
    image_id1 = (
        nb_to_name[image_id1.split("/")[0]]
        + "/"
        + image_id1.split("/")[1][1:].zfill(ZFILL_NB)
    )
    image_id2 = (
        nb_to_name[image_id2.split("/")[0]]
        + "/"
        + image_id2.split("/")[1][1:].zfill(ZFILL_NB)
    )

    frame_embedding = embeddings[image_id1]
    frame_embedding2 = embeddings[image_id2]
    dist_to_center = torch.dist(frame_embedding, frame_embedding2).item()
    return dist_to_center


def most_similar_images_metrics(
    name, total_seed=10, total_round=20, img_path=IMG_PATH, use_dist=True, sphere=False
):
    all_seed_avg = []
    for SEED in range(total_seed):
        all_rounds_avg = []
        for round_ in range(total_round):
            new_labels = get_sampled_labels(
                name + "/", path2="new", round_=round_, seed=SEED
            )
            curr_labels = get_sampled_labels(
                name + "/", path2="curr", round_=round_, seed=SEED
            )
            embeddings = get_embeddings(name + "/", seed=SEED)
            if sphere:
                embeddings = {k: F.normalize(v, dim=0) for k, v in embeddings.items()}
            avg_score = plot_most_similar_images(
                new_labels,
                curr_labels,
                embeddings,
                img_path=img_path,
                use_dist=use_dist,
                plot=False,
            )
            all_rounds_avg.append(np.mean(avg_score))
        all_seed_avg.append(all_rounds_avg)
    all_seed_avg = np.array(all_seed_avg)
    plt.plot(np.mean(all_seed_avg, axis=0))
    plt.title("new samples to current dataset proximity")
    if use_dist:
        plt.ylabel("image embedding space euclidian distance")
    else:
        plt.ylabel("image embedding space cosine similarity")
    plt.xlabel("round")


def diff_per_round(exp_folder1, exp_folder2, nb_round=10, seed=1):
    nb_diff_per_round = []
    for round_ in range(nb_round):
        new_labeled_path = f"{exp_folder1}new_labeled_SEED={seed}_round={round_}.json"
        with open(new_labeled_path, "r") as f:
            new_labeled1 = json.load(f)

        new_labeled_path = f"{exp_folder2}new_labeled_SEED={seed}_round={round_}.json"
        with open(new_labeled_path, "r") as f:
            new_labeled2 = json.load(f)

        diff = 0
        for k, v in new_labeled1.items():
            if k not in new_labeled2:
                diff += len(v)
            else:
                for nb in v:
                    if nb not in new_labeled2[k]:
                        diff += 1
        nb_diff_per_round.append(diff)
        # print('---------')
        # print(f'round {round_}: {diff} new labeled frames')
        # print(new_labeled1)
        # print(new_labeled2)
    return nb_diff_per_round


def fit_TSNE(embeddings, centers1, perplexity=30, random_state=0):
    tsne = TSNE(
        n_components=2,
        verbose=0,
        perplexity=perplexity,
        n_iter=1000,
        n_iter_without_progress=300,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedding_keys = []
    embedding_values = []
    for k, v in embeddings.items():
        embedding_keys.append(k)
        embedding_values.append(v)

    for i, center in enumerate(centers1):
        embedding_keys.append(f"center {i}")
        embedding_values.append(center)

    tsne_results = tsne.fit_transform(np.stack(embedding_values))
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    return x, y, embedding_keys


def plot_TSNE_video(
    x, y, embedding_keys, video_marker_color_pair, name_to_nb, ylim=-25
):
    plt.figure(figsize=(16, 8))
    # generate matplotlib colors
    for i, frame_id in enumerate(embedding_keys):
        if frame_id[: len("center")] == "center":
            continue
        video_id = frame_id.split("/")[0]
        frame_nb = frame_id.split("/")[1][len("frame") :]
        marker, color = video_marker_color_pair[video_id]
        marker_size = 7
        im = plt.plot([x[i]], [y[i]], c=color, marker=marker, markersize=marker_size)

    for video_id in video_marker_color_pair.keys():
        marker, color = video_marker_color_pair[video_id]
        plt.plot(
            [0],
            [-100],
            marker=marker,
            markersize=10,
            color=color,
            label=f"{name_to_nb[video_id]}",
        )

    # increase legend box size
    plt.legend(
        bbox_to_anchor=(1, 1),
        loc="upper left",
        borderaxespad=0.5,
        handletextpad=1.5,
        handlelength=0,
        borderpad=1,
    )
    plt.ylim(ylim)
    # plt.savefig('../../../MICCAI/TSNE.pdf', bbox_inches='tight')


def plot_TSNE_cluster_first_conv(
    x,
    y,
    embedding_keys,
    k_means_name_centers,
    curr_labeled,
    color_marker_pair,
    center_representatives,
    fixed_cluster,
    ylim=-25,
    only_cluster=None,
):
    plt.figure(figsize=(16, 8))
    # generate matplotlib colors
    for i, frame_id in enumerate(embedding_keys):
        marker_size = 8
        alpha = 0.7
        zorder = 1
        if frame_id[: len("center")] == "center":
            if int(frame_id[len("center ") :]) in fixed_cluster:
                color = "red"
                # marker_size = 14
                frame_cluster = frame_id.split(" ")[1]
                frame_cluster = int(frame_cluster)
                marker, _ = color_marker_pair[frame_cluster]
                alpha = 1
                zorder = 10
            else:
                continue
        else:
            video_id = frame_id.split("/")[0]
            frame_nb = frame_id.split("/")[1][len("frame") :]
            frame_cluster = k_means_name_centers[frame_id]
            marker, color = color_marker_pair[frame_cluster]
            if video_id in curr_labeled and frame_nb in curr_labeled[video_id]:
                alpha = 1
                zorder = 10
                color = "orange"
                # marker_size = 14
            elif (
                frame_cluster not in fixed_cluster
                and center_representatives[frame_cluster] == frame_id
            ):
                zorder = 10
                color = "red"
                alpha = 1
                # marker_size = 14
                # if video_id + '/' + frame_nb == '984f0f1c36/00090':
                #     color = 'blue'
                # elif video_id + '/' + frame_nb == '07652ee4af/00075':
                #     color = 'cyan'

        if marker in ["1", "2", "3", "4"] and alpha == 1:
            marker_size = marker_size + 10

        if only_cluster and frame_cluster == only_cluster:
            im = plt.plot(
                [x[i]],
                [y[i]],
                c=color,
                marker=marker,
                markersize=marker_size,
                alpha=alpha,
                zorder=zorder,
            )
        elif not only_cluster:
            im = plt.plot(
                [x[i]],
                [y[i]],
                c=color,
                marker=marker,
                markersize=marker_size,
                alpha=alpha,
                zorder=zorder,
            )

    # make legend for color and marker
    for cluster in np.unique(list(k_means_name_centers.values())):
        marker, color = color_marker_pair[cluster]
        plt.plot(
            [0],
            [-100],
            marker=marker,
            markersize=10,
            color=color,
            label=f"cluster {cluster}",
        )
    plt.plot([0], [-100], label="----------------")
    plt.plot(
        [0], [-100], marker="o", markersize=10, color="orange", label="labeled frame"
    )
    plt.plot(
        [0], [-100], marker="o", markersize=10, color="red", label="cluster center"
    )
    plt.legend(
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
        borderaxespad=0.5,
        handletextpad=1.5,
        handlelength=0,
        borderpad=1,
    )
    plt.ylim(ylim)


def plot_TSNE_cluster_second_conv(
    x,
    y,
    embeddings,
    ML_entropy,
    all_centers,
    all_k_means_centers_name,
    fixed_cluster,
    color_marker_pair,
    iter_ind=0,
    ylim=-25,
    only_cluster=None,
):

    centers = all_centers[iter_ind]
    k_means_centers_name = all_k_means_centers_name[iter_ind]

    k_means_name_centers = {}
    for k, v in k_means_centers_name.items():
        for name in v:
            k_means_name_centers[name] = k

    center_representatives = {}
    for i, center in enumerate(centers):
        if i in fixed_cluster:
            continue

        if ML_entropy is not None:
            ent = -float("inf")
        else:
            min_ = 100000

        name = None
        for k, v in embeddings.items():
            if k not in k_means_centers_name[i]:
                continue

            if ML_entropy is not None:
                if ML_entropy[k] > ent:
                    ent = ML_entropy[k]
                    name = k
            else:
                dist = euc_distance(torch.tensor(center), v).item()
                if dist < min_:
                    min_ = dist
                    name = k

        center_representatives[i] = name

    plt.figure(figsize=(16, 8))
    # generate matplotlib colors
    for i, frame_id in enumerate(embeddings.keys()):
        frame_cluster = k_means_name_centers[frame_id]
        video_id, frame_nb = frame_id.split("/")
        frame_nb = frame_nb[len("frame") :]
        marker, color = color_marker_pair[frame_cluster]
        marker_size = 8
        zorder = 1
        alpha = 0.7

        if (
            frame_cluster in center_representatives
            and center_representatives[frame_cluster] == frame_id
        ):
            # if video_id in new_labeled and frame_nb in new_labeled[video_id]:
            color = "orange"
            zorder = 10
            alpha = 1
        elif (
            frame_cluster in fixed_cluster and fixed_cluster[frame_cluster] == frame_id
        ):
            color = "red"
            zorder = 10
            alpha = 1

        if marker in ["2", "3", "4"] and alpha == 1:
            marker_size = marker_size + 10

        if only_cluster and frame_cluster == only_cluster:
            im = plt.plot(
                [x[i]],
                [y[i]],
                c=color,
                marker=marker,
                markersize=marker_size,
                alpha=alpha,
                zorder=zorder,
            )
        elif not only_cluster:
            im = plt.plot(
                [x[i]],
                [y[i]],
                c=color,
                marker=marker,
                markersize=marker_size,
                alpha=alpha,
                zorder=zorder,
            )

    # make legend for color and marker
    for cluster in np.unique(list(k_means_name_centers.values())):
        marker, color = color_marker_pair[cluster]
        plt.plot(
            [0],
            [-100],
            marker=marker,
            markersize=10,
            color=color,
            label=f"cluster {cluster}",
        )

    plt.plot([0], [-100], label="----------------")
    plt.plot(
        [0],
        [-100],
        marker="o",
        markersize=10,
        color="orange",
        label="frame to be labeled",
    )
    plt.plot(
        [0],
        [-100],
        marker="o",
        markersize=10,
        color="red",
        label="center of fixed clusters",
    )
    plt.legend(
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
        borderaxespad=0.5,
        handletextpad=1.5,
        handlelength=0,
        borderpad=1,
    )
    plt.ylim(ylim)
    # plt.savefig('../../../MICCAI/TSNE_cluster.pdf', bbox_inches='tight')

    return centers, k_means_name_centers, center_representatives


def set_dataset_notebook(SEED, img_path, lab_path):
    set_random(SEED)

    ### SPLIT TRAIN SET ###
    if config["DATASET"] == "intuitive":
        train_data, val_data, test_data = None, None, None
        TRAIN_SEQ = []
        for n in sorted(os.listdir(img_path)):
            if "seq" in n and n != "seq_40":
                TRAIN_SEQ.append(n)

        TEST_SEQ = []
        for n in sorted(os.listdir(img_path.replace("train", "test"))):
            if "seq" in n:
                TEST_SEQ.append(n)

        TRAIN_SEQ, VAL_SEQ = CV_split(
            TRAIN_SEQ, SEED=SEED, num_folds=config["NUM_FOLDS"]
        )

        print(
            f"embedding method: {config['EMBEDDING_METHOD']}\n"
            + f"weight path: {config['PRETRAINED_WEIGHT_PATH']}\n"
            + f"TRAIN SEQ: {TRAIN_SEQ}\n"
            + f"VAL SEQ: {VAL_SEQ}\n"
            + f"TEST SEQ: {TEST_SEQ}\n"
        )
        curr_labeled = init_labeled_set(
            config, TRAIN_SEQ, img_path, frame_keyword, class_id_type
        )
    elif config["DATASET"] == "uavid":
        train_data, val_data, test_data = None, None, None
        TRAIN_SEQ = []
        for n in sorted(os.listdir(img_path)):
            if "seq" in n:
                TRAIN_SEQ.append(n)

        TEST_SEQ = []
        for n in sorted(os.listdir(img_path.replace("train", "val"))):
            if "seq" in n:
                TEST_SEQ.append(n)

        TRAIN_SEQ, VAL_SEQ = CV_split(
            TRAIN_SEQ, SEED=SEED, num_folds=config["NUM_FOLDS"]
        )

        print(
            f"embedding method: {config['EMBEDDING_METHOD']}\n"
            + f"weight path: {config['PRETRAINED_WEIGHT_PATH']}\n"
            + f"TRAIN SEQ: {TRAIN_SEQ}\n"
            + f"VAL SEQ: {VAL_SEQ}\n"
            + f"TEST SEQ: {TEST_SEQ}\n"
        )
        curr_labeled = init_labeled_set(
            config, TRAIN_SEQ, img_path, frame_keyword, class_id_type
        )
    else:
        TRAIN_SEQ, VAL_SEQ, TEST_SEQ = None, None, None
        train_data, val_data, test_data = divide_data(
            img_path,
            lab_path,
            num_train_val=num_train_val,
            data_path=data_path,
            SEED=SEED,
        )

        print(
            f"embedding method: {config['EMBEDDING_METHOD']}\n"
            + f"weight path: {config['PRETRAINED_WEIGHT_PATH']}\n"
            + f"train video: {sorted(train_data.keys())}, {len(train_data)}\n"
            + f"val video: {sorted(val_data.keys())}, {len(val_data)}\n"
            + f"test video: {sorted(test_data.keys())}, {len(test_data)}\n"
        )
        curr_labeled = init_labeled_set(
            config, train_data.items(), img_path, frame_keyword, class_id_type
        )

    set_random(SEED)
    train_data, val_data, test_data = divide_data_split_auris(img_path, lab_path)

    train_val_keys = list(train_data.keys()) + list(val_data.keys())
    print(
        f"number of train video: {sorted(train_data.keys())}\n"
        + f"number of val video: {sorted(val_data.keys())}\n"
        + f"number of test video: {sorted(test_data.keys())}\n"
    )

    all_labeled = collections.defaultdict(list)
    ind_keyword = len(lab_path + CLASS_ID_TYPE)
    for class_ID, data_paths in train_data.items():
        inds = np.arange(len(data_paths))
        L = []
        for ind in inds:
            mask_path = data_paths[ind][1]
            number = mask_path[ind_keyword + len(FRAME_KEYWORD) : -len(FILE_TYPE)]
            L.append(number)
        all_labeled[class_ID] = L

    test_imgTrans = T.Compose(
        [
            T.Resize(config["IMG_SIZE"], interpolation=Inter.BILINEAR),
        ]
    )
    test_labelTrans = T.Compose(
        [T.Resize(config["IMG_SIZE"], interpolation=Inter.NEAREST)]
    )

    all_train_dataset = DataHandler(
        data_path=train_data,
        img_trans=test_imgTrans,
        label_trans=test_labelTrans,
        lab_path=lab_path,
    )

    test_dataset = DataHandler(
        data_path=test_data,
        img_trans=test_imgTrans,
        label_trans=test_labelTrans,
        lab_path=lab_path,
    )
    video_dataset = VideoDataHandler(data_path=train_data)
    OF_dataset = OFDataHandler(data_path=train_data, label_path=lab_path)

    return test_dataset, all_labeled, all_train_dataset, train_val_keys, OF_dataset


def set_parrot_dataset(SEED, img_path, lab_path, data_path):
    set_random(SEED)
    train_data, val_data, test_data = divide_data_split_youtube(
        img_path, lab_path, SEED=SEED, data_path=data_path
    )

    train_val_keys = list(train_data.keys()) + list(val_data.keys())
    print(
        f"number of train video: {sorted(train_data.keys())}\n"
        + f"number of val video: {sorted(val_data.keys())}\n"
        + f"number of test video: {sorted(test_data.keys())}\n"
    )

    all_labeled = collections.defaultdict(list)
    ind_keyword = len(lab_path + CLASS_ID_TYPE_YOUTUBE)
    for i, (class_ID, data_paths) in enumerate(train_data.items()):
        inds = np.arange(len(data_paths))
        L = []
        for ind in inds:
            mask_path = data_paths[ind][1]
            number = mask_path[ind_keyword + len(FRAME_KEYWORD_YOUTUBE) :]
            L.append(number)
        all_labeled[class_ID] = L

    test_imgTrans = T.Compose(
        [
            T.Resize(
                (img_size_youtube["IMG_SIZE"], img_size_youtube["IMG_SIZE2"]),
                interpolation=Inter.BILINEAR,
            ),
        ]
    )
    test_labelTrans = T.Compose(
        [
            T.Resize(
                (img_size_youtube["IMG_SIZE"], img_size_youtube["IMG_SIZE2"]),
                interpolation=Inter.NEAREST,
            )
        ]
    )

    all_train_dataset = DataHandlerYoutube(
        data_path=train_data,
        img_trans=test_imgTrans,
        label_trans=test_labelTrans,
        img_path=img_path,
        path=data_path,
    )

    test_dataset = DataHandlerYoutube(
        data_path=test_data,
        img_trans=test_imgTrans,
        label_trans=test_labelTrans,
        img_path=img_path,
        path=data_path,
    )

    return test_dataset, all_labeled, all_train_dataset, train_val_keys

def mask_to_rgb(mask):
    # Define the color mapping
    color_map = {
        0: [0, 0, 0],         # Black
        1: [128, 0, 0],       # Maroon
        2: [0, 128, 0],       # Dark Green
        3: [128, 128, 0],     # Olive
        4: [0, 0, 128],       # Navy
        5: [128, 0, 128],     # Purple
        6: [0, 128, 128],     # Teal
        7: [192, 192, 192],   # Silver
        8: [255, 0, 0],       # Red
        9: [0, 255, 0],       # Green
        10: [0, 0, 255],      # Blue
        11: [255, 255, 0],    # Yellow
        12: [255, 0, 255],    # Fuchsia
        13: [0, 255, 255],    # Aqua
        14: [128, 128, 128],  # Gray
        15: [128, 0, 255],    # Violet
        16: [255, 128, 0],    # Orange
        17: [0, 128, 255],    # Sky Blue
        18: [128, 255, 0],    # Lime
        19: [255, 128, 128],  # Pink
        20: [128, 255, 128],  # Light Green
        255: [255, 255, 255]  # White
    }

    # Create an empty RGB image with the same dimensions as the mask
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Populate the RGB image with colors based on the mask values
    for key, value in color_map.items():
        rgb_image[mask == key] = value

    return rgb_image

def convert_label(label, auc=None, max_length=20):
    label = label.split('/')[-1]
    if label[0] == '_':
        label = label[1:]
    if label[:3] == 'AL_':
        label = label[3:]
    # if label == 'COAL_entropy_simCLR':
    #     label = 'COAL (ours)'#            | 0.7271'
    if label == 'COWAL_entropy_simCLR_patch':
        label = 'COWAL patches (ours)'#            | 0.828'
    if label == 'kmeans_fulldataset_entropy_simCLR_noHung':
        label = 'COAL'#            | ' + ('0.7271' if VERSION == 'v4' else '0.8194')
    if label == 'COWAL_entropy_simCLR':
        label = 'COWAL (ours)'#            | 0.828'
    if label == 'COWAL_center_simCLR':
        label = 'COWAL center (ours)'#            | 0.828'
    if label == 'COAL_entropy_model':
        label = 'COWAL with TME (ours)' #           | 0.7271'
    if label == 'kmeans_fulldataset_entropy_simCLR_hung':
        label = 'COAL with Hungarian Matching'#       | 0.7270'
    if label == 'kmeans_fulldataset_entropy_resnet':
        label = 'ImageNet Embedding'#      | 0.7255'
    if label == 'kmeans_fulldataset_entropy_resnet_lastLayer':
        label = 'ImageNet Embedding Last layer'#      | 0.7255'
    if label == 'kmeans_fulldataset_entropy_model':
        label = 'AL model Embedding'#       | 0.7213'
    if label == 'kmeans_entropy_simCLR':
        label = 'K-means, K=15                | 0.7226'
    if label == 'MC_Dropout':
        label = 'MC Dropout'#                | 0.824'
    if label == 'BALD':
        label = 'BALD'#                          | 0.822'
    if label == 'density_entropy':
        label = 'Density Entropy'#         | ' + ('0.7262' if VERSION == 'v4' else '0.8189')
    if label == 'density':
        label = 'Temporal Coverage'#    | ' + ('0.7256' if VERSION == 'v4' else '0.826')
    if label == 'entropy':
        label = 'Entropy'#                      | ' + ('0.7195' if VERSION == 'v4' else '0.820')
    if label == 'random':
        label = 'Random'#                     | ' + ('0.7168' if VERSION == 'v4' else '0.825')
    if label == 'coreset':
        label = 'Coreset'#                      | ' + ('0.7181' if VERSION == 'v4' else '0.825')
    if label == 'suggestive_ann':
        label = 'SA'#                              | ' + ('0.7217' if VERSION == 'v4' else '0.825')
    if label == 'VAAL':
        label = 'VAAL'#                          | ' + ('0.7217' if VERSION == 'v4' else '0.821')
    if label == 'COAL_entropy_model':
        label = 'COAL with AL model embedding'#                | ' + ('0.7217' if VERSION == 'v4' else '0.8138')
    if label == 'kmeans_fulldataset_entropy_simCLR':
        label = 'COAL with Hungarian Matching'#                | ' + ('0.7217' if VERSION == 'v4' else '0.8138')
    if label == 'coreset_model':
        label = 'Coreset with TME'#                | ' + ('0.7217' if VERSION == 'v4' else '0.8138')
    if label == 'suggestive_ann_model':
        label = 'SA with TME'#
    if label == 'coreset_entropy':
        label = 'Coreset Entropy'#
    if label == 'classEntropy_DO':
        label = 'OREAL (ours)'
    if label == 'classEntropy_DO_Mean':
        label = 'OREAL Mean'
    if label == 'classEntropy_DO_ML':
        label = 'OREAL Prob x Ent'
    if label == 'classEntropy_DO_start=0':
        label = 'OREAL delta'
    if label == 'CBAL_DO':
        label = 'CBAL'
    if label == 'CBAL_DO_start=1':
        label = 'CBAL δ' # 'CBAL Δ'
    if label == 'random_DO':
        label = 'random'
    if label == 'BvSB_DO':
        label = 'BvSB'
    if label == 'revisiting_DO':
        label = 'revisiting SP'
    if label == 'pixelBal_DO':
        label = 'Pixel Bal'
    if label == 'revisiting_MUL-1':
        label = 'revisiting - stage 1'
    if label == 'revisiting_MUL-2':
        label = 'revisiting - stage 2'
    if label == 'pixelBal_MUL-1':
        label = 'Pixel Bal - stage 1'
    if label == 'pixelBal_MUL-2':
        label = 'Pixel Bal - stage 2'
    if label == 'classEntropy_MUL-1':
        label = 'OREAL - stage 1'
    if label == 'classEntropy_MUL-2':
        label = 'OREAL - stage 2'
    if label == 'random_MUL-1':
        label = 'random - stage 1'
    if label == 'random_MUL-2':
        label = 'random - stage 2'

    if auc is not None and isinstance(auc, dict):
        # max_length = 0
        # for k, v in auc.items():
        #     if len(k) > max_length:
        #         max_length = len(k)

        score = auc[label]
        label = label.ljust(max_length, ' ')
        label += '  | ' + f"{score:.3f}"
        

    return label

# calculate area under the curve
def calculate_auc(scores, all_AL, full_dataset_score):
    dict_scores = {}
    for i, score in enumerate(scores):
        label = all_AL[i][6:-1]
        name = convert_label(label)
        dict_scores[name] = np.array(score)
    
    for k, score in dict_scores.items():
        for j, s in enumerate(score):
            score[j] = s / (full_dataset_score or 1)
        dict_scores[k] = score

    auc = {}
    max_length = 0
    for k, score in dict_scores.items():
        auc[k] = np.trapz(score, dx=1/len(score))
        if len(k) > max_length:
            max_length = len(k)
    return auc, dict_scores, max_length