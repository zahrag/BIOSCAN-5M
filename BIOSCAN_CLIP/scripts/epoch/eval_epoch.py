import os.path

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import io
from PIL import Image
import faiss
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

def load_tensor_from_hdf5_group(group, key):
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.CenterCrop(size=224)])
    curr_image_input = np.asarray(group[key])
    curr_image_input = np.array(Image.open(io.BytesIO(curr_image_input)))
    curr_image_input = transform(curr_image_input)
    return curr_image_input

def convert_label_dict_to_list_of_dict(label_batch):
    order = label_batch['order']

    family = label_batch['family']
    genus = label_batch['genus']
    species = label_batch['species']

    list_of_dict = [
        {'order': o, 'family': f, 'genus': g, 'species': s}
        for o, f, g, s in zip(order, family, genus, species)
    ]

    return list_of_dict

def get_keys_dna_feature_and_label(all_keys_dataloader, model, device, multi_gpu=False):
    # TODO, in the future, replace get_keys_dna_feature_and_label with get_key_feature
    keys_encoded_feature_list = []
    keys_label_list = []
    pbar = tqdm(enumerate(all_keys_dataloader), total=len(all_keys_dataloader))
    for step, batch in pbar:

        if len(batch) == 7:
            _, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        else:
            _, image_input_batch, dna_batch, label_batch = batch

        pbar.set_description("Getting keys dna feature")

        dna_batch = dna_batch.to(device)

        if multi_gpu:
            encoded_dna_feature_batch = model.module.dna_encoder(dna_batch)
        else:
            if model.dna_encoder is None:
                encoded_dna_feature_batch = F.normalize(dna_batch, dim=-1)
            else:
                encoded_dna_feature_batch = model.dna_encoder(dna_batch)

        keys_encoded_feature_list = keys_encoded_feature_list + encoded_dna_feature_batch.cpu().tolist()
        keys_label_list = keys_label_list + convert_label_dict_to_list_of_dict(label_batch)

    keys_encoded_dna_feature = np.array(keys_encoded_feature_list)
    return keys_encoded_dna_feature, keys_label_list

def show_confusion_metrix(ground_truth_labels, predicted_labels, path_to_save=None, labels=None, normalize=True):
    plt.figure(figsize=(12, 12))
    if labels is None:
        labels = list(set(ground_truth_labels))
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels, labels=labels)
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False,xticklabels=labels,
                yticklabels=labels)
    plt.xticks(rotation=30)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")

    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()

def get_micro_and_macro_acc_in_four_taxonomic_levels(pred_dict, gt_dict, total_number_of_record, k_list=None):
    if k_list is None:
        k_list = [1, 3, 5]
    four_levels = ['order', 'family', 'genus', 'species']
    micro_acc_dict = {}
    macro_acc_dict = {}
    for k in k_list:
        micro_acc_dict[k] = {}
        macro_acc_dict[k] = {}
        pred_list = pred_dict[k]
        gt_list = gt_dict[k]
        micro_count_dict = {}
        macro_count_dict = {}

        for pred, gt in zip(pred_list, gt_list):
            for level in four_levels:
                k_pred = [pred[i][level] for i in range(len(pred))]
                curr_gt = gt[level]

                if level not in micro_count_dict.keys():
                    micro_count_dict[level] = 0
                if level not in macro_count_dict.keys():
                    macro_count_dict[level] = {}

                if curr_gt not in macro_count_dict[level].keys():
                    macro_count_dict[level][curr_gt] = {'correct': 0, 'total': 0}
                macro_count_dict[level][curr_gt]['total'] = macro_count_dict[level][curr_gt]['total'] + 1
                if curr_gt in k_pred:
                    micro_count_dict[level] = micro_count_dict[level] + 1
                    macro_count_dict[level][curr_gt]['correct'] = macro_count_dict[level][curr_gt]['correct'] + 1

        for level in four_levels:
            macro_acc_list = []
            for curr_gt in list(macro_count_dict[level].keys()):
                macro_acc_list.append(macro_count_dict[level][curr_gt]['correct'] * 1.0 / macro_count_dict[level][curr_gt]['total'])
            macro_acc_dict[k][level] = sum(macro_acc_list) * 1.0 / len(macro_acc_list)
            micro_acc_dict[k][level] = micro_count_dict[level] * 1.0 / total_number_of_record
    return micro_acc_dict, macro_acc_dict

def remove_specific_species(keys_encoded_feature, keys_label, species_to_drop):
    idxs_to_keep = []
    labels_after_drop = []
    for idx, label in enumerate(keys_label):
        try:
            if label['species'] not in species_to_drop:
                idxs_to_keep.append(idx)
                labels_after_drop.append(label)
        except:
            print(label)
            print(species_to_drop)
            exit()
    keys_encoded_feature_after_drop = keys_encoded_feature[idxs_to_keep]

    return keys_encoded_feature_after_drop, labels_after_drop


def eval_epoch(model, val_dataloader, keys_encoded_feature, keys_label, dataset_name, device, save_inference=False, multi_gpu=False, k_list=None, return_image_and_dna_feature=False, species_to_drop=None):
    if species_to_drop is not None:
        keys_encoded_feature, keys_label = remove_specific_species(keys_encoded_feature, keys_label, species_to_drop)

    if k_list is None:
        k_list = [1, 3, 5]

    with torch.no_grad():
        total = 0
        index = faiss.IndexFlatIP(keys_encoded_feature.shape[-1])
        index.add(keys_encoded_feature)

        pred_dict = {}
        gt_dict = {}
        pbar = tqdm(val_dataloader, total=len(val_dataloader))

        if save_inference:
            all_image_feature = []

        if return_image_and_dna_feature:
            all_image_feature = []
            all_dna_feature = []

        for batch in pbar:

            image_file_name, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch

            pbar.set_description("Eval on " + dataset_name + ":...")

            # For image
            image_input_batch = image_input_batch.to(device)
            if multi_gpu:
                curr_image_output_batch = model.module.image_encoder(image_input_batch)
            else:
                curr_image_output_batch = model.image_encoder(image_input_batch)

            if save_inference:
                all_image_feature.append(curr_image_output_batch)

            # For DNA
            if return_image_and_dna_feature:
                dna_batch = dna_batch.to(device)

                if multi_gpu:
                    encoded_dna_feature_batch = model.module.dna_encoder(dna_batch)
                else:
                    if model.dna_encoder is None:
                        encoded_dna_feature_batch = F.normalize(dna_batch, dim=-1)
                    else:
                        encoded_dna_feature_batch = model.dna_encoder(dna_batch)
                all_dna_feature = all_dna_feature + encoded_dna_feature_batch.cpu().tolist()
            list_of_label_dict = convert_label_dict_to_list_of_dict(label_batch)

            if species_to_drop is not None:
                remove_specific_species(curr_image_output_batch, list_of_label_dict, species_to_drop)

            for k in k_list:
                if k not in pred_dict.keys():
                    pred_dict[k] = []
                    gt_dict[k] = []
                distances, indices = index.search(curr_image_output_batch.cpu().numpy(), k)
                for idx in range(len(indices)):
                    total = total + 1
                    k_pred = [keys_label[i] for i in np.array(indices[idx])]
                    curr_gt_label = list_of_label_dict[idx]
                    pred_dict[k].append(k_pred)
                    gt_dict[k].append(curr_gt_label)
        total_number_of_record = len(gt_dict[k_list[0]])

        micro_acc_dict, macro_acc_dict = get_micro_and_macro_acc_in_four_taxonomic_levels(pred_dict, gt_dict, total_number_of_record, k_list=k_list)

        if return_image_and_dna_feature:
            np_all_image_feature = [tensor.cpu().numpy() for tensor in all_image_feature]
            np_all_image_feature = np.concatenate(np_all_image_feature, axis=0)
            np_all_dna_feature = np.array(all_dna_feature)

            return pred_dict, gt_dict, micro_acc_dict, macro_acc_dict, np_all_image_feature, np_all_dna_feature

        if not save_inference:
            return micro_acc_dict, macro_acc_dict
        else:
            np_all_image_feature = [tensor.cpu().numpy() for tensor in all_image_feature]
            np_all_image_feature = np.concatenate(np_all_image_feature, axis=0)
            return pred_dict, gt_dict, micro_acc_dict, macro_acc_dict, np_all_image_feature





