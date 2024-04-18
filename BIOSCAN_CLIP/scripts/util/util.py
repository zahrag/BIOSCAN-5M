import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import json
import faiss


class Table:
    def __init__(self, headers, data):
        self.headers = headers
        self.data = data
        self.column_widths = [max(len(str(item)) for item in column) for column in zip(headers, *data)]

    def print_table(self):
        self.print_row(self.headers)
        self.print_separator()
        for row in self.data:
            self.print_row(row)

    def print_row(self, row):
        formatted_row = "|".join(f"{str(item):^{width}}" for item, width in zip(row, self.column_widths))
        print(f"|{formatted_row}|")

    def print_separator(self):
        separator = "+".join("-" * (width + 2) for width in self.column_widths)
        print(f"+{separator}+")


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))


class KmerTokenizer(object):
    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i : i + self.k]
            tokens.append(k_mer)
        return tokens


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def load_bert_model(bert_model, path_to_ckpt):
    state_dict = torch.load(path_to_ckpt, map_location=torch.device("cpu"))
    state_dict = remove_extra_pre_fix(state_dict)
    bert_model.load_state_dict(state_dict)


def print_result(
    args,
    seen_micro_acc_dict,
    seen_macro_acc_dict,
    unseen_micro_acc_dict,
    unseen_macro_acc_dict,
    k_list,
    dict_for_wandb=None,
):
    if dict_for_wandb is None:
        dict_for_wandb = {}
    overall_acc = []
    for k in k_list:
        print(f"When k is {k}:")
        print("\tFor seen split: ")
        for level in list(seen_micro_acc_dict[k_list[0]].keys()):
            print(f"\t\tFor {level} level:")
            print(f"\t\t\tMicro acc:\t {seen_micro_acc_dict[k][level]}")
            print(f"\t\t\tMacro acc:\t {seen_macro_acc_dict[k][level]}")
            overall_acc.append(seen_micro_acc_dict[k][level])
            overall_acc.append(seen_macro_acc_dict[k][level])
            if dict_for_wandb is not None:
                dict_for_wandb[f"micro acc of seen split in {level} level with k={k}"] = seen_micro_acc_dict[k][level]
                dict_for_wandb[f"macro acc of seen split in {level} level with k={k}"] = seen_macro_acc_dict[k][level]
        print("\tFor unseen split: ")
        for level in list(unseen_micro_acc_dict[k_list[0]].keys()):
            print(f"\t\tFor {level} level:")
            print(f"\t\t\tMicro acc:\t {unseen_micro_acc_dict[k][level]}")
            print(f"\t\t\tMacro acc:\t {unseen_macro_acc_dict[k][level]}")
            overall_acc.append(unseen_micro_acc_dict[k][level])
            overall_acc.append(unseen_macro_acc_dict[k][level])
            if dict_for_wandb is not None:
                dict_for_wandb[f"macro acc of seen split in {level} level with k={k}"] = unseen_micro_acc_dict[k][level]
                dict_for_wandb[f"macro acc of unseen split in {level} level with k={k}"] = unseen_macro_acc_dict[k][
                    level
                ]
    overall_acc = sum(overall_acc) * 1.0 / len(overall_acc)
    print("Naive overall acc: " + str(overall_acc))

    print("For copy to google sheet")

    for k in k_list:
        curr_row = ""
        for level in list(seen_micro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{seen_micro_acc_dict[k][level]:.4f}\t"
        for level in list(unseen_micro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{unseen_micro_acc_dict[k][level]:.4f}\t"
        print(curr_row)
    for k in k_list:
        curr_row = ""
        for level in list(seen_macro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{seen_macro_acc_dict[k][level]:.4f}\t"
        for level in list(unseen_macro_acc_dict[k_list[0]].keys()):
            curr_row = curr_row + f"{unseen_macro_acc_dict[k][level]:.4f}\t"
        print(curr_row)

    return overall_acc, dict_for_wandb


def tokenize_dna_sequence(curr_dna_input, sequence_pipeline):
    curr_dna_input = tokenize_dna_sequence(sequence_pipeline, np.array(curr_dna_input))
    return curr_dna_input


def load_small_species(args):
    small_species_list = None
    if hasattr(args.bioscan_data, "path_to_small_species_list_json"):
        with open(args.bioscan_data.path_to_small_species_list_json, "r") as json_file:
            small_species_list = json.load(json_file)
    return small_species_list


def find_k_closest_records(
    input_file_name_list, input_feature_np_array, keys_file_name_list, keys_feature_np_array, k=5
):
    result_dict = {}
    index = faiss.IndexFlatIP(keys_feature_np_array.shape[-1])
    index.add(keys_feature_np_array)
    distances, indices = index.search(input_feature_np_array, k)
    for record_idx, input_file_name in enumerate(input_file_name_list):
        k_closest_records_file_name = [keys_file_name_list[key_idx] for key_idx in indices[record_idx]]
        result_dict[input_file_name_list[record_idx]] = k_closest_records_file_name
    return result_dict


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Create a colormap with a certain number of shades of colours.

    Based on https://stackoverflow.com/a/47232942/1960959

    Parameters
    ----------
    nc : int
        Number of categories.
    nsc : int
        Number of shades per category.
    cmap : str, default=tab10
        Original colormap to extend into multiple shades.
    continuous : bool, default=False
        Whether ``cmap`` is continous. Otherwise it is treated
        as categorical with adjacent colors unrelated.

    Returns
    -------
    matplotlib.colors.ListedColormap
        New cmap which alternates between ``nsc`` shades of ``nc``
        colors from ``cmap``.
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
