import io
import json
import os
import random
from collections import Counter, defaultdict
import h5py
from sklearn.preprocessing import normalize
import faiss
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import torch
from omegaconf import DictConfig
from sklearn.metrics import silhouette_samples
from umap import UMAP
from PIL import Image

from epoch.inference_epoch import get_feature_and_label
from model.simple_clip import load_clip_model
from util.dataset import load_bioscan_dataloader, load_bioscan_dataloader_with_train_seen_and_separate_keys, load_bioscan_6M_dataloader, load_bioscan_6M_dataloader_with_train_seen_and_separate_keys, load_bioscan_dataloader_for_test, load_bioscan_dataloader_for_test_6m
from util.util import Table, categorical_cmap
from util.util import check_if_using_6m_data

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"
All_TYPE_OF_FEATURES_OF_QUERY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
]
All_TYPE_OF_FEATURES_OF_KEY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
    "all_key_features",
]
LEVELS = ["order", "family", "genus", "species"]


def get_all_unique_species_from_dataloader(dataloader):
    all_species = []

    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch["species"]
    all_species = list(set(all_species))
    return all_species


def save_prediction(pred_dict, gt_dict, json_path):
    data = {"gt_labels": gt_dict, "pred_labels": pred_dict}

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)


def load_from_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    pred_list = data["pred_labels"]
    gt_list = data["gt_labels"]
    correct_predictions = sum(1 for true, predicted in zip(gt_list, pred_list) if true == predicted)
    total_samples = len(gt_list)
    eval_bioscan_1m_acc = correct_predictions / total_samples
    return pred_list, gt_list, eval_bioscan_1m_acc


def show_distribution(list):
    counts = Counter(list)

    # Get values and corresponding counts, sorted by count in descending order
    sorted_values, sorted_occurrences = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # Create bar plot with log-scaled y-axis and raw counts
    plt.bar(sorted_values, sorted_occurrences)
    plt.yscale("log")  # Set y-axis to a logarithmic scale

    # Add labels and title
    plt.title("Distribution of BioScan-1M validation data")

    plt.xticks(rotation=30)
    # Display the raw count on top of each bar
    for value, occurrence in zip(sorted_values, sorted_occurrences):
        plt.text(value, occurrence, f"{occurrence}", ha="center", va="bottom")

    # Show the plot
    plt.show()


def get_labels(my_list):
    counts = Counter(my_list)

    # Sort values by count in descending order
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    list_of_labels = []
    for i in sorted_counts:
        list_of_labels.append(i[0])

    return list_of_labels


def generate_embedding_plot(args, image_features, dna_features, language_features, gt_labels, num_classifications=10):
    def get_language_feature_mapping(language_features):
        return np.unique(language_features, axis=0, return_index=True, return_inverse=True)

    levels = ["order", "family", "genus", "species"]

    unique_lang_features, lang_indices, inv_indices = get_language_feature_mapping(language_features)
    # compute 2D embeddings
    umap_2d = UMAP(n_components=2, init="random", random_state=0, min_dist=0.5, metric="cosine")
    proj_2d = umap_2d.fit_transform(np.concatenate((image_features, dna_features, unique_lang_features), axis=0))

    all_indices = []
    for level_idx, level in enumerate(levels):
        # apply filter to points
        if level_idx > 0 and levels[level_idx - 1] in args.inference_and_eval_setting.embeddings_filters:
            prev_level = levels[level_idx - 1]
            indices = [
                i
                for i in range(len(gt_labels))
                if gt_labels[i][prev_level] == args.inference_and_eval_setting.embeddings_filters[prev_level]
            ]
            print(np.unique([gt_labels[i][prev_level] for i in indices]))
        else:
            indices = [i for i in range(image_features.shape[0])]

        all_indices.append(indices)
        # filter small classes
        taxon_counts = dict(zip(*np.unique([gt_labels[i][level] for i in indices], return_counts=True)))
        taxons = sorted(taxon_counts.keys(), key=lambda x: taxon_counts[x], reverse=True)[:num_classifications]
        indices = [i for i in indices if gt_labels[i][level] in taxons]
        random.shuffle(indices)

        number_of_sample = len(indices)
        full_indices = [
            *indices,
            *[i + number_of_sample for i in indices],
            *(np.unique(inv_indices[indices]) + 2 * number_of_sample).tolist(),
        ]  # for DNA and language features
        proj_2d_selected = proj_2d[full_indices]

        gt_list = [gt_labels[i][level] for i in indices]
        unique_values, unique_counts = np.unique(gt_list, return_counts=True)
        idx_sorted = np.argsort(unique_counts)[::-1]
        level_order = unique_values[idx_sorted]
        level_order = [f"{level_name}-{type_}" for level_name in level_order for type_ in ["image", "dna", "text"]]
        count_unique = len(unique_values)
        gt_list = [
            *[f"{gt}-image" for gt in gt_list],
            *[f"{gt}-dna" for gt in gt_list],
            *[f"{gt_labels[i][level]}-text" for i in lang_indices[np.unique(inv_indices[indices])]],
        ]

        # colors = [matplotlib.colors.to_rgb(col) for col in px.colors.qualitative.Dark24][:count_unique]
        colors = [matplotlib.colors.to_hex(col) for col in categorical_cmap(count_unique, 3).colors]

        fig_2d = px.scatter(
            proj_2d_selected,
            x=0,
            y=1,
            color=gt_list,
            opacity=1.0,
            labels={"color": level},
            color_discrete_sequence=colors,
            size_max=1,
            # title=f"Embedding plot for image and DNA features with {level} labels",
            category_orders={"color": level_order},
        )

        # fix legend to not render symbol
        region_lst = set()
        for trace in fig_2d["data"]:
            trace["name"] = trace["name"].split(",")[0]

            if trace["name"] not in region_lst:
                trace["showlegend"] = True
                region_lst.add(trace["name"])
            else:
                trace["showlegend"] = False

        fig_2d.update_layout(
            {
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "legend_title": level,
                "yaxis": {"visible": False},
                "xaxis": {"visible": False},
                "margin": dict(
                    l=5,  # left
                    r=5,  # right
                    t=5,  # top
                    b=5,  # bottom
                ),
                "activeselection_opacity": 1.0,
            }
        )

        folder_path = os.path.join(
            "/local-scratch2/projects/BioScan-CLIP", f"{PLOT_FOLDER}/{args.model_config.model_output_name}"
        )
        os.makedirs(folder_path, exist_ok=True)
        # fig_3d.update_traces(marker_size=5)
        fig_2d.write_html(os.path.join(folder_path, f"{level}_2d.html"))
        plotly.io.write_image(
            fig_2d, os.path.join(folder_path, f"{level}_2d.pdf"), format="pdf", height=600, width=800
        )
        # fig_3d.write_html(os.path.join(folder_path, f'{level}_3d.html'))
        fig_2d.show()
        # fig_3d.show()


def retrieve_images(
    args,
    name,
    query_dict,
    keys_dict,
    queries,
    keys,
    query_data,
    key_data,
    num_queries=5,
    max_k=5,
    taxon="order",
    seed=None,
):
    """
    for X in {image, DNA}:
        for _ in range(num_queries):
            1 random input image as the query (per taxon)
            {num_retrieved} retrieved images as the key using the closest X embedding
    """

    def load_image_from_h5(data, idx):
        enc_length = data["image_mask"][idx]
        image_enc_padded = data["image"][idx].astype(np.uint8)
        image_enc = image_enc_padded[:enc_length]
        image = Image.open(io.BytesIO(image_enc))
        return image.resize((256, 256))

    folder_path = os.path.join(
        "/local-scratch2/projects/BioScan-CLIP",
        RETRIEVAL_FOLDER,
        args.model_config.model_output_name,
        name,
    )
    os.makedirs(folder_path, exist_ok=True)

    # select queries
    query_indices_by_taxon = defaultdict(list)
    for i, label in enumerate(query_dict["label_list"]):
        query_indices_by_taxon[label[taxon]].append(i)
    taxon_to_sample = np.random.choice(
        list(query_indices_by_taxon.keys()), size=num_queries, replace=len(query_indices_by_taxon) < num_queries
    )
    query_indices = [
        np.random.choice(query_indices_by_taxon[taxon], size=1, replace=False)[0] for taxon in taxon_to_sample
    ]

    # retrieve with image keys
    keys_label = keys_dict["label_list"]

    for query_feature_type in queries:
        for key_feature_type in keys:
            retrieval_results = []
            queries_feature = query_dict[query_feature_type]
            keys_feature = keys_dict[key_feature_type]

            # select random queries
            queries_feature = queries_feature[query_indices, :]
            for query_index in query_indices:
                retrieval_results.append(
                    {
                        "query": {
                            "file_name": query_dict["file_name_list"][query_index],
                            "taxonomy": query_dict["label_list"][query_index],
                        },
                    }
                )

            if keys_feature is None or queries_feature is None or keys_feature.shape[-1] != queries_feature.shape[-1]:
                continue

            _, indices_per_query = make_prediction(
                queries_feature, keys_feature, keys_label, with_indices=True, max_k=max_k
            )

            for idx, indices_per_query in enumerate(indices_per_query):
                retrieval_results[idx]["predictions"] = []
                for retrieved_index in indices_per_query:
                    retrieval_results[idx]["predictions"].append(
                        {
                            "file_name": keys_dict["file_name_list"][retrieved_index],
                            "taxonomy": keys_dict["label_list"][retrieved_index],
                        }
                    )

            # save out predictions
            os.makedirs(os.path.join(folder_path, f"query_{query_feature_type}_key_{key_feature_type}"), exist_ok=True)
            with open(
                os.path.join(
                    folder_path, f"query_{query_feature_type}_key_{key_feature_type}", f"retrieved_images.json"
                ),
                "w",
            ) as json_file:
                json.dump(retrieval_results, json_file, indent=4)

            # save out images
            width_ratios = [1, 0.1, *[1 for _ in range(max_k)]]
            fig, axes = plt.subplots(
                nrows=num_queries,
                ncols=max_k + 2,
                figsize=(22, 16),
                gridspec_kw={"width_ratios": width_ratios, "hspace": 0.05, "wspace": 0.05},
            )
            query_image_file_map = {filename.decode("utf-8"): j for j, filename in enumerate(query_data["image_file"])}
            key_image_file_map = {filename.decode("utf-8"): j for j, filename in enumerate(key_data["image_file"])}
            for i, pred_dict in enumerate(retrietest_results):
                # save query
                query_file_name = pred_dict["query"]["file_name"]
                image_idx = query_image_file_map[query_file_name]
                image = load_image_from_h5(query_data, image_idx)
                axes[i, 0].imshow(image)
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
                axes[i, 0].set_ylabel(
                    "\n".join(pred_dict["query"]["taxonomy"]["species"].split()),
                    rotation="horizontal",
                    ha="right",
                    fontsize=20,
                )
                plt.setp(axes[i, 0].spines.values(), color=None)

                axes[i, 1].axis("off")

                for j, pred in enumerate(pred_dict["predictions"]):
                    key_file_name = pred["file_name"]
                    image_idx = key_image_file_map[key_file_name]
                    image = load_image_from_h5(key_data, image_idx)
                    axes[i, 2 + j].imshow(image)  # subplot in col 1 is invisible
                    if i != 0 or j != 0:
                        axes[i, 2 + j].axis("off")
                    else:
                        axes[i, 2].set_xticks([])
                        axes[i, 2].set_yticks([])
                        plt.setp(axes[i, 2].spines.values(), color=None)

                    # species is correct
                    if pred_dict["query"]["taxonomy"]["species"] == pred["taxonomy"]["species"]:
                        bbox = axes[i, 2 + j].get_tightbbox(fig.canvas.get_renderer())
                        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                        fig.add_artist(
                            plt.Rectangle((x0, y0), width, height, edgecolor="#4C7C32", linewidth=3, fill=False)
                        )

                    # genus is correct
                    elif pred_dict["query"]["taxonomy"]["genus"] == pred["taxonomy"]["genus"]:
                        bbox = axes[i, 2 + j].get_tightbbox(fig.canvas.get_renderer())
                        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                        fig.add_artist(
                            plt.Rectangle((x0, y0), width, height, edgecolor="#C4D050", linewidth=3, fill=False)
                        )

            display_map = {"dna": "DNA", "image": "Image"}
            axes[0, 0].set_xlabel(
                f"Queries ({display_map[query_feature_type.split('_')[1]]})", loc="left", fontsize=24, labelpad=10
            )
            axes[0, 0].xaxis.set_label_position("top")
            axes[0, 2].set_xlabel(
                f"Keys ({display_map[key_feature_type.split('_')[1]]})", loc="left", fontsize=24, labelpad=10
            )
            axes[0, 2].xaxis.set_label_position("top")

            # draw line in between queries and keys
            fig.tight_layout()
            x0, _, width, _ = (
                axes[0, 1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
            )
            line_x = x0 + width / 2
            line = plt.Line2D((line_x, line_x), (0.12, 0.9), color="k", linewidth=1.5)
            fig.add_artist(line)

            fig.savefig(
                os.path.join(
                    folder_path,
                    f"query_{query_feature_type}_key_{key_feature_type}",
                    f"retrieval-images-{name}-query-{query_feature_type}-key-{key_feature_type}.pdf",
                ),
                transparent=True,
                bbox_inches="tight",
            )

    return retrieval_results


def avg_list(l):
    return sum(l) * 1.0 / len(l)


def calculate_silhouette_score(args, image_features, labels):
    for level in ["order", "family", "genus", "species"]:
        gt_list = [labels[1][i][level] for i in range(len(labels[1]))]
        silhouette_score = silhouette_samples(image_features, gt_list)
        print(f"The silhouette score for {level} level is : {avg_list(silhouette_score)}")


def make_prediction(query_feature, keys_feature, keys_label, with_similarity=False, with_indices=False, max_k=5):
    index = faiss.IndexFlatIP(keys_feature.shape[-1])
    keys_feature = normalize(keys_feature, norm="l2", axis=1).astype(np.float32)
    query_feature = normalize(query_feature, norm="l2", axis=1).astype(np.float32)
    index.add(keys_feature)
    pred_list = []

    similarities, indices = index.search(query_feature, max_k)
    for key_indices in indices:
        k_pred_in_diff_level = {}
        for level in LEVELS:
            if level not in k_pred_in_diff_level.keys():
                k_pred_in_diff_level[level] = []
            for i in key_indices:
                k_pred_in_diff_level[level].append(keys_label[i][level])


        pred_list.append(k_pred_in_diff_level)

    out = [pred_list]

    if with_similarity:
        out.append(similarities)

    if with_indices:
        out.append(indices)

    if len(out) == 1:
        return out[0]
    return out


def top_k_micro_accuracy(pred_list, gt_list, k_list=None):
    total_samples = len(pred_list)
    k_micro_acc = {}
    for k in k_list:
        if k not in k_micro_acc.keys():
            k_micro_acc[k] = {}
        for level in LEVELS:
            correct_in_curr_level = 0
            for pred_dict, gt_dict in zip(pred_list, gt_list):

                pred_labels = pred_dict[level][:k]
                gt_label = gt_dict[level]
                if gt_label in pred_labels:
                    correct_in_curr_level += 1
            k_micro_acc[k][level] = correct_in_curr_level * 1.0 / total_samples

    return k_micro_acc


def top_k_macro_accuracy(pred_list, gt_list, k_list=None):
    if k_list is None:
        k_list = [1, 3, 5]

    macro_acc_dict = {}
    per_class_acc = {}
    pred_counts = {}
    gt_counts = {}

    for k in k_list:
        macro_acc_dict[k] = {}
        per_class_acc[k] = {}
        pred_counts[k] = {}
        gt_counts[k] = {}
        for level in LEVELS:
            pred_counts[k][level] = {}
            gt_counts[k][level] = {}
            for pred, gt in zip(pred_list, gt_list):

                pred_labels = pred[level][:k]
                gt_label = gt[level]
                if gt_label not in pred_counts[k][level].keys():
                    pred_counts[k][level][gt_label] = 0
                if gt_label not in gt_counts[k][level].keys():
                    gt_counts[k][level][gt_label] = 0

                if gt_label in pred_labels:
                    pred_counts[k][level][gt_label] = pred_counts[k][level][gt_label] + 1
                gt_counts[k][level][gt_label] = gt_counts[k][level][gt_label] + 1

    for k in k_list:
        for level in LEVELS:
            sum_in_this_level = 0
            list_of_labels = list(gt_counts[k][level].keys())
            per_class_acc[k][level] = {}
            for gt_label in list_of_labels:
                sum_in_this_level = (
                    sum_in_this_level + pred_counts[k][level][gt_label] * 1.0 / gt_counts[k][level][gt_label]
                )
                per_class_acc[k][level][gt_label] = (
                    pred_counts[k][level][gt_label] * 1.0 / gt_counts[k][level][gt_label]
                )
            macro_acc_dict[k][level] = sum_in_this_level / len(list_of_labels)

    return macro_acc_dict, per_class_acc


def print_micro_and_macro_acc(acc_dict, k_list):
    header = [
        " ",
        "Seen Order",
        "Seen Family",
        "Seen Genus",
        "Seen Species",
        "Unseen Order",
        "Unseen Family",
        "Unseen Genus",
        "Unseen Species",
    ]

    rows = []
    rows_for_copy_to_google_doc = []
    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            for type_of_acc in ["micro_acc", "macro_acc"]:
                for k in k_list:
                    if len(list(acc_dict[query_feature_type][key_feature_type].keys())) == 0:
                        continue
                    curr_row = [
                        f"Query_feature: {query_feature_type}||Key_feature: {key_feature_type}||{type_of_acc} top-{k}"
                    ]
                    row_for_copy_to_google_doc = ""
                    for spit in ["seen_test", "unseen_test"]:
                        for level in LEVELS:
                            curr_row.append(
                                f"\t{round(acc_dict[query_feature_type][key_feature_type][spit][type_of_acc][k][level], 4)}"
                            )
                            row_for_copy_to_google_doc = (
                                row_for_copy_to_google_doc
                                + f"{round(acc_dict[query_feature_type][key_feature_type][spit][type_of_acc][k][level], 4)}\t"
                            )
                    rows.append(curr_row)
                    rows_for_copy_to_google_doc.append(row_for_copy_to_google_doc)
    table = Table(header, rows)
    table.print_table()

    print("For copy to google doc")
    for row in rows_for_copy_to_google_doc:
        print(row)


def inference_and_print_result(keys_dict, seen_test_dict, unseen_test_dict, small_species_list=None, k_list=None):
    acc_dict = {}
    per_class_acc = {}
    if k_list is None:
        k_list = [1, 3, 5]

    max_k = k_list[-1]

    seen_test_gt_label = seen_test_dict["label_list"]
    unseen_test_gt_label = unseen_test_dict["label_list"]
    keys_label = keys_dict["label_list"]
    pred_dict = {}

    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        acc_dict[query_feature_type] = {}
        per_class_acc[query_feature_type] = {}
        pred_dict[query_feature_type] = {}
        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            acc_dict[query_feature_type][key_feature_type] = {}
            per_class_acc[query_feature_type][key_feature_type] = {}
            pred_dict[query_feature_type][key_feature_type] = {}

            curr_seen_test_feature = seen_test_dict[query_feature_type]
            curr_unseen_test_feature = unseen_test_dict[query_feature_type]

            curr_keys_feature = keys_dict[key_feature_type]
            if key_feature_type == "all_key_features":
                if keys_dict["all_key_features_label"] is None:
                    continue
                else:
                    keys_label = keys_dict["all_key_features_label"]

            if (
                curr_keys_feature is None
                or curr_seen_test_feature is None
                or curr_unseen_test_feature is None
                or curr_keys_feature.shape[-1] != curr_seen_test_feature.shape[-1]
                or curr_keys_feature.shape[-1] != curr_unseen_test_feature.shape[-1]
            ):
                continue

            curr_seen_test_pred_list = make_prediction(
                curr_seen_test_feature, curr_keys_feature, keys_label, with_similarity=False, max_k=max_k
            )
            curr_unseen_test_pred_list = make_prediction(
                curr_unseen_test_feature, curr_keys_feature, keys_label, max_k=max_k
            )

            pred_dict[query_feature_type][key_feature_type] = {
                "curr_seen_test_pred_list": curr_seen_test_pred_list,
                "curr_unseen_test_pred_list": curr_unseen_test_pred_list,
            }

            acc_dict[query_feature_type][key_feature_type]["seen_test"] = {}
            acc_dict[query_feature_type][key_feature_type]["unseen_test"] = {}
            acc_dict[query_feature_type][key_feature_type]["seen_test"]["micro_acc"] = top_k_micro_accuracy(
                curr_seen_test_pred_list, seen_test_gt_label, k_list=k_list
            )
            acc_dict[query_feature_type][key_feature_type]["unseen_test"]["micro_acc"] = top_k_micro_accuracy(
                curr_unseen_test_pred_list, unseen_test_gt_label, k_list=k_list
            )

            seen_macro_acc, seen_per_class_acc = top_k_macro_accuracy(
                curr_seen_test_pred_list, seen_test_gt_label, k_list=k_list
            )

            unseen_macro_acc, unseen_per_class_acc = top_k_macro_accuracy(
                curr_unseen_test_pred_list, unseen_test_gt_label, k_list=k_list
            )

            per_class_acc[query_feature_type][key_feature_type]["seen_test"] = seen_per_class_acc
            per_class_acc[query_feature_type][key_feature_type]["unseen_test"] = unseen_per_class_acc

            acc_dict[query_feature_type][key_feature_type]["seen_test"]["macro_acc"] = seen_macro_acc
            acc_dict[query_feature_type][key_feature_type]["unseen_test"]["macro_acc"] = unseen_macro_acc

    print_micro_and_macro_acc(acc_dict, k_list)

    return acc_dict, per_class_acc, pred_dict


def check_for_acc_about_correct_predict_seen_or_unseen(final_pred_list, species_list):
    for k in [1, 3, 5]:
        correct = 0
        total = 0
        for record in final_pred_list:
            top_k_species = record["species"]
            curr_top_k_pred = top_k_species[:k]
            for single_pred in curr_top_k_pred:
                if single_pred in species_list:
                    correct = correct + 1
                    break
            total = total + 1

        print(f"for k = {k}: {correct * 1.0 / total}")


def get_features_and_label(dataloader, model, device, for_key_set=False):
    _, encoded_language_feature, label_list = get_feature_and_label(
        dataloader, model, device, type_of_feature="text", multi_gpu=False
    )

    _, encoded_dna_feature, label_list = get_feature_and_label(
        dataloader, model, device, type_of_feature="dna", multi_gpu=False
    )

    file_name_list, encoded_image_feature, label_list = get_feature_and_label(
        dataloader, model, device, type_of_feature="image", multi_gpu=False
    )

    averaged_feature = None
    concatenated_feature = None
    all_key_features = None
    all_key_features_label = None
    if encoded_dna_feature is not None and encoded_image_feature is not None:
        averaged_feature = np.mean([encoded_image_feature, encoded_dna_feature], axis=0)
        concatenated_feature = np.concatenate((encoded_image_feature, encoded_dna_feature), axis=1)

    dictionary_of_split = {
        "file_name_list": file_name_list,
        "encoded_dna_feature": encoded_dna_feature,
        "encoded_image_feature": encoded_image_feature,
        "encoded_language_feature": encoded_language_feature,
        "averaged_feature": averaged_feature,
        "concatenated_feature": concatenated_feature,
        "label_list": label_list,
    }

    if (
        for_key_set
        and encoded_image_feature is not None
        and encoded_dna_feature is not None
        and encoded_language_feature is not None
    ):
        for curr_feature in [encoded_image_feature, encoded_dna_feature, encoded_language_feature]:
            if all_key_features is None:
                all_key_features = curr_feature
                all_key_features_label = label_list
            else:
                all_key_features = np.concatenate((all_key_features, curr_feature), axis=0)
                all_key_features_label = all_key_features_label + label_list

    dictionary_of_split["all_key_features"] = all_key_features
    dictionary_of_split["all_key_features_label"] = all_key_features_label

    return dictionary_of_split


@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join(
        args.visualization.output_dir, args.model_config.model_output_name, "features_and_prediction"
    )
    os.makedirs(folder_for_saving, exist_ok=True)
    extracted_features_path = os.path.join(folder_for_saving, "5m_test_embedding_trained_with_5m.npz")
    labels_path = os.path.join(folder_for_saving, "labels.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(extracted_features_path) and os.path.exists(labels_path) and args.load_inference:
        print("Loading predictions from file...")
        npzfile = np.load(extracted_features_path)
        seen_test_dict = {
            "encoded_image_feature": npzfile["seen_np_all_image_feature"],
            "encoded_dna_feature": npzfile["seen_np_all_dna_feature"],
        }
        unseen_test_dict = {
            "encoded_image_feature": npzfile["unseen_np_all_image_feature"],
            "encoded_dna_feature": npzfile["unseen_np_all_dna_feature"],
        }
        keys_dict = {
            "encoded_image_feature": npzfile["keys_encoded_image_feature"],
            "encoded_dna_feature": npzfile["keys_encoded_dna_feature"],
        }

        with open(labels_path, "r") as json_file:
            total_dict = json.load(json_file)
        seen_test_dict["label_list"] = total_dict["seen_gt_dict"]
        unseen_test_dict["label_list"] = total_dict["unseen_gt_dict"]

    else:
        print("Init model...")
        model = load_clip_model(args, device)
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)

        print("Construct dataloader...")
        # Load data
        args.model_config.batch_size = 96
        if check_if_using_6m_data(args):
            _, _, seen_keys_dataloader, unseen_keys_dataloader = (
                load_bioscan_6M_dataloader_with_train_seen_and_separate_keys(args, for_pretrain=False)
            )
            _, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader = load_bioscan_dataloader_for_test_6m(
                args, for_pretrain=False)
            all_unique_seen_species = get_all_unique_species_from_dataloader(seen_keys_dataloader)
            all_unseen_species = get_all_unique_species_from_dataloader(unseen_keys_dataloader)

        else:
            _, _, _, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader = (
                load_bioscan_dataloader_with_train_seen_and_separate_keys(args, for_pretrain=False)
            )
            _, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader = load_bioscan_dataloader_for_test_6m(
                args, for_pretrain=False)

            all_unique_seen_species = get_all_unique_species_from_dataloader(seen_keys_dataloader)
            all_unique_test_unseen_species = get_all_unique_species_from_dataloader(val_unseen_keys_dataloader)
            all_unique_test_unseen_species = get_all_unique_species_from_dataloader(test_unseen_keys_dataloader)
            all_unseen_species = all_unique_test_unseen_species + all_unique_test_unseen_species

        keys_dict = get_features_and_label(all_keys_dataloader, model, device, for_key_set=True)

        seen_test_dict = get_features_and_label(seen_test_dataloader, model, device)

        unseen_test_dict = get_features_and_label(unseen_test_dataloader, model, device)

        # small_species_list = load_small_species(args)

        acc_dict, per_class_acc, pred_dict = inference_and_print_result(
            keys_dict,
            seen_test_dict,
            unseen_test_dict,
            small_species_list=None,
            k_list=args.inference_and_eval_setting.k_list,
        )

        seen_test_final_pred = pred_dict["encoded_image_feature"]["encoded_dna_feature"]["curr_seen_test_pred_list"]
        unseen_test_final_pred = pred_dict["encoded_image_feature"]["encoded_dna_feature"]["curr_unseen_test_pred_list"]



        print("For seen")
        check_for_acc_about_correct_predict_seen_or_unseen(seen_test_final_pred, all_unique_seen_species)
        print("For unseen")
        check_for_acc_about_correct_predict_seen_or_unseen(
            unseen_test_final_pred, all_unseen_species
        )

        with open("per_class_acc.json", "w") as json_file:
            json.dump(per_class_acc, json_file, indent=4)

        if args.save_inference:
            np.savez(
                extracted_features_path,
                seen_np_all_image_feature=seen_test_dict["encoded_image_feature"],
                seen_np_all_dna_feature=seen_test_dict["encoded_dna_feature"],
                unseen_np_all_image_feature=unseen_test_dict["encoded_image_feature"],
                unseen_np_all_dna_feature=unseen_test_dict["encoded_dna_feature"],
                keys_encoded_image_feature=keys_dict["encoded_image_feature"],
                keys_encoded_dna_feature=keys_dict["encoded_dna_feature"],
            )
            total_dict = {
                "seen_gt_dict": seen_test_dict["label_list"],
                "unseen_gt_dict": unseen_test_dict["label_list"],
                "seen_pred_dict_with_dna_key": pred_dict["encoded_image_feature"]["encoded_dna_feature"][
                    "curr_seen_test_pred_list"
                ],
                "unseen_pred_dict_with_dna_key": pred_dict["encoded_image_feature"]["encoded_dna_feature"][
                    "curr_unseen_test_pred_list"
                ],
                "seen_pred_dict_with_image_key": pred_dict["encoded_image_feature"]["encoded_image_feature"][
                    "curr_seen_test_pred_list"
                ],
                "unseen_pred_dict_with_image_key": pred_dict["encoded_image_feature"]["encoded_image_feature"][
                    "curr_unseen_test_pred_list"
                ],
            }

            with open(labels_path, "w") as json_file:
                json.dump(total_dict, json_file, indent=4)

    # if args.inference_and_eval_setting.plot_embeddings:
    #     generate_embedding_plot(
    #         args,
    #         seen_test_dict["encoded_image_feature"],
    #         seen_test_dict["encoded_dna_feature"],
    #         seen_test_dict["encoded_language_feature"],
    #         seen_test_dict["label_list"],
    #     )

    # if args.inference_and_eval_setting.retrieve_images:
    #     image_data = h5py.File(args.bioscan_data.path_to_hdf5_data, "r")
    #     retrieve_images(
    #         args,
    #         "val_seen",
    #         seen_test_dict,
    #         keys_dict,
    #         queries=["encoded_image_feature", "encoded_dna_feature"],
    #         keys=["encoded_image_feature", "encoded_dna_feature"],
    #         query_data=image_data["val_seen"],
    #         key_data=image_data["all_keys"],
    #     )
    #     retrieve_images(
    #         args,
    #         "val_unseen",
    #         unseen_test_dict,
    #         keys_dict,
    #         queries=["encoded_image_feature", "encoded_dna_feature"],
    #         keys=["encoded_image_feature", "encoded_dna_feature"],
    #         query_data=image_data["val_unseen"],
    #         key_data=image_data["all_keys"],
    #     )


if __name__ == "__main__":
    main()
