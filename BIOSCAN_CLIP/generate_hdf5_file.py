import copy
import datetime
import json
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from PIL import Image
import io
from model.language_encoder import load_pre_trained_bert

# Manually calculated max length of the image size
MAX_LEN = 29523

def pil_image_to_byte(img):
    binary_data_io = io.BytesIO()
    img.save(binary_data_io, format='JPEG')
    binary_data = binary_data_io.getvalue()
    curr_image_np = np.frombuffer(binary_data, dtype=np.uint8)
    return curr_image_np


@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()

    # load metadata
    metadata = pd.read_csv(args.bioscan_6m_data.path_to_tsv_data, sep="\t")
    targer_splits = ['all_keys', 'no_split', 'no_split_and_seen_train', 'seen_keys', 'single_species', 'test_seen', 'test_unseen', 'test_unseen_keys', 'train_seen', 'val_seen', 'val_unseen', 'val_unseen_keys']

    datasets_to_create_for_each_split = ['barcode', 'dna_features', 'family', 'genus', 'image',
                                         'image_file', 'image_mask', 'language_tokens_attention_mask',
                                         'language_tokens_input_ids', 'language_tokens_token_type_ids', 'order',
                                         'sampleid', 'species']

    special_datasets = ['language_tokens_attention_mask', 'language_tokens_input_ids', 'language_tokens_token_type_ids', 'image', 'image_mask']

    map_dict = {'all_keys': ['test_unseen_keys', 'val_unseen_keys', 'seen_keys'], 'no_split': ['pre_train'], 'no_split_and_seen_train': ['pre_train', 'train_seen'],
                'seen_keys': ['seen_keys'], 'single_species': ['single_species'], 'test_seen': ['seen_test_queries'], 'test_unseen': ['test_unseen_queries'],
                'test_unseen_keys': ['test_unseen_keys'], 'train_seen': ['seen_train'], 'val_seen': ['seen_val_queries'], 'val_unseen': ['val_unseen_queries'],
                'val_unseen_keys': ['val_unseen_keys']}

    special_arg_image_dir = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized'

    # Load language tokenizer for pre-processing
    language_tokenizer, model = load_pre_trained_bert()


    for meta_split in map_dict.keys():
        sub_splits = map_dict[meta_split]

        datasets = {}
        for dataset_name_in_split in datasets_to_create_for_each_split:
            datasets[dataset_name_in_split] = []
        for sub_split in sub_splits:
            curr_sub_split_df = metadata[metadata['split'] == sub_split]

            # Process image and image_mask
            image_file_names = curr_sub_split_df['image_file'].tolist()
            chunk_numaber = curr_sub_split_df['chunk_number'].tolist()
            num_of_images = len(image_file_names)

            image_enc_padded = np.zeros((num_of_images, MAX_LEN), dtype=np.uint8)
            enc_lengths = np.zeros((num_of_images,), dtype=int)
            for idx, curr_image_name in enumerate(image_file_names):
                curr_chunk_number = chunk_numaber[idx]
                curr_image_path = os.path.join(special_arg_image_dir, f'part{str(curr_chunk_number)}', curr_image_name)
                curr_image = Image.open(curr_image_path)
                curr_image_byte_np = pil_image_to_byte(curr_image)

                image_enc_padded[idx, : curr_image_byte_np.size] = curr_image_byte_np
                enc_lengths[idx] = curr_image_byte_np.size

            # Process language tokens

            # For other datasets
            for dataset_name_in_split in datasets_to_create_for_each_split:
                if dataset_name_in_split in special_datasets:
                    continue
                datasets[dataset_name_in_split] = datasets[dataset_name_in_split] + curr_sub_split_df[dataset_name_in_split].tolist()


if __name__ == '__main__':
    main()
