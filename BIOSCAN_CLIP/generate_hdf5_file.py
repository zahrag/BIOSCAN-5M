import io
import math
import os
import time
from multiprocessing import Pool, Manager

import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from model.language_encoder import load_pre_trained_bert

# Manually calculated max length of the image size
MAX_LEN = 29523

def replace_non_with_not_classified(input):
    if input is None or (isinstance(input, float) and math.isnan(input)):
        return "not_classified"
    return input

def pil_image_to_byte(img):
    binary_data_io = io.BytesIO()
    img.save(binary_data_io, format='JPEG')
    binary_data = binary_data_io.getvalue()
    curr_image_np = np.frombuffer(binary_data, dtype=np.uint8)
    return curr_image_np


def process_image(args):
    idx, curr_image_name, chunk_number, special_arg_image_dir = args
    curr_image_path = os.path.join(special_arg_image_dir, f'part{str(chunk_number)}', curr_image_name)
    try:
        curr_image = Image.open(curr_image_path)
        curr_image_byte_np = pil_image_to_byte(curr_image)
        return idx, curr_image_byte_np.size, curr_image_byte_np
    except:
        return idx, None, None

@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()

    # load metadata
    metadata = pd.read_csv(args.bioscan_6m_data.path_to_tsv_data, sep="\t")
    target_splits = ['all_keys', 'no_split', 'no_split_and_seen_train', 'seen_keys', 'single_species', 'test_seen', 'test_unseen', 'test_unseen_keys', 'train_seen', 'val_seen', 'val_unseen', 'val_unseen_keys']

    datasets_to_create_for_each_split = ['barcode', 'family', 'genus', 'image',
                                         'image_file', 'image_mask', 'language_tokens_attention_mask',
                                         'language_tokens_input_ids', 'language_tokens_token_type_ids', 'order',
                                         'sampleid', 'species']

    special_datasets = ['language_tokens_attention_mask', 'language_tokens_input_ids', 'language_tokens_token_type_ids', 'image', 'image_mask', 'barcode']

    map_dict = {'all_keys': ['test_unseen_keys', 'val_unseen_keys', 'seen_keys'], 'no_split': ['pre_train'], 'no_split_and_seen_train': ['pre_train', 'train_seen'],
                'seen_keys': ['seen_keys'], 'single_species': ['single_species'], 'test_seen': ['seen_test_queries'], 'test_unseen': ['test_unseen_queries'],
                'test_unseen_keys': ['test_unseen_keys'], 'train_seen': ['seen_train'], 'val_seen': ['seen_val_queries'], 'val_unseen': ['val_unseen_queries'],
                'val_unseen_keys': ['val_unseen_keys']}

    special_arg_image_dir = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized'

    # Load language tokenizer for pre-processing

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    language_tokenizer, model = load_pre_trained_bert()

    start_time = time.time()

    count_for_missing_images = 0
    data_dict = {}

    print(f"All meta-splits {list(map_dict.keys())}")

    for meta_split in map_dict.keys():
        print(f"~~Meta split: Processing for {meta_split}")

        sub_splits = map_dict[meta_split]
        datasets = {}
        for dataset_name in datasets_to_create_for_each_split:
            datasets[dataset_name] = []
        for sub_split in sub_splits:
            print(f"~~~~Sub-split: Processing for {meta_split}")
            curr_sub_split_df = metadata[metadata['split'] == sub_split]

            # Process image and image_mask
            image_file_names = curr_sub_split_df['image_file'].tolist()
            chunk_numbers = curr_sub_split_df['chunk_number'].tolist()
            num_of_images = len(image_file_names)

            image_enc_padded = np.zeros((num_of_images, MAX_LEN), dtype=np.uint8)
            enc_lengths = np.zeros((num_of_images,), dtype=int)
            print(f"~~~~~~Processing images")

            with Manager() as manager:
                pbar = tqdm(total=num_of_images)
                with Pool() as pool:
                    results = []
                    for idx, curr_image_name in enumerate(image_file_names):
                        curr_chunk_number = chunk_numbers[idx]
                        args = (idx, curr_image_name, curr_chunk_number, special_arg_image_dir)
                        result = pool.apply_async(process_image, args=(args,))
                        results.append(result)

                    for result in results:
                        idx, byte_size, byte_data = result.get()
                        if byte_data is not None:
                            image_enc_padded[idx, :byte_size] = byte_data
                            enc_lengths[idx] = byte_size
                        else:
                            count_for_missing_images += 1
                        pbar.update(1)
                        pbar.set_description(f"Count of missing images: {count_for_missing_images}")

                pbar.close()
            """
            # Old version
            pbar = tqdm(enumerate(image_file_names), total=len(image_file_names))
            for idx, curr_image_name in pbar:
                curr_chunk_number = chunk_numbers[idx]
                curr_image_path = os.path.join(special_arg_image_dir, f'part{str(curr_chunk_number)}', curr_image_name)
                try:
                    curr_image = Image.open(curr_image_path)
                except:
                    count_for_missing_images = count_for_missing_images + 1
                    continue
                curr_image_byte_np = pil_image_to_byte(curr_image)

                image_enc_padded[idx, : curr_image_byte_np.size] = curr_image_byte_np
                enc_lengths[idx] = curr_image_byte_np.size
                pbar.set_description(f"Count of missing images: {count_for_missing_images}")
            """

            if len(datasets['image']) == 0:
                datasets['image'] = image_enc_padded
            else:
                datasets['image'] = np.concatenate((datasets['image'], image_enc_padded), axis=0)
            if len(datasets['image_mask']) == 0:
                datasets['image_mask'] = enc_lengths
            else:
                datasets['image_mask'] = np.concatenate((datasets['image_mask'], enc_lengths), axis=0)

            # Process language tokens
            list_of_language_input = []
            all_orders = curr_sub_split_df['order'].tolist()
            all_family = curr_sub_split_df['family'].tolist()
            all_genus = curr_sub_split_df['genus'].tolist()
            all_species = curr_sub_split_df['species'].tolist()

            for curr_order, curr_family, curr_genus, curr_species in zip(all_orders, all_family, all_genus, all_species):
                curr_order = replace_non_with_not_classified(curr_order)
                curr_family = replace_non_with_not_classified(curr_family)
                curr_genus = replace_non_with_not_classified(curr_genus)
                curr_species = replace_non_with_not_classified(curr_species)
                curr_language_input = curr_order + " " + curr_family + " " + curr_genus + " " + curr_species

                list_of_language_input.append(curr_language_input)


            language_tokens = language_tokenizer(list_of_language_input, padding="max_length", max_length=20,
                                                 truncation=True)
            language_tokens_input_ids = language_tokens['input_ids']
            language_tokens_token_type_ids = language_tokens['token_type_ids']
            language_tokens_attention_mask = language_tokens['attention_mask']
            datasets['language_tokens_input_ids'] = datasets['language_tokens_input_ids'] + language_tokens_input_ids
            datasets['language_tokens_token_type_ids'] = datasets['language_tokens_token_type_ids'] + language_tokens_token_type_ids
            datasets['language_tokens_attention_mask'] = datasets['language_tokens_attention_mask'] + language_tokens_attention_mask


            # For barcode
            datasets['barcode'] = datasets['barcode'] + curr_sub_split_df['nucraw'].tolist()

            # For other datasets
            for dataset_name in datasets_to_create_for_each_split:
                if dataset_name in special_datasets:
                    continue
                datasets[dataset_name] = datasets[dataset_name] + curr_sub_split_df[dataset_name].tolist()
        data_dict[meta_split] = datasets



    # Write data to hdf5
    new_file = h5py.File(file_path, "w")
    for meta_split in map_dict.keys():
        group = new_file.create_group(meta_split)
        for dataset_name in datasets_to_create_for_each_split:
            group.create_dataset(dataset_name, data=data_dict[meta_split][dataset_name])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"In： {(elapsed_time/60/60):.2f} seconds")
    print(f"Missing images {count_for_missing_images}")


if __name__ == '__main__':
    main()
    print()
