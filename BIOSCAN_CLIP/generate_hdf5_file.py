import io
import math
import os
import time
from multiprocessing import Pool, Manager, cpu_count

import hydra
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm
import h5py

from model.language_encoder import load_pre_trained_bert
import sys

# Manually calculated max length of the image size
MAX_LEN = 29598
SPECIAL_ARG_IMAGE_DIR = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized/new_org'


class Tee:
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        if self.stdout != sys.stdout:
            sys.stdout = self.stdout
        if self.file:
            self.file.close()


def replace_non_with_not_classified(input):
    if input is None or (isinstance(input, float) and math.isnan(input)):
        return "not_classified"
    return input


def replace_non_with_not_classified_for_list(inputs):
    result = []
    for input in inputs:
        if input is None or (isinstance(input, float) and math.isnan(input)):
            result.append("not_classified")
        else:
            result.append(input)
    return result


def pil_image_to_byte(img):
    binary_data_io = io.BytesIO()
    img.save(binary_data_io, format='JPEG')
    binary_data = binary_data_io.getvalue()
    curr_image_np = np.frombuffer(binary_data, dtype=np.uint8)
    return curr_image_np


def check_element(arr):
    for element in arr:
        if not isinstance(element, str):  # Check if the element is not a string
            print(f'Element "{element}" is of type {type(element).__name__}')  # Print the data type of the element


def process_batch_of_images_to_bytes(args):
    idx, curr_image_name, chunk_number, special_arg_image_dir = args
    curr_image_path = os.path.join(special_arg_image_dir, curr_image_name[:2], curr_image_name)
    try:
        curr_image = Image.open(curr_image_path)
        curr_image_byte_np = pil_image_to_byte(curr_image)
        return idx, curr_image_byte_np.size, curr_image_byte_np
    except:
        return idx, None, None


def split_list_into_sub_lists(input_list, sub_list_size):
    return [input_list[i:i + sub_list_size] for i in range(0, len(input_list), sub_list_size)]


def add_new_info_to_the_dataset(dataset, new_data):
    original_shape = dataset.shape[0]
    new_size = original_shape + new_data.shape[0]
    dataset.resize(new_size, axis=0)
    if len(dataset.shape) == 2:
        dataset[original_shape:new_size, :] = new_data
    else:
        dataset[original_shape:new_size] = new_data


def image_process_for_unit_size(group, image_file_names, chunk_numbers, special_arg_image_dir,
                                count_for_missing_images):
    num_of_images = len(image_file_names)
    image_enc_padded = np.zeros((num_of_images, MAX_LEN), dtype=np.uint8)
    enc_lengths = np.zeros((num_of_images,), dtype=int)
    with Manager() as manager:
        pbar = tqdm(total=num_of_images)
        with Pool() as pool:
            results = []
            for idx, curr_image_name in enumerate(image_file_names):
                curr_chunk_number = chunk_numbers[idx]
                args = (idx, curr_image_name, curr_chunk_number, special_arg_image_dir)
                result = pool.apply_async(process_batch_of_images_to_bytes, args=(args,))
                results.append(result)

            for result in results:
                idx, byte_size, byte_data = result.get()
                if byte_data is not None:
                    image_enc_padded[idx, :byte_size] = byte_data
                    enc_lengths[idx] = byte_size
                else:
                    count_for_missing_images += 1
                # Get the memory details
                mem = psutil.virtual_memory()

                total_memory = mem.total / (1024 ** 3)  # Convert bytes to GB

                used_memory = mem.used / (1024 ** 3)  # Convert bytes to GB
                pbar.update(1)
                pbar.set_description(
                    f"Count of missing images: {count_for_missing_images}|| Memory usage: {used_memory}/{total_memory} GB")
                if used_memory / total_memory >= 0.9:
                    print(
                        f"Count of missing images: {count_for_missing_images}|| Memory usage: {used_memory}/{total_memory} GB")
                    print('Memory overflow.')
                    exit()
        pbar.close()

    add_new_info_to_the_dataset(group['image'], image_enc_padded)
    add_new_info_to_the_dataset(group['image_mask'], enc_lengths)

    return count_for_missing_images


def image_process(group, image_file_names, chunk_numbers, special_arg_image_dir, count_for_missing_images,
                  max_len_for_each_loop=100000):
    num_of_images = len(image_file_names)

    if num_of_images > max_len_for_each_loop:
        list_of_list_of_image_file_names = split_list_into_sub_lists(image_file_names, max_len_for_each_loop)
        list_of_list_of_chunk_numbers = split_list_into_sub_lists(chunk_numbers, max_len_for_each_loop)
        for idx in range(len(list_of_list_of_image_file_names)):
            curr_list_of_image_file_names = list_of_list_of_image_file_names[idx]
            curr_list_of_chunk_numbers = list_of_list_of_chunk_numbers[idx]
            count_for_missing_images = image_process_for_unit_size(group, curr_list_of_image_file_names,
                                                                   curr_list_of_chunk_numbers, special_arg_image_dir,
                                                                   count_for_missing_images)
    else:
        count_for_missing_images = image_process_for_unit_size(group, image_file_names, chunk_numbers,
                                                               special_arg_image_dir,
                                                               count_for_missing_images)
    return count_for_missing_images


def convert_to_numpy_if_list(input_data):
    if isinstance(input_data, list) and isinstance(input_data[0], str):
        return np.array(input_data, dtype='S')
    else:
        return input_data

def check_image_exists(args):
    processid, chunk_number, image_dir = args
    curr_image_name = f'{processid}.1024px.jpg'
    curr_image_path = os.path.join(image_dir, curr_image_name[:2], curr_image_name)
    return os.path.exists(curr_image_path), processid

def remove_rows_with_missing_images(metadata):
    process_id_list = metadata['processid'].tolist()
    chunk_number_list = [process_id[:2] for process_id in process_id_list]

    args_list = [(row['processid'], chunk_number_list[index], SPECIAL_ARG_IMAGE_DIR) for index, row in metadata.iterrows()]

    pool = Pool(processes=cpu_count())

    results = list(tqdm(pool.imap(check_image_exists, args_list), total=len(args_list)))

    pool.close()
    pool.join()

    existing_images = {processid for exists, processid in results if exists}
    missing_images = [processid for exists, processid in results if not exists]
    print(f'Missing {len(missing_images)} images')

    metadata = metadata[metadata['processid'].isin(existing_images)]

    return metadata

@hydra.main(config_path="config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    print(args.bioscan_6m_data.path_to_tsv_data)
    # load metadata
    metadata = pd.read_csv(args.bioscan_6m_data.path_to_tsv_data, sep="\,")

    print("Before cleaning")
    print(len(metadata))
    metadata = remove_rows_with_missing_images(metadata)
    print("After cleaning")
    print(len(metadata))

    datasets_to_create_for_each_split = ['barcode', 'family', 'genus', 'image',
                                         'image_file', 'image_mask', 'language_tokens_attention_mask',
                                         'language_tokens_input_ids', 'language_tokens_token_type_ids', 'order',
                                         'sampleid', 'species', 'processid']

    special_datasets = ['language_tokens_attention_mask', 'language_tokens_input_ids', 'language_tokens_token_type_ids',
                        'image', 'image_mask', 'barcode']

    map_dict = {'all_keys': {'split': ['key_unseen', 'train']},
                'train_keys': {'split': ['train']},
                'val_seen': {'split': ['val']},
                'test_seen': {'split': ['test']},
                'seen_keys': {'split': ['train']},
                'test_unseen': {'split': ['test_unseen']},
                'val_unseen': {'split': ['val_unseen']},
                'unseen_keys': {'split': ['key_unseen']},
                'no_split_and_seen_train': {'split': ['pretrain', 'train']},
                'other_heldout': {'split': ['other_heldout']},
                }

    # Load language tokenizer for pre-processing

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    language_tokenizer, model = load_pre_trained_bert()

    start_time = time.time()

    count_for_missing_images = 0

    print(f"All meta-splits {list(map_dict.keys())}")
    new_file = h5py.File(args.bioscan_6m_data.path_to_hdf5_data, "w")

    for meta_split in map_dict.keys():
        print(f"~~Meta split: Processing for {meta_split}")
        group = new_file.create_group(meta_split)
        group.create_dataset('image', shape=(0, MAX_LEN), maxshape=(None, MAX_LEN), dtype='uint8')
        group.create_dataset('image_mask', shape=(0,), maxshape=(None,), dtype='int')
        sub_split = map_dict[meta_split]
        datasets = {}
        for dataset_name in datasets_to_create_for_each_split:
            datasets[dataset_name] = []
        s = None
        r = None
        if meta_split == 'pretrain_only':
            ss = sub_split['species_status']
            curr_sub_split_df = metadata[metadata['species_status'].isin(ss)]
        elif 'role' not in sub_split.keys():
            s = sub_split['split']
            curr_sub_split_df = metadata[metadata['split'].isin(s)]
        else:
            s = sub_split['split']
            r = sub_split['role']
            curr_sub_split_df = metadata[metadata['split'].isin(s) & metadata['role'].isin(r)]
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
        datasets['language_tokens_token_type_ids'] = datasets[
                                                         'language_tokens_token_type_ids'] + language_tokens_token_type_ids
        datasets['language_tokens_attention_mask'] = datasets[
                                                         'language_tokens_attention_mask'] + language_tokens_attention_mask
        # For barcode
        datasets['barcode'] = datasets['barcode'] + curr_sub_split_df['dna_barcode'].tolist()
        # For other datasets
        for dataset_name in datasets_to_create_for_each_split:
            if dataset_name in special_datasets:
                continue
            datasets[dataset_name] = datasets[dataset_name] + curr_sub_split_df[dataset_name].tolist()
        # Process image and image_mask
        print(f"~~~~~~Processing images")
        image_file_names = [f'{processid}.1024px.jpg' for processid in curr_sub_split_df['processid'].tolist()]
        chunk_numbers = curr_sub_split_df['chunk_number'].tolist()
        image_process(group, image_file_names, chunk_numbers, SPECIAL_ARG_IMAGE_DIR, count_for_missing_images)
        print(f"~~~~~~Done with images")

        # Writing to hdf5
        pbar = tqdm(datasets_to_create_for_each_split)
        for dataset_name in pbar:
            pbar.set_description(f"Writing {dataset_name} for {meta_split}")
            if dataset_name == "image" or dataset_name == "image_mask":
                continue
            if dataset_name == "image_file":
                correct_image_file_name_list = [f'{processid}.1024px.jpg' for processid in datasets['processid']]
                datasets[dataset_name] = convert_to_numpy_if_list(correct_image_file_name_list)
            elif dataset_name in ['order', 'family', 'genus', 'species']:
                datasets[dataset_name] = replace_non_with_not_classified_for_list(datasets[dataset_name])
            else:
                datasets[dataset_name] = convert_to_numpy_if_list(datasets[dataset_name])
            try:
                group.create_dataset(dataset_name, data=datasets[dataset_name])
            except:
                check_element(datasets[dataset_name])
                print("Failed when writ to hdf5")
                exit()

    new_file.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"In： {(elapsed_time / 60 / 60):.2f} hours")
    print(f"Missing images {count_for_missing_images}")


if __name__ == '__main__':
    log = Tee('output_for_save_hdf5.log', 'w')
    try:
        main()
    finally:
        log.close()
