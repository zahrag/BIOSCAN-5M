import io
import os
from typing import Any
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from model.dna_encoder import get_sequence_pipeline
from model.language_encoder import load_pre_trained_bert
from torch.utils.data.distributed import DistributedSampler
import json
import time
from util.util import check_if_using_6m_data


class BioscanDataset(Dataset):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, metadata, image_hdf5=None, image_transforms=None, dna_transforms=None, **kwargs) -> None:
        super().__init__()

        self.metadata = pd.read_csv(metadata, sep="\t")
        self.images = h5py.File(image_hdf5, "r")["bioscan_dataset"]
        self.image_transforms = image_transforms
        self.dna_transforms = dna_transforms

    @property
    def image_ids(self):
        return self.metadata["image_files"]

    def load_image(self, key: str):
        byte_data = np.asarray(self.images[key])
        image = Image.open(io.BytesIO(byte_data))
        return image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> Any:
        row = self.metadata.iloc[index]
        image_file_name = row["image_file"]
        curr_image = self.load_image(image_file_name)
        if self.image_transforms:
            curr_image = self.image_transforms(curr_image).to(self.DEVICE).half()

        # prepare dna input
        dna_barcode = row["nucraw"]
        if self.dna_transforms:
            dna_barcode = self.dna_transforms(dna_barcode)
        curr_dna_input = torch.tensor(dna_barcode, dtype=torch.int64).to(self.DEVICE)
        return curr_image, curr_dna_input


class BadirliDataset(Dataset):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, metadata, image_folder=None, image_transforms=None, dna_transforms=None, **kwargs) -> None:
        super().__init__()

        metadata = sio.loadmat(metadata)
        image_ids = [id_.item() for id_ in metadata["ids"].flatten()]
        barcodes = [bc.item() for bc in metadata["nucleotides"].flatten()]
        species = [sp.item() for sp in metadata["species"].flatten()]
        self.metadata = pd.DataFrame.from_dict({"image_id": image_ids, "barcode": barcodes, "species": species})

        self.images = BadirliDataset.compile_images(image_folder)
        self.image_transforms = image_transforms
        self.dna_transforms = dna_transforms

        print(
            f"# overlapping images: {len(set(self.metadata['image_id'].tolist()).intersection(set(self.images.keys())))}"
        )

    @property
    def image_ids(self):
        return self.metadata["image_id"]

    @staticmethod
    def compile_images(image_folder):
        image_paths = {}
        for root, _, files in os.walk(image_folder):
            for file in files:
                image_id, ext = os.path.splitext(file)
                image_paths[image_id] = os.path.join(root, file)
        return image_paths

    def load_image(self, key: str):
        image_path = self.images[key]
        image = Image.open(image_path)
        return image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> Any:
        row = self.metadata.iloc[index]
        image_id = row["image_id"]
        # print(row["image_id"], row["species"])
        curr_image = self.load_image(image_id)
        if self.image_transforms:
            curr_image = self.image_transforms(curr_image).to(self.DEVICE).half()

        # prepare dna input
        dna_barcode = row["barcode"]
        if self.dna_transforms:
            dna_barcode = self.dna_transforms(dna_barcode)
        curr_dna_input = torch.tensor(dna_barcode, dtype=torch.int64).to(self.DEVICE)
        return curr_image, curr_dna_input


def get_label_ids(input_labels):
    label_to_id = {}

    for label in input_labels:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

    label_ids = np.array([label_to_id[label] for label in input_labels])
    return label_ids, label_to_id


def tokenize_dna_sequence(pipeline, dna_input):
    list_of_output = []
    for i in dna_input:
        list_of_output.append(pipeline(i))
    return list_of_output


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0, shuffle=False):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=sampler
    )

    return dataloader


def get_array_of_label_dicts(hdf5_inputs_path, split):
    hdf5_split_group = h5py.File(hdf5_inputs_path, "r", libver="latest")[split]
    np_order = np.array([item.decode("utf-8") for item in hdf5_split_group["order"][:]])
    np_family = np.array([item.decode("utf-8") for item in hdf5_split_group["family"][:]])
    np_genus = np.array([item.decode("utf-8") for item in hdf5_split_group["genus"][:]])
    np_species = np.array([item.decode("utf-8") for item in hdf5_split_group["species"][:]])
    array_of_dicts = np.array(
        [
            {"order": o, "family": f, "genus": g, "species": s}
            for o, f, g, s in zip(np_order, np_family, np_genus, np_species)
        ],
        dtype=object,
    )
    return array_of_dicts


def load_split_df_and_merge_with_df(args, df):
    split_df = pd.read_csv(args.bioscan_data.path_to_the_split, sep="\t")
    split_df = split_df[["sampleid", "split"]]
    df = pd.merge(df, split_df, on="sampleid", how="inner")

    return df

def get_bin_from_tsv(split, hef5_path, tsv_path):
    with h5py.File(hef5_path, "r") as h5file:
        sample_id_list = [item.decode('utf-8') for item in h5file[split]['sampleid']]
    df = pd.read_csv(tsv_path, sep='\t')
    filtered_df = df[df['sampleid'].isin(sample_id_list)]
    uri_list = filtered_df['uri'].tolist()
    return uri_list

def convert_uri_to_index_list(uri_list):
    string_to_int = {}
    next_int = 0
    integers = []
    for s in uri_list:
        if s not in string_to_int:
            string_to_int[s] = next_int
            next_int += 1
        integers.append(string_to_int[s])

    return integers

class Dataset_for_CL(Dataset):
    def __init__(
        self,
        args,
        split,
        length,
        image_type,
        dna_type,
        dna_tokens=None,
        return_language=False,
        labels=None,
        for_training=False,
    ):
        if check_if_using_6m_data(args):
            self.hdf5_inputs_path = args.bioscan_6m_data.path_to_hdf5_data
        else:
            self.hdf5_inputs_path = args.bioscan_data.path_to_hdf5_data

        self.split = split
        self.image_input_type = image_type
        self.dna_inout_type = dna_type
        if dna_tokens is not None:
            self.dna_tokens = torch.tensor(dna_tokens)
        self.length = length
        self.return_language = return_language
        self.for_training = for_training

        if self.for_training:
            if hasattr(args.model_config, "bin_for_positive_and_negative_pairs") and args.model_config.bin_for_positive_and_negative_pairs:
                self.labels = list(get_bin_from_tsv(split, self.hdf5_inputs_path, args.bioscan_data.path_to_tsv_data))
                self.labels = np.array(convert_uri_to_index_list(self.labels))
            elif labels is None:
                self.labels = np.array(list(range(length)))
            else:
                self.labels = labels
        else:
            self.labels = get_array_of_label_dicts(self.hdf5_inputs_path, split)

        if self.image_input_type == "image":
            if self.for_training:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.RandomResizedCrop(224, antialias=True),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=(-45, 45)),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.CenterCrop(224),
                    ]
                )
        elif self.image_input_type == "feature":
            self.transform = None
        else:
            raise TypeError(
                f"Image input type can not be {self.image_input_type}, it must be either 'image' or 'feature'. Please check the config file."
            )

        if self.dna_inout_type not in ["sequence", "feature"]:
            raise TypeError(
                f"DNA input type can not be {self.dna_inout_type}, it must be either 'sequence' or 'feature'. Please check the config file."
            )

    def __len__(self):
        return self.length

    def _open_hdf5(self):
        self.hdf5_split_group = h5py.File(self.hdf5_inputs_path, "r", libver="latest")[self.split]

    def load_image(self, idx):
        image_enc_padded = self.hdf5_split_group["image"][idx].astype(np.uint8)
        enc_length = self.hdf5_split_group["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        if self.transform is not None:
            curr_image = self.transform(curr_image)
        return curr_image

    def __getitem__(self, idx):
        if not hasattr(self, "hdf5_split_group"):
            self._open_hdf5()
        if self.image_input_type == "image":
            curr_image_input = self.load_image(idx)
        else:
            # Check if image feature are loaded correctly
            curr_image_input = self.hdf5_split_group["image_features"][idx].astype(np.float32)

        if self.dna_inout_type == "sequence":
            if self.dna_tokens is None:
                curr_dna_input = self.hdf5_split_group["barcode"][idx].decode("utf-8")
            else:
                curr_dna_input = self.dna_tokens[idx]
        else:
            # Check if DNA feature are loaded correctly
            curr_dna_input = self.hdf5_split_group["dna_features"][idx].astype(np.float32)

        if self.for_training:
            if self.return_language is False:
                return curr_image_input, curr_dna_input, self.labels[idx]
            else:
                language_input_ids = self.hdf5_split_group["language_tokens_input_ids"][idx]
                language_token_type_ids = self.hdf5_split_group["language_tokens_token_type_ids"][idx]
                language_attention_mask = self.hdf5_split_group["language_tokens_attention_mask"][idx]

                return (
                    curr_image_input,
                    curr_dna_input,
                    language_input_ids,
                    language_token_type_ids,
                    language_attention_mask,
                    self.labels[idx],
                )

        else:
            curr_file_name = self.hdf5_split_group["image_file"][idx].decode("utf-8")
            if self.return_language is False:
                return curr_file_name, curr_image_input, curr_dna_input, self.labels[idx]
            else:
                language_input_ids = self.hdf5_split_group["language_tokens_input_ids"][idx]
                language_token_type_ids = self.hdf5_split_group["language_tokens_token_type_ids"][idx]
                language_attention_mask = self.hdf5_split_group["language_tokens_attention_mask"][idx]
                return (
                    curr_file_name,
                    curr_image_input,
                    curr_dna_input,
                    language_input_ids,
                    language_token_type_ids,
                    language_attention_mask,
                    self.labels[idx],
                )


def get_len_dict(args):
    length_dict = {}
    if hasattr(args.model_config, 'dataset') and args.model_config.dataset == 'BIOSCAN_6M':
        with h5py.File(args.bioscan_6m_data.path_to_hdf5_data, "r") as h5file:
            for split in list(h5file.keys()):
                length_dict[split] = len(h5file[split]["image"])
    else:
        with h5py.File(args.bioscan_data.path_to_hdf5_data, "r") as h5file:
            for split in list(h5file.keys()):
                length_dict[split] = len(h5file[split]["image"])
    return length_dict


def construct_dataloader(
    args,
    split,
    length,
    sequence_pipeline,
    return_language=False,
    labels=None,
    for_pre_train=False,
    world_size=None,
    rank=None,
    shuffle=False,
):
    barcode_bert_dna_tokens = None
    # For now, just use sequence, but not feature.
    image_type = "image"
    if hasattr(args.model_config, "image"):
        image_type = args.model_config.image.input_type

    dna_type = "sequence"
    if hasattr(args.model_config, "dna"):
        dna_type = args.model_config.dna.input_type

    if dna_type == "sequence":
        if check_if_using_6m_data(args):
            hdf5_file = h5py.File(args.bioscan_6m_data.path_to_hdf5_data, "r", libver="latest")
        else:
            hdf5_file = h5py.File(args.bioscan_data.path_to_hdf5_data, "r", libver="latest")
        unprocessed_dna_barcode = np.array([item.decode("utf-8") for item in hdf5_file[split]["barcode"][:]])
        barcode_bert_dna_tokens = tokenize_dna_sequence(sequence_pipeline, unprocessed_dna_barcode)

    dataset = Dataset_for_CL(
        args,
        split,
        length,
        image_type=image_type,
        dna_type=dna_type,
        dna_tokens=barcode_bert_dna_tokens,
        return_language=return_language,
        labels=labels,
        for_training=for_pre_train,
    )

    num_workers = 8

    if hasattr(args.model_config, "num_workers"):
        num_workers = args.model_config.num_workers

    if for_pre_train:
        if world_size is not None and rank is not None:
            dataloader = prepare(
                dataset,
                rank,
                batch_size=args.model_config.batch_size,
                world_size=world_size,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=args.model_config.batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=num_workers,
            )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.model_config.batch_size, num_workers=num_workers, shuffle=shuffle
        )
    return dataloader


def load_bioscan_dataloader_with_train_seen_and_separate_keys(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    return_language = True

    sequence_pipeline = get_sequence_pipeline()

    train_seen_dataloader = construct_dataloader(
        args,
        "train_seen",
        length_dict["train_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_keys_dataloader = construct_dataloader(
        args,
        "seen_keys",
        length_dict["seen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    val_unseen_keys_dataloader = construct_dataloader(
        args,
        "val_unseen_keys",
        length_dict["val_unseen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    test_unseen_keys_dataloader = construct_dataloader(
        args,
        "test_unseen_keys",
        length_dict["test_unseen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    return (
        train_seen_dataloader,
        seen_val_dataloader,
        unseen_val_dataloader,
        seen_keys_dataloader,
        val_unseen_keys_dataloader,
        test_unseen_keys_dataloader,
    )


def load_bioscan_6M_dataloader(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    # TODO add label for supervised learning

    return_language = True

    sequence_pipeline = get_sequence_pipeline()

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    if for_pretrain:
        if (
                hasattr(args.model_config, "using_train_seen_for_pre_train")
                and args.model_config.using_train_seen_for_pre_train
        ):
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split_and_seen_train",
                length_dict["no_split_and_seen_train"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split",
                length_dict["no_split"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        return pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader
    else:
        train_seen_dataloader = construct_dataloader(
            args,
            "train_seen",
            length_dict["train_seen"],
            sequence_pipeline,
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        return train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader

def load_bioscan_dataloader(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    # TODO add label for supervised learning

    return_language = True

    sequence_pipeline = get_sequence_pipeline()

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    if for_pretrain:
        if (
            hasattr(args.model_config, "using_train_seen_for_pre_train")
            and args.model_config.using_train_seen_for_pre_train
        ):
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split_and_seen_train",
                length_dict["no_split_and_seen_train"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split",
                length_dict["no_split"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        return pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader
    else:
        train_seen_dataloader = construct_dataloader(
            args,
            "train_seen",
            length_dict["train_seen"],
            sequence_pipeline,
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        return train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader


def load_bioscan_dataloader_for_test(args, world_size=None, rank=None, for_pretrain=True):
    length_dict = get_len_dict(args)

    # TODO add label for supervised learning

    return_language = True

    sequence_pipeline = get_sequence_pipeline()

    seen_test_dataloader = construct_dataloader(
        args,
        "test_seen",
        length_dict["test_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_test_dataloader = construct_dataloader(
        args,
        "test_unseen",
        length_dict["test_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    all_keys_dataloader = construct_dataloader(
        args,
        "all_keys",
        length_dict["all_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    if for_pretrain:
        if (
            hasattr(args.model_config, "using_train_seen_for_pre_train")
            and args.model_config.using_train_seen_for_pre_train
        ):
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split_and_seen_train",
                length_dict["no_split_and_seen_train"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            pre_train_dataloader = construct_dataloader(
                args,
                "no_split",
                length_dict["no_split"],
                sequence_pipeline,
                return_language=return_language,
                labels=None,
                for_pre_train=True,
                world_size=world_size,
                rank=rank,
                shuffle=True,
            )
        return pre_train_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader
    else:
        train_seen_dataloader = construct_dataloader(
            args,
            "train_seen",
            length_dict["train_seen"],
            sequence_pipeline,
            return_language=return_language,
            labels=None,
            for_pre_train=False,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        return train_seen_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader


def load_bioscan_dataloader_all_small_splits(args, world_size=None, rank=None):
    length_dict = get_len_dict(args)

    return_language = True

    sequence_pipeline = get_sequence_pipeline()

    train_seen_dataloader = construct_dataloader(
        args,
        "train_seen",
        length_dict["train_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_val_dataloader = construct_dataloader(
        args,
        "val_seen",
        length_dict["val_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_val_dataloader = construct_dataloader(
        args,
        "val_unseen",
        length_dict["val_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_test_dataloader = construct_dataloader(
        args,
        "test_seen",
        length_dict["test_seen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    unseen_test_dataloader = construct_dataloader(
        args,
        "test_unseen",
        length_dict["test_unseen"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    seen_keys_dataloader = construct_dataloader(
        args,
        "seen_keys",
        length_dict["seen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    val_unseen_keys_dataloader = construct_dataloader(
        args,
        "val_unseen_keys",
        length_dict["val_unseen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )
    test_unseen_keys_dataloader = construct_dataloader(
        args,
        "test_unseen_keys",
        length_dict["test_unseen_keys"],
        sequence_pipeline,
        return_language=return_language,
        labels=None,
        for_pre_train=False,
        world_size=world_size,
        rank=rank,
    )

    return (
        train_seen_dataloader,
        seen_val_dataloader,
        unseen_val_dataloader,
        seen_test_dataloader,
        unseen_test_dataloader,
        seen_keys_dataloader,
        val_unseen_keys_dataloader,
        test_unseen_keys_dataloader,
    )
