import io
import os
from typing import Any

import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from model.dna_encoder import get_sequence_pipeline
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import time
import json
import numpy as np
import h5py

def tokenize_dna_sequence(pipeline, dna_input):
    list_of_output = []
    for i in dna_input:
        list_of_output.append(pipeline(i))
    return list_of_output


def species_list_to_input_string_list(species_list, species_to_others):
    levels = ['order', 'family', 'genus']
    four_labels_lits = []

    for species in species_list:
        curr_str = ""
        for level in levels:
            if level not in species_to_others[species]:
                species_to_others[species][level] = 'not_classified'

            curr_str = curr_str + species_to_others[species][level] + " "
        curr_str = curr_str + species
        four_labels_lits.append(curr_str)

    return four_labels_lits

def species_list_to_labels(species_list, species_to_others):
    levels = ['order', 'family', 'genus']
    four_labels_lits = []
    for key in species_to_others.keys():
        for level in levels:
            if level not in species_to_others[key].keys():
                species_to_others[key][level] = 'not_classified'

    np_order = np.array([species_to_others[species]['order'] for species in species_list])
    np_family = np.array([species_to_others[species]['family'] for species in species_list])
    np_genus = np.array([species_to_others[species]['genus'] for species in species_list])
    np_species = np.array(species_list)


    array_of_dicts = np.array([
        {'order': o, 'family': f, 'genus': g, 'species': s}
        for o, f, g, s in zip(np_order, np_family, np_genus, np_species)
    ], dtype=object)

    return array_of_dicts


class INSECTDataset(Dataset):


    def __init__(self, path_to_att_splits_mat, path_to_res_101_mat, image_hdf5_path, dna_transforms, species_to_others,
                 split, for_training=False, cl_label=False, **kwargs) -> None:
        super().__init__()
        # self.metadata = pd.read_csv(metadata, sep="\t")

        att_splits_mat = sio.loadmat(path_to_att_splits_mat)
        res_101_mat = sio.loadmat(path_to_res_101_mat)
        # split_loc = att_splits_mat[split][0]

        image_ids = [id_.item() for id_ in res_101_mat["ids"].flatten()]
        barcodes = [bc.item() for bc in res_101_mat["nucleotides"].flatten()]
        species = [sp.item() for sp in res_101_mat["species"].flatten()]

        if split != "all":
            split_loc = att_splits_mat[split][0]
            image_ids = [image_ids[i - 1] for i in split_loc]
            barcodes = [barcodes[i - 1] for i in split_loc]
            species = [species[i - 1] for i in split_loc]

        four_label_str = species_list_to_input_string_list(species, species_to_others)

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        batch_encoded = tokenizer(four_label_str, return_tensors='pt', padding=True)

        barcodes = tokenize_dna_sequence(dna_transforms, barcodes)

        self.for_training = for_training

        if self.for_training and cl_label:
            self.labels = np.array(list(range(len(image_ids))))
        else:
            self.labels = species_list_to_labels(species, species_to_others)

        self.input_ids = batch_encoded['input_ids']
        self.attention_mask = batch_encoded['attention_mask']
        self.token_type_ids = batch_encoded.get('token_type_ids')
        self.metadata = pd.DataFrame.from_dict({"image_id": image_ids, "barcode": barcodes, "species": species})

        # self.images = INSECTDataset.compile_images(image_folder)
        self.images = image_hdf5_path

        if self.for_training:
            self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=256, antialias=True),
                                                        transforms.RandomResizedCrop(224, antialias=True),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.RandomRotation(degrees=(-45, 45)),
                                                        transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                               saturation=0.5,
                                                                               hue=0.5), ])
        else:
            self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=256, antialias=True),
                                                        transforms.CenterCrop(224),
                                                        ])

    @property
    def image_ids(self):
        return self.metadata["image_id"]

    def _open_hdf5(self):
        self.hdf5_images = h5py.File(self.images, "r")

    def load_image(self, id):
        selected_dataset = self.hdf5_images['images'][id]
        curr_image = Image.open(io.BytesIO(selected_dataset[:]))
        if self.image_transforms is not None:
            curr_image = self.image_transforms(curr_image)
        return curr_image

    def __len__(self):
        return len(self.metadata["image_id"])

    def __getitem__(self, index) -> Any:
        if not hasattr(self, 'hdf5_images'):
            self._open_hdf5()
        row = self.metadata.iloc[index]
        image_id = row["image_id"]

        curr_image = self.load_image(image_id)

        # prepare dna input
        dna_barcode = row["barcode"]
        dna_barcode = torch.tensor(dna_barcode, dtype=torch.int64)

        label = self.labels[index]

        if self.for_training:
            return curr_image, dna_barcode, self.input_ids[index], self.attention_mask[index], \
                self.token_type_ids[index], label

        return image_id, curr_image, dna_barcode, self.input_ids[index], self.attention_mask[index], \
        self.token_type_ids[index], label


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=8, shuffle=False):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            sampler=sampler)

    return dataloader


def load_insect_dataloader_trainval(args,num_workers=8, shuffle_for_train_seen_key=False):
    filename = args.insect_data.species_to_other
    with open(filename, 'r') as file:
        specie_to_other_labels = json.load(file)

    sequence_pipeline = get_sequence_pipeline()

    trainval_dataset = INSECTDataset(
        args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
        species_to_others=specie_to_other_labels, split="trainval_loc",
        image_hdf5_path=args.insect_data.path_to_image_hdf5,
        dna_transforms=sequence_pipeline, for_training=True, cl_label=False
    )


    insect_trainval_dataloader = DataLoader(trainval_dataset, batch_size=args.model_config.batch_size,
                                       num_workers=num_workers, shuffle=True)

    return insect_trainval_dataloader

def load_insect_dataloader(args, world_size=None, rank=None, num_workers=8, load_all_in_one=False, shuffle_for_train_seen_key=False):
    filename = args.insect_data.species_to_other
    with open(filename, 'r') as file:
        specie_to_other_labels = json.load(file)

    sequence_pipeline = get_sequence_pipeline()

    if load_all_in_one:
        all_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="all",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=False
        )

        all_dataloader = DataLoader(all_dataset, batch_size=args.model_config.batch_size,
                                    num_workers=num_workers, shuffle=False)
        return all_dataloader

    else:
        train_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="train_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=True
        )

        train_dataset_for_key = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="train_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=False
        )

        val_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="val_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=False
        )

        test_seen_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="test_seen_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=False
        )

        test_unseen_dataset = INSECTDataset(
            args.insect_data.path_to_att_splits_mat, args.insect_data.path_to_res_101_mat,
            species_to_others=specie_to_other_labels, split="test_unseen_loc",
            image_hdf5_path=args.insect_data.path_to_image_hdf5,
            dna_transforms=sequence_pipeline, for_training=False
        )
        if rank is None:
            print(rank)
            insect_train_dataloader = DataLoader(train_dataset, batch_size=args.model_config.batch_size,
                                              num_workers=num_workers, shuffle=True)
        else:
            insect_train_dataloader = prepare(train_dataset, rank, batch_size=args.model_config.batch_size,
                                          world_size=world_size,
                                          num_workers=num_workers, shuffle=True)

        insect_train_dataloader_for_key = DataLoader(train_dataset_for_key, batch_size=args.model_config.batch_size,
                                                         num_workers=num_workers, shuffle=shuffle_for_train_seen_key)
        insect_val_dataloader = DataLoader(val_dataset, batch_size=args.model_config.batch_size,
                                           num_workers=num_workers, shuffle=False)

        insect_test_seen_dataloader = DataLoader(test_seen_dataset, batch_size=args.model_config.batch_size,
                                                 num_workers=num_workers, shuffle=False)

        insect_test_unseen_dataloader = DataLoader(test_unseen_dataset, batch_size=args.model_config.batch_size,
                                                   num_workers=num_workers, shuffle=False)

        return insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader
