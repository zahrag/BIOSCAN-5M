"""
create_splits.py
----------------
Create data splits for BIOSCAN dataset. The splits are designed as follows:

all data -> filter(num_samples_per_species > 10) -> filtered_data
species -> seen (0.8), unseen (0.2)
seen species -> train_seen (0.7), val_seen (0.1), test_seen (0.1), key_seen (0.1)
unseen species -> val_unseen (0.25), test_unseen (0.25), val_unseen_query (0.25), test_unseen_query (0.25)
tail_species -> species with less than 10 and more than 2 samples. -> tail_metadata(0.5), tail_query(0.5)
For now, tail_metadata are merged with test_unseen, and tail_query are merged with test_unseen_query

single_species -> species with single sample in the dataset, for now are using for pre-train.
"""
import os
import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm
from dataset_helper import CustomArg

np.random.seed(42)

def split_common_species_df(metadata_df_common_species):
    unique_species = metadata_df_common_species['species'].unique()
    np.random.shuffle(unique_species)

    n_species = len(unique_species)
    n_seen = int(0.8 * n_species)
    n_val_unseen = int(0.1 * n_species)

    seen_species = unique_species[:n_seen]
    val_unseen_species = unique_species[n_seen:n_seen + n_val_unseen]
    test_unseen_species = unique_species[n_seen + n_val_unseen:]


    # Create the DataFrames based on these splits
    seen_df = metadata_df_common_species[metadata_df_common_species['species'].isin(seen_species)]
    val_unseen_df = metadata_df_common_species[metadata_df_common_species['species'].isin(val_unseen_species)]
    test_unseen_df = metadata_df_common_species[metadata_df_common_species['species'].isin(test_unseen_species)]

    return seen_df, val_unseen_df, test_unseen_df

def split_less_common_species_df(metadata_df_less_species):
    unique_species = metadata_df_less_species['species'].unique()
    np.random.shuffle(unique_species)

    n_species = len(unique_species)
    n_val_unseen = int(0.5 * n_species)

    val_unseen_species = unique_species[:n_val_unseen]
    test_unseen_species = unique_species[n_val_unseen:]

    val_unseen_df = metadata_df_less_species[metadata_df_less_species['species'].isin(val_unseen_species)]
    test_unseen_df = metadata_df_less_species[metadata_df_less_species['species'].isin(test_unseen_species)]

    return val_unseen_df, test_unseen_df


def split_seen(group):
    train = group.sample(frac=0.7, random_state=42)
    group = group.drop(train.index)

    seen_keys = group.sample(frac=1.0/3.0, random_state=42)
    group = group.drop(seen_keys.index)

    seen_val_queries = group.sample(frac=1.0/2.0, random_state=42)
    seen_test_queries = group.drop(seen_val_queries.index)

    return train, seen_val_queries, seen_test_queries, seen_keys


def split_val_and_test_unseen(group):
    unseen_queries = group.sample(frac=0.5, random_state=42)
    unseen_keys = group.drop(unseen_queries.index)
    return unseen_queries, unseen_keys


def make_split(configs):

    if not configs['split']:
        return

    args = CustomArg(configs)
    metadata_df = pd.read_csv(args.metadata, sep='\t', low_memory=False)
    metadata_df.replace('no_data', np.nan, inplace=True)
    metadata_df['split'] = None
    metadata_df.loc[metadata_df['species'].isnull(), 'split'] = 'pre_train'

    metadata_df_pre_train = metadata_df[metadata_df['split'] == 'pre_train']
    metadata_df_species_not_none = metadata_df[metadata_df['species'].notnull()]


    species_list = metadata_df_species_not_none['species'].tolist()
    species_counter = Counter(species_list)
    most_common_species = [item for item, count in species_counter.items() if count >= 10]
    less_common_species = [item for item, count in species_counter.items() if count < 10 and count >= 2]
    single_species = [item for item, count in species_counter.items() if count == 1]

    metadata_df_common_species = metadata_df_species_not_none[metadata_df_species_not_none['species'].isin(most_common_species)]
    metadata_df_less_common_species = metadata_df_species_not_none[metadata_df_species_not_none['species'].isin(less_common_species)]
    metadata_df_single_species = metadata_df_species_not_none[metadata_df_species_not_none['species'].isin(single_species)]

    seen_df, val_unseen_df, test_unseen_df = split_common_species_df(metadata_df_common_species)
    val_unseen_df_from_less_common, test_unseen_df_from_less_common = split_less_common_species_df(metadata_df_less_common_species)

    val_unseen_df = pd.concat([val_unseen_df, val_unseen_df_from_less_common])
    test_unseen_df = pd.concat([test_unseen_df, test_unseen_df_from_less_common])

    seen_train_df = pd.DataFrame()
    seen_val_queries_df = pd.DataFrame()
    seen_test_queries_df = pd.DataFrame()
    seen_keys_df = pd.DataFrame()

    pbar = tqdm(seen_df.groupby('species'),
                total=len(seen_df['species'].unique()))
    for _, group in pbar:
        pbar.set_description("Processing seen species: ")
        seen_train, seen_val_queries, seen_test_queries, seen_keys = split_seen(group)
        seen_train_df = pd.concat([seen_train_df, seen_train])
        seen_val_queries_df = pd.concat([seen_val_queries_df, seen_val_queries])
        seen_test_queries_df = pd.concat([seen_test_queries_df, seen_test_queries])
        seen_keys_df = pd.concat([seen_keys_df, seen_keys])

    val_unseen_queries_df = pd.DataFrame()
    val_unseen_keys_df = pd.DataFrame()

    pbar = tqdm(val_unseen_df.groupby('species'),
                total=len(val_unseen_df['species'].unique()))
    for _, group in pbar:
        pbar.set_description("Processing val unseen species: ")
        val_unseen_queries, val_unseen_keys = split_val_and_test_unseen(group)
        val_unseen_queries_df = pd.concat([val_unseen_queries_df, val_unseen_queries])
        val_unseen_keys_df = pd.concat([val_unseen_keys_df, val_unseen_keys])

    test_unseen_queries_df = pd.DataFrame()
    test_unseen_keys_df = pd.DataFrame()

    pbar = tqdm(test_unseen_df.groupby('species'),
                total=len(test_unseen_df['species'].unique()))
    for _, group in pbar:
        pbar.set_description("Processing test unseen species: ")
        test_unseen_queries, test_unseen_keys = split_val_and_test_unseen(group)
        test_unseen_queries_df = pd.concat([test_unseen_queries_df, test_unseen_queries])
        test_unseen_keys_df = pd.concat([test_unseen_keys_df, test_unseen_keys])

    seen_train_df['split'] = 'seen_train'
    seen_val_queries_df['split'] = 'seen_val_queries'
    seen_test_queries_df['split'] = 'seen_test_queries'
    seen_keys_df['split'] = 'seen_keys'
    val_unseen_queries_df['split'] = 'val_unseen_queries'
    val_unseen_keys_df['split'] = 'val_unseen_keys'
    test_unseen_queries_df['split'] = 'test_unseen_queries'
    test_unseen_keys_df['split'] = 'test_unseen_keys'
    metadata_df_single_species['split'] = 'single_species'
    split_metadata_df = pd.concat([metadata_df_pre_train, seen_train_df, seen_val_queries_df, seen_test_queries_df, seen_keys_df,
                                     val_unseen_queries_df, val_unseen_keys_df, test_unseen_queries_df, test_unseen_keys_df,
                                     metadata_df_single_species])

    df_final = split_metadata_df.fillna('no_data', inplace=True)

    split_metadata_path = os.path.dirname(args.metadata)
    split_metadata = os.path.splitext(os.path.basename(args.metadata))[0] + '_split'
    df_final.to_csv(os.path.join(split_metadata_path, split_metadata), sep='\t', index=False)

