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

import argparse
from decimal import Decimal
import numpy as np
import pandas as pd

TAIL_THRESHOLD = 10
def filter_no_species(metadata: pd.DataFrame):
    return metadata[metadata["species"] != "not_classified"]


def get_tail_species(species_metadata: pd.DataFrame, threshold: int = TAIL_THRESHOLD):
    species = species_metadata.groupby("species")
    species_count = species.size()
    mask = species_count < threshold
    return species_count.index[mask]


def create_split_boundaries(size: int, split_ratios: list[float]) -> list[int]:
    assert sum(split_ratios) == 1
    split_sizes = list(map(lambda x: int(x * size), split_ratios))
    boundaries = []
    for ss in split_sizes[:-1]:
        if len(boundaries) == 0:
            boundaries.append(ss)
        else:
            boundaries.append(ss + boundaries[-1])
    return boundaries


def split_species(metadata: pd.DataFrame, split_ratios: float | list[float] = 0.8, seed=None):
    if isinstance(split_ratios, float):
        split_ratios = [split_ratios, 1 - split_ratios]
    assert sum(split_ratios) == 1
    all_species = pd.unique(metadata["species"])
    rand_gen = np.random.default_rng(seed=seed)
    split_sizes = create_split_boundaries(len(all_species), split_ratios)
    species_splits = np.split(rand_gen.permutation(all_species), split_sizes)
    return [metadata[metadata["species"].isin(species)] for species in species_splits]


def split_samples_per_species(metadata, split_ratios: list[float] | float, seed=None):
    if isinstance(split_ratios, float):
        split_ratios = [split_ratios, 1 - split_ratios]
    split_ratios = [Decimal(str(f)) for f in split_ratios]
    assert sum(split_ratios) == 1
    metadata = metadata.reset_index()
    all_species = pd.unique(metadata["species"])
    split_assignments = [[] for _ in range(len(split_ratios))]
    rand_gen = np.random.default_rng(seed=seed)
    for species in all_species:
        # generate random splits
        sample_indices = metadata[metadata["species"] == species].index.to_numpy()
        split_sizes = create_split_boundaries(sample_indices.shape[0], split_ratios)

        sample_splits = np.split(rand_gen.permutation(sample_indices), split_sizes)

        # collect split assignments in container
        for split_idx, indices in enumerate(sample_splits):
            split_assignments[split_idx].append(indices)
    return [metadata.loc[np.concatenate(indices)].set_index("index") for indices in split_assignments]


def assert_no_overlap(source: np.ndarray, targets: list[np.ndarray], assume_unique=True):
    for target in targets:
        intersection = np.intersect1d(source, target, assume_unique=assume_unique)
        if len(intersection) > 0:
            raise ValueError("Found overlap in splits.")


def create_final_metadata(metadata, **kwargs):
    split_metadata = metadata[["sampleid", "uri", "image_file", "species"]].copy()
    split_metadata["split"] = "no_split"
    for split_name, split in kwargs.items():
        split_metadata.loc[split_metadata["sampleid"].isin(split["sampleid"]), "split"] = split_name
    return split_metadata


def main(args):
    metadata = pd.read_csv(args.metadata, sep="\t")
    print("Creating splits...")
    # columns:
    #   ids: sampleid, processid, uri, name (same as order), image_file, chunk_number
    #   taxonomy: phylum, class, order, family, subfamily, tribe, genus, species, subspecies
    #   barcode: nucraw
    #   splits: {large,medium,small}_diptera_family, {large,medium,small}_insect_order

    # remove samples which do not have any species
    species_metadata = filter_no_species(metadata)

    # get series of species which have few samples
    tail_species = get_tail_species(species_metadata, threshold=args.min_species_size)
    tail_metadata = species_metadata[species_metadata["species"].isin(tail_species)]
    common_metadata = species_metadata[~species_metadata["species"].isin(tail_species)]

    # split seen species for train, val, and test
    seen_species, unseen_species = split_species(common_metadata, args.split_ratios_species, seed=args.seed)
    train_seen, val_seen, test_seen, seen_query = split_samples_per_species(seen_species, args.split_ratios_seen, seed=args.seed)

    # split unseen species between val and test
    val_unseen, test_unseen = split_species(unseen_species, args.percent_unseen_val, seed=args.seed)

    # Further split val_unseen and test_unseen so we get the query dataset.
    val_unseen, val_unseen_query = split_samples_per_species(val_unseen, args.percent_unseen_val, seed=args.seed)
    test_unseen, test_unseen_query = split_samples_per_species(test_unseen, args.percent_unseen_val, seed=args.seed)

    # Filter out the species with less than 2 samples. Then split tail_metadata in to tail_metadata and tail_query_metadata, add them to test_unseen and test_unseen_query

    species_with_one_sample = get_tail_species(tail_metadata, threshold=2)
    single_species = tail_metadata[tail_metadata["species"].isin(species_with_one_sample)]
    tail_metadata = tail_metadata[~tail_metadata["species"].isin(species_with_one_sample)]
    tail_metadata_to_val_unseen, tail_metadata_to_test_unseen = split_species(tail_metadata, 0.5, seed=args.seed)

    tail_metadata_to_val_unseen, tail_metadata_to_val_unseen_query = split_samples_per_species(tail_metadata_to_val_unseen, 0.5, seed=args.seed)
    val_unseen = pd.concat([val_unseen, tail_metadata_to_val_unseen])
    val_unseen_query = pd.concat([val_unseen_query, tail_metadata_to_val_unseen_query])
    tail_metadata_to_test_unseen, tail_metadata_to_test_unseen_query = split_samples_per_species(
        tail_metadata_to_test_unseen, 0.5, seed=args.seed)
    test_unseen = pd.concat([test_unseen, tail_metadata_to_test_unseen])
    test_unseen_query = pd.concat([test_unseen_query, tail_metadata_to_test_unseen_query])

    # validate results
    print("Validating splits...")
    train_seen_species = pd.unique(train_seen["species"])
    val_seen_species = pd.unique(val_seen["species"])
    test_seen_species = pd.unique(test_seen["species"])
    val_unseen_species = pd.unique(val_unseen["species"])
    test_unseen_species = pd.unique(test_unseen["species"])
    assert_no_overlap(
        val_unseen_species, [train_seen_species, val_seen_species, test_seen_species, test_unseen_species]
    )
    assert_no_overlap(test_unseen_species, [train_seen_species, val_seen_species, test_seen_species])
    assert_no_overlap(train_seen["sampleid"], [val_seen["sampleid"], test_seen["sampleid"]])
    assert_no_overlap(val_seen["sampleid"], [test_seen["sampleid"]])

    # final splits
    split_metadata = create_final_metadata(
        metadata,
        train_seen=train_seen,
        val_seen=val_seen,
        val_unseen=val_unseen,
        test_seen=test_seen,
        test_unseen=test_unseen,
        query_seen=seen_query,
        val_query_unseen=val_unseen_query,
        test_query_unseen=test_unseen_query,
        single_species=single_species
    )
    split_metadata.to_csv(args.output, sep="\t")
    print(split_metadata['split'].value_counts())


    return split_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", dest="metadata", type=str, help="path to dataset metadata file", default='data/BioScan_1M/BioScan_Insect_Dataset_metadata_2.tsv')
    parser.add_argument(
        "-s",
        "--min-species-size",
        dest="min_species_size",
        type=int,
        help="minimum number of samples in species to be considered for training",
        default=TAIL_THRESHOLD,
    )
    parser.add_argument(
        "-r",
        "--seen-ratio",
        dest="split_ratios_species",
        type=float,
        help="percentage of species to consider seen",
        default=0.8,
    )
    parser.add_argument(
        "-e",
        "--seen-splits",
        dest="split_ratios_seen",
        type=float,
        nargs=3,
        help="ratio of seen species split between train, val, test, and query",
        default=[0.7, 0.1, 0.1, 0.1],
    )
    parser.add_argument(
        "-u",
        "--unseen-splits",
        dest="percent_unseen_val",
        type=float,
        help="percent of unseen species to use in val, test , and query",
        default=0.5
    )
    parser.add_argument("-x", "--seed", dest="seed", type=int, help="random seed", default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, help="path to output TSV file", default='data/BioScan_1M/splits.tsv')
    args = parser.parse_args()

    main(args)
