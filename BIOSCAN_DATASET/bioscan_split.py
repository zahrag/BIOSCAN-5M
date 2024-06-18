#!/usr/bin/env python
# coding: utf-8

"""
BIOSCAN-5M Dataset Partitioning.

:Date: 2024-05-27
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

# # Setup
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display

tqdm.pandas()

taxon_cols = ["phylum", "class", "order", "family", "subfamily", "genus", "species"]

df_dtypes = {
    "processid": "str",
    "sampleid": "str",
    "uri_preconflictres": "category",
    "taxon_preconflictres": "category",
    "phylum_preconflictres": "category",
    "class_preconflictres": "category",
    "order_preconflictres": "category",
    "family_preconflictres": "category",
    "subfamily_preconflictres": "category",
    "genus_preconflictres": "category",
    "species_preconflictres": "category",
    "nucraw": str,
    "dna_barcode": str,
    "dna_bin": "category",
    "country": "category",
    "province/state": "category",
    "province_state": "category",
    "coord-lat": float,
    "coord-lon": float,
    "image_measurement_value": float,
    "image_measurement_context": "category",
    "area_fraction": float,
    "scale_factor": float,
    "image_file": str,
    "chunk_number": "uint8",
    "chunk": str,
    "index_bioscan_1M_insect": "Int64",
    "_phylum_original": "category",
    "_class_original": "category",
    "_order_original": "category",
    "_family_original": "category",
    "_subfamily_original": "category",
    "_genus_original": "category",
    "_species_original": "category",
    "_taxon_original": "category",
    "_uri_original": "category",
    "_nucraw_original": "category",
    "_phylum_fixtypos": "category",
    "_class_fixtypos": "category",
    "_order_fixtypos": "category",
    "_family_fixtypos": "category",
    "_subfamily_fixtypos": "category",
    "_genus_fixtypos": "category",
    "_species_fixtypos": "category",
    "_taxon_fixtypos": "category",
    "_rank": "Int64",
    "phylum_conflictres": "category",
    "class_conflictres": "category",
    "order_conflictres": "category",
    "family_conflictres": "category",
    "subfamily_conflictres": "category",
    "genus_conflictres": "category",
    "species_conflictres": "category",
    "taxon_conflictres": "category",
    "uri_conflictres": "category",
    "ratio": float,
    "conflicted": bool,
    "conflicted_uri": bool,
    "is_novel_species": "boolean",
    "inferred_ranks": "uint8",
    "label_was_reworded": "uint8",
    "label_was_manualeditted": "uint8",
    "label_was_inferred": "uint8",
    "label_was_dropped": "uint8",
    "split": "str",
    "taxon": "category",
    "phylum": "category",
    "class": "category",
    "order": "category",
    "family": "category",
    "subfamily": "category",
    "genus": "category",
    "species": "category",
}


def main(fname_input, output_csv, verbose=1):
    # ## Load
    if verbose >= 1:
        print(f"Loading {fname_input}")
    df = pd.read_csv(fname_input, dtype=df_dtypes)
    if verbose >= 1:
        print("Finished loading metadata.")

    df["dna_barcode_strip"] = df["dna_barcode"].str.strip("N")

    df

    df.columns

    df.dtypes

    df.memory_usage()

    sum(df.memory_usage()) / 1024 / 1024

    # # Partition

    # ```
    # placeholders = Maliaise0386 + ["sp.", "cf.", "nr.", "aff." , "n.sp.",  "sp. ex", "grp."]
    #
    # column: [partition/split]
    #
    # - pretrain    = everything without species labels + single_species[placeholders] + [other placeholders]
    #
    # - train       = (most samples of most species) + single_species[are proper names]
    #
    # - val         = small amount of all species in train (not single_species)
    #
    #                 (set of species = subset(train))
    #
    #                 stratfied 5%-10% of train data, same distrbn as train;
    #
    #                    exception species with fewer than 4 samples
    #
    # - test_seen   = small amount of all species in train [no placeholders]
    #
    #                 (set of species = subset(train) > num species in val)
    #
    #                 from species which have >8 samples, place half of num for species - 20
    #
    # - test_unseen = the species not in train [no placeholders]
    #
    #                 all samples from those species
    #
    #                 some species with up to 20 (say) samples, at least 2
    #
    # column: [role]
    #
    # - key / query / other=""=NA (/ more?)
    #
    #   do stratified 50/50 key/query where appropriate - apply to train, val, test_seen, test_unseen
    #
    # column: [species_status]
    #
    # - seen / unseen / placeholder / unknown
    #
    # ```
    df["split"] = "unk"
    df.loc[df["species"].isna() | df["is_novel_species"], "split"] = "pretrain"

    print(sum(df["split"] == "pretrain"))
    print(sum((df["split"] == "pretrain") & df["species"].notna()))
    print(df.loc[df["split"] == "pretrain", taxon_cols].nunique())

    catalogued_species = df.loc[df["species"].notna() & (df["split"] == "unk"), "species"].unique()

    catalogued_species = sorted(list(catalogued_species))

    catalogued_species

    len(catalogued_species)

    # ## Single sample species
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    partition_candidates

    sp_sz = partition_candidates.groupby("species", observed=True).size()

    sp_sz

    single_species = list(sp_sz[sp_sz == 1].index)

    len(single_species)

    df.loc[df["species"].isin(single_species), "split"] = "train"

    # ## test_unseen
    np.unique(df["split"], return_counts=True)

    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    sp_sz = partition_candidates.groupby("species", observed=True).size()

    sp_sz

    plt.hist(sp_sz, np.arange(100))

    plt.hist(sp_sz, np.arange(max(sp_sz)), cumulative=True)

    plt.hist(sp_sz, np.arange(200), cumulative=True)

    print("cut  at_cut  cumm  (%)     n_samp  (%)")
    print("-----------------------------------------")
    for cutoff in range(51):
        sel_sp = sp_sz <= cutoff
        n_sel_sp = sum(sel_sp)
        n_sel_samp = sum(partition_candidates['species'].isin(sp_sz[sel_sp].index))
        print(f"{cutoff:3d}  {sum(sp_sz == cutoff):5d}  {n_sel_sp:5d}  {100 * n_sel_sp / len(sel_sp):5.2f}%  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%")

    ge_sz = partition_candidates.groupby("genus", observed=True).size()

    ge_sz

    plt.hist(ge_sz, np.arange(200), cumulative=True)

    print("cut  n_genus (%)     n_samp  (%)")
    print("----------------------------------")
    for cutoff in range(51):
        sel_ge = ge_sz <= cutoff
        n_sel_ge = sum(sel_ge)
        n_sel_samp = sum(partition_candidates['genus'].isin(ge_sz[sel_ge].index))
        print(f"{cutoff:3d}  {n_sel_ge:5d}  {100 * n_sel_ge / len(ge_sz):5.2f}%  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%")

    nspecies_per_genus = partition_candidates[["genus", "species"]].drop_duplicates().groupby("genus", observed=True).size()
    nspecies_per_genus

    max(nspecies_per_genus)

    plt.hist(nspecies_per_genus, np.arange(40), cumulative=True)

    _sp_sz_trainval_dna = df.loc[df["split"].isin(["unk", "train", "val", "test_seen"])].drop_duplicates("dna_barcode_strip").groupby("species", observed=True, dropna=False).size()

    len(_sp_sz_trainval_dna)

    max(_sp_sz_trainval_dna)

    plt.hist(_sp_sz_trainval_dna, np.arange(40))
    plt.show()

    plt.hist(_sp_sz_trainval_dna, np.arange(40), cumulative=True)
    plt.show()

    print("cut  at_cut  cumm  (%)     n_samp  (%)")
    print("-----------------------------------------")
    for cutoff in range(51):
        sel_sp = _sp_sz_trainval_dna <= cutoff
        n_sel_sp = sum(sel_sp)
        n_sel_samp = sum(partition_candidates['species'].isin(_sp_sz_trainval_dna[sel_sp].index))
        print(f"{cutoff:3d}  {sum(_sp_sz_trainval_dna == cutoff):5d}  {n_sel_sp:5d}  {100 * n_sel_sp / len(sel_sp):5.2f}%  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%")

    # ### actual
    sel_known_genus = df["genus"].isin(df.loc[df["split"] != "pretrain", "genus"])

    sel_viable = sel_known_genus & (df["split"] == "pretrain") & df["species"].notna()
    sum(sel_viable)

    df.loc[sel_viable, taxon_cols].groupby(taxon_cols, dropna=False, observed=True).size()

    g_pt_species = df.loc[sel_viable, ["genus", "species"]].groupby("species", observed=True).size()
    g_pt_species

    transfer_species = g_pt_species[g_pt_species > 1].index
    print(len(transfer_species))
    print(sum(df["species"].isin(transfer_species)))

    transfer_species = g_pt_species[g_pt_species >= 4].index
    print(len(transfer_species))
    print(sum(df["species"].isin(transfer_species)))

    transfer_species = g_pt_species[g_pt_species >= 6].index
    print(len(transfer_species))
    print(sum(df["species"].isin(transfer_species)))

    transfer_species = g_pt_species[g_pt_species >= 8].index
    print(len(transfer_species))
    print(sum(df["species"].isin(transfer_species)))

    transfer_species = g_pt_species[g_pt_species >= 10].index
    print(len(transfer_species))
    print(sum(df["species"].isin(transfer_species)))

    g_pt_species[g_pt_species >= 8]

    df.loc[df["species"].isin(g_pt_species[g_pt_species >= 8].index), "split"] = "test_unseen"

    for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}%")

    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    sorted(df.loc[df["split"] == "test_unseen", "genus"].unique())

    df.loc[df["split"] == "pretrain", "species"].nunique()

    df.drop(columns=['label_was_reworded', 'label_was_manualeditted', 'label_was_inferred', 'label_was_dropped']).to_csv("BIOSCAN-5M_Dataset_v3.0rc1.csv")

    sum(df["dna_bin"].isna())

    for split in df["split"].unique():
        print(split, sum(df.loc[df["split"] == split, "dna_bin"].isna()))

    # ## test_seen
    df.loc[df["split"].isin(["test_seen", "train", "val"]), "split"] = "unk"
    df.loc[df["species"].isin(single_species), "split"] = "train"

    df["dna_barcode_strip"] = df["dna_barcode"].str.strip("N")

    for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}%")

    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "dna_barcode_strip", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    partition_candidates

    sp_sz = partition_candidates.groupby("species", observed=True).size()

    sp_sz

    sum(sp_sz >= 6)

    sum(sp_sz >= 8)

    sum(sp_sz >= 10)

    np.sum(np.minimum(20, np.floor(sp_sz[sp_sz >= 8] / 2)))

    np.sum(np.minimum(20, np.floor(sp_sz[sp_sz >= 10] / 2)))


    def test_split_fn(n):
        if n < 8:
            return 0
        k = int(np.floor(4 + (n - 8) / 4))
        k = min(k, 25)
        return k


    def test_split_fn_lb(n):
        if n < 8:
            return 0
        k = int(np.floor(3 + (n - 8) / 5))
        return k


    def test_split_fn_ub(n):
        if n < 8:
            return 0
        k = int(np.floor(4 + (n - 8) / 3))
        return k


    def test_split_fn_lb_barcodes(n):
        if n < 3:
            return 0
        k = int(np.floor(1 + (n - 2) / 5))
        k = min(k, 15)
        return k


    def test_split_fn_ub_barcodes(n):
        if n < 3:
            return 0
        k = int(np.floor(1 + (n - 2) / 3))
        return k


    for i in range(1, 100):
        print(f"{i:3d}  {test_split_fn(i):3d}  {100 * test_split_fn(i) / i:5.2f}%")

    for i in range(1, 100):
        print(f"{i:3d}  {test_split_fn_lb(i):3d}  {100 * test_split_fn_lb(i) / i:5.2f}%")

    for i in range(1, 100):
        print(f"{i:3d}  {test_split_fn_ub(i):3d}  {100 * test_split_fn_ub(i) / i:5.2f}%")

    for i in range(1, 100):
        print(f"{i:3d}  {test_split_fn_lb_barcodes(i):3d}  {100 * test_split_fn_lb_barcodes(i) / i:5.2f}%")

    for i in range(1, 100):
        print(f"{i:3d}  {test_split_fn_ub_barcodes(i):3d}  {100 * test_split_fn_ub_barcodes(i) / i:5.2f}%")

    sp_sztest = sp_sz.apply(test_split_fn)
    sp_sztest

    print(sum(sp_sztest > 0), "species")
    print(sum(sp_sztest), "samples")

    df_tmp = pd.concat([sp_sz, sp_sztest], axis=1, keys=["total", "test"])

    df_tmp

    df_tmp["train"] = df_tmp["total"] - df_tmp["test"]

    df_tmp["test_pc"] = 100 * df_tmp["test"] / df_tmp["total"]
    df_tmp["train_pc"] = 100 * df_tmp["train"] / df_tmp["total"]

    df_tmp

    df_tmp["test_pc"]

    max(df_tmp["total"])

    100 * 24 / max(df_tmp["total"])

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "dna_barcode_strip", "genus", "species"]]
    partition_candidates = partition_candidates.sample(frac=1, random_state=0)

    partition_candidates

    for species, k in sp_sztest.items():
        print(species, k)
        break

    sp_sztest["Aphis varians"]

    g_test = partition_candidates.groupby("species", observed=True)

    for species, grp in g_test:
        if sp_sztest[species] > 0:
            break

    species

    grp

    sp_sztest[species]

    sp_sz[species]

    len(grp)

    grp.groupby("dna_barcode_strip")

    dnasz = grp.groupby("dna_barcode_strip").size().sort_values()

    dnasz

    dnasz_cs = dnasz.sort_values().cumsum()

    n = len(grp)
    target = test_split_fn(n)
    target_lb = 3
    target_ub = test_split_fn_ub(n)

    options = dnasz[(dnasz <= target) & (dnasz_cs >= target)]

    rng = np.random.default_rng(seed=1)

    idx = rng.integers(len(options), size=1)[0]
    idx

    options.index[idx]


    def stratified_dna_image_partition(g_test, target_fn, lower_fn, upper_fn, dna_upper_fn, soft_upper=1.1, center_rand=False, top_rand=False, seed=None):
        rng = np.random.default_rng(seed=seed)
        barcodes_selected = []
        for species, grp in tqdm(g_test):
            verbose = False
            if False:
                verbose = True
                print()
                print(species)
                display(grp)
            n = len(grp)
            target = target_fn(n)
            if target == 0:
                continue
            lb_samp = lower_fn(n)
            ub_samp = upper_fn(n)
            n_alloc = 0
            indices_used = []
            grp_barcodes_selected = []
            dnasz = grp.groupby("dna_barcode_strip").size().sort_values()
            if verbose:
                display(dnasz)
            n_barcodes = len(dnasz)
            # lb_barcodes = dna_lower_fn(n_barcodes)
            ub_barcodes = dna_upper_fn(n_barcodes)
            if verbose:
                print(n, "samples for the species")
                print(n_barcodes, "barcodes for the species")
                print("target", target)
                print("lb_samp", lb_samp)
                print("ub_samp", ub_samp)
                print("ub_barcodes", ub_barcodes)
            if n_barcodes == 1:
                # Can't allocate our only DNA barcode
                continue
            dnasz_cs = dnasz.cumsum()
            if verbose:
                pass
                # break
            while True:
                if verbose:
                    print("n_alloc", n_alloc)
                    # display(dnasz)
                # We only want to add samples which keep us below or at the target
                # And cumsum indicates the largest value we can reach with the smallest k entries,
                # so we need to add one of the samples that takes the cumsum to the target.
                is_option = (dnasz <= target - n_alloc).values & (dnasz_cs >= target * soft_upper - n_alloc).values
                if sum(is_option) == 0:
                    if verbose:
                        print("No ideal options")
                    # No ideal options, so let's make do
                    if n_alloc == 0 and not any(dnasz <= target):
                        # No options and nothing added so far
                        if dnasz.iloc[0] >= lb_samp and dnasz.iloc[0] < n / 2:
                            if verbose:
                                print("Nothing added, taking first")
                            idx = 0
                        else:
                            break
                    elif dnasz.iloc[0] > target - n_alloc:  # not any(dnasz <= target - n_alloc):
                        # Everything would take us above the remaining target
                        if dnasz.iloc[0] > (target - n_alloc) / 2:
                            # Shouldn't add the smallest because the residual would be larger than it is now
                            if verbose:
                                print("Last possibility avoiding smallest, increases residual")
                            break
                        if n_alloc + dnasz.iloc[0] > ub_samp:
                            # Can't add the smallest as it takes us above our upperbound
                            if verbose:
                                print("Can't add smallest due to upperbound")
                            break
                        # Add the smallest
                        idx = 0
                    elif not any(dnasz_cs >= target * soft_upper - n_alloc):
                        raise ValueError("Impossible outcome")
                    else:
                        # Find where our two criteria meet
                        option_cutoff_idx = np.nonzero(dnasz_cs >= target * soft_upper - n_alloc)[0][0]
                        idx = option_cutoff_idx
                        if (
                            option_cutoff_idx > 0
                            and abs(n_alloc + dnasz.iloc[option_cutoff_idx - 1] - target) < abs(target - (n_alloc + dnasz.iloc[option_cutoff_idx]))
                        ):
                            idx = option_cutoff_idx - 1
                        if n_alloc + dnasz.iloc[option_cutoff_idx] > ub_samp:
                            break
                        if verbose:
                            print("Breaking between criteria")
                elif len(grp_barcodes_selected) + 1 >= ub_barcodes:
                    # We can only add one more, so add the most populated DNA that makes sense to add
                    option_cutoff_idx = np.nonzero(dnasz <= target - n_alloc)[0][-1]
                    idx = option_cutoff_idx
                    if (
                        option_cutoff_idx < len(dnasz) - 1
                        and abs(n_alloc + dnasz.iloc[option_cutoff_idx + 1] - target) < abs(target - (n_alloc + dnasz.iloc[option_cutoff_idx]))
                        and n_alloc + dnasz.iloc[option_cutoff_idx + 1] <= ub_samp
                    ):
                        idx = option_cutoff_idx + 1
                    if verbose:
                        print("Choosing last barcode to add, so using largest")
                else:
                    n_options = sum(is_option)
                    min_idx = 0
                    max_idx = n_options
                    if center_rand:
                        # Only use a barcode from the intermediate set of options
                        min_idx = max(min_idx, int(n_options * 0.25))
                        max_idx = int(np.ceil(n_options * 0.75))
                    elif top_rand:
                        # Only use a barcode from the largest options
                        min_idx = max(min_idx, int(n_options * 0.5))
                    else:
                        idx = rng.integers(n_options, size=1)[0]
                    if max_idx <= min_idx:
                        min_idx = max(min_idx - 1, 0)
                        max_idx = min(max_idx + 1, n_options)
                    idx = rng.integers(min_idx, max_idx, size=1)[0]
                    idx = np.nonzero(is_option)[0][idx]
                    if verbose:
                        print(f"Randomly selected item #{idx} from {n_options} options with {dnasz.iloc[idx]} samp")
                if verbose:
                    print(f"Adding item {idx}: {dnasz.index[idx]}")
                grp_barcodes_selected.append(dnasz.index[idx])
                n_alloc += dnasz.iloc[idx]
                if n_alloc >= target:
                    break
                dnasz_cs[idx:] -= dnasz.iloc[idx]
                indices_used.append(idx)
                len_pre = len(dnasz)
                dnasz_cs.drop(dnasz.index[idx], inplace=True)
                dnasz.drop(dnasz.index[idx], inplace=True)
                len_post = len(dnasz)
                if len_pre == len_post:
                    raise ValueError()
                if len(grp_barcodes_selected) >= ub_barcodes:
                    break
                if verbose:
                    print(f"End of loop with {len(dnasz)} = {len(dnasz_cs)} / {n_barcodes} barcodes remaining")
            if n_alloc >= lb_samp:
                barcodes_selected.append(grp_barcodes_selected)
                if verbose:
                    print(f"Added {len(grp_barcodes_selected)} barcodes")
            elif verbose:
                print(f"Skipped adding {len(grp_barcodes_selected)} barcodes")
            if verbose:
                print(len(grp_barcodes_selected), "barcodes selected")
                display(grp_barcodes_selected)
        return barcodes_selected


    barcodes_selected = stratified_dna_image_partition(g_test, test_split_fn, lambda x: 3, test_split_fn_ub, test_split_fn_ub_barcodes, top_rand=True, seed=1)

    print("target", target)
    # print("n_alloc", n_alloc)
    print("dnasz")
    print(dnasz.values)
    print("dnasz_cs")
    print(dnasz_cs.values)

    print(len(barcodes_selected))

    barcodes_selected = [b for bs in barcodes_selected for b in bs]

    len(barcodes_selected)

    sel = df["dna_barcode_strip"].isin(barcodes_selected)

    df.loc[sel, "species"].nunique()

    df.loc[df["split"] == "test_seen", "split"] = "unk"
    df.loc[sel, "split"] = "test_seen"

    _sp_sz_trainval = df.loc[df["split"].isin(["unk", "train", "val"])].groupby("species", observed=True, dropna=False).size()

    _sp_sz_test = df.loc[df["split"] == "test_seen"].groupby("species", observed=True, dropna=False).size()

    print(_sp_sz_test.sum())

    print(_sp_sz_trainval.sum())

    print(100 * _sp_sz_test.sum() / _sp_sz_trainval.sum())

    _sp_sz_trainval

    _sp_sz_test

    df_tmp2 = pd.concat([_sp_sz_trainval, _sp_sz_test], axis=1, keys=["trainval", "test"])
    df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
    df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
    df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]

    df_tmp2

    plt.scatter(df_tmp2["total"], df_tmp2["test"])

    plt.scatter(df_tmp2["total"], df_tmp2["test"])
    plt.xscale("log")

    plt.scatter(df_tmp2["total"], df_tmp2["test"])
    plt.xscale("log")
    plt.yscale("log")

    df_tmp2.index[df_tmp2["test"] > 200]

    df_tmp2.loc[df_tmp2["test"] > 0, "test"].min()

    df_tmp2.loc[df_tmp2["test"] == 1].index

    df_tmp2["test_pc"].sort_values()

    _sp_sz_trainval_dna = df.loc[df["split"].isin(["unk", "train", "val"])].drop_duplicates("dna_barcode_strip").groupby("species", observed=True, dropna=False).size()

    _sp_sz_test_dna = df.loc[df["split"] == "test_seen"].drop_duplicates("dna_barcode_strip").groupby("species", observed=True, dropna=False).size()

    df_tmp2 = pd.concat([_sp_sz_trainval_dna, _sp_sz_test_dna], axis=1, keys=["trainval", "test"])
    df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
    df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
    df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]

    df_tmp2

    plt.scatter(df_tmp2["total"], df_tmp2["test"])

    plt.scatter(df_tmp2["total"], df_tmp2["test"])
    plt.xscale("log")

    plt.scatter(df_tmp2["total"], df_tmp2["test_pc"])
    plt.xscale("log")

    # Without restricting the random selection:
    # ```
    # pretrain        4754367 92.30%
    # test_seen         39119  0.76%
    # test_unseen       53171  1.03%
    # train              3757  0.07%
    # unk              300466  5.83%
    # pretrain        2325492 93.52%
    # test_seen         22293  0.90%
    # test_unseen       18009  0.72%
    # train              3757  0.15%
    # unk              116955  4.70%
    # samples to place 300466
    # species to place 8090
    # ```
    #
    # Restricting to top 50%
    # ```
    # pretrain        4754367 92.30% 1384.73%
    # test_seen         39353  0.76%   11.46%
    # test_unseen       53171  1.03%   15.49%
    # train              3757  0.07%    1.09%
    # unk              300232  5.83%   87.44%
    #
    # pretrain        2325492 93.52% 1626.16%
    # test_seen         18966  0.76%   13.26%
    # test_unseen       18009  0.72%   12.59%
    # train              3757  0.15%    2.63%
    # unk              120282  4.84%   84.11%
    #
    # samples to place 300232
    # species to place 8090
    # ```
    n_samp_trainvaltest = sum(df["split"].isin(["train", "val", "test_seen", "unk"]))
    n_dna_trainvaltest = df.loc[df["split"].isin(["train", "val", "test_seen", "unk"]), "dna_barcode"].nunique()

    for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%")

    print()
    n_samp_trainvaltest = df["split"].isin(["train", "val", "test_seen", "unk"])
    n_barcodes = df["dna_barcode"].nunique()
    for partition, n_samp in zip(*np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / n_barcodes:5.2f}% {100 * n_samp / n_dna_trainvaltest:7.2f}%")

    print()
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    100 * 39119 / (299_584 + 40_001 + 3_757)

    100 * 22293 / (22293 + 116955 + 3757)

    100 * 40_001 / (299_584 + 40_001 + 3_757)

    df.drop(columns=['label_was_reworded', 'label_was_manualeditted', 'label_was_inferred', 'label_was_dropped']).to_csv("BIOSCAN-5M_Dataset_v3.0rc4.csv")

    # ## val
    (301301 + 38284 + 3_757) * 5 / 100

    (116955 + 3757 + 22293) * 5 / 100

    (299_584 + 40_001 + 3_757) * 7.5 / 100

    (299_584 + 3_757) * 5 / 100

    (299_584 + 3_757) * 7.5 / 100

    target_nval = 17_167

    partition_candidates

    sel = (sp_sz < 6) | ((8 <= sp_sz) & (sp_sz < 13))
    sp_to_not_take = sp_sz[sel].index

    sp_to_not_take

    len(sp_sz)

    sp_sz_val = partition_candidates.groupby("species", observed=True).size()

    sp_sz_val

    sp_sz_val[sp_sz_val >= 20]

    sp_sz_val[sp_sz_val >= 20].index

    sp_sz_val

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "dna_barcode_strip", "genus", "species"]]
    partition_candidates = partition_candidates.sample(frac=1, random_state=2)

    partition_candidates

    sp_sz_val[sp_sz_val >= 20].index

    g_test = partition_candidates.groupby("species", observed=True)

    val_pc_target = 5

    # barcodes_selected = stratified_dna_image_partition(g_test, lambda x: val_pc_target / 100 * x, lambda x: 0, lambda x: val_pc_target / 50 * x, lambda x: val_pc_target / 100 * x, top_rand=True, seed=3)

    rng = np.random.default_rng(seed=3)
    barcodes_selected = []
    soft_upper = 1.1
    for species, grp in tqdm(g_test):
        verbose = False
        n = len(grp)
        if n < 20:
            continue
        target = val_pc_target / 100 * n
        if target == 0:
            continue
        target_lb = 0
        target_ub = 2 * target
        n_alloc = 0
        indices_used = []
        grp_barcodes_selected = []
        dnasz = grp.groupby("dna_barcode_strip").size().sort_values()
        if verbose:
            display(dnasz)
        n_barcodes = len(dnasz)
        ub_barcodes = val_pc_target / 100 * n_barcodes
        if verbose:
            print(n, "samples for the species")
            print(n_barcodes, "barcodes for the species")
            print("target", target)
            print("target_lb", target_lb)
            print("target_ub", target_ub)
            print("ub_barcodes", ub_barcodes)
        if n_barcodes == 1:
            # Can't allocate our only DNA barcode
            continue
        dnasz_cs = dnasz.cumsum()
        if verbose:
            pass
            # break
        while True:
            if verbose:
                print("n_alloc", n_alloc)
                # display(dnasz)
            # We only want to add samples which keep us below or at the target
            # And cumsum indicates the largest value we can reach with the smallest k entries,
            # so we need to add one of the samples that takes the cumsum to the target.
            is_option = (dnasz <= target - n_alloc).values & (dnasz_cs >= target * soft_upper - n_alloc).values
            if sum(is_option) == 0:
                if verbose:
                    print("No ideal options")
                # No ideal options, so let's make do
                if n_alloc == 0 and not any(dnasz <= target):
                    # No options and nothing added so far
                    if dnasz.iloc[0] >= target_lb and dnasz.iloc[0] < n / 2:
                        if verbose:
                            print("Nothing added, taking first")
                        idx = 0
                    else:
                        break
                elif dnasz.iloc[0] > target * soft_upper - n_alloc:  # not any(dnasz <= target - n_alloc):
                    # Everything would take us above the remaining target
                    if dnasz.iloc[0] > (target - n_alloc) / 2:
                        # Shouldn't add the smallest because the residual would be larger than it is now
                        if verbose:
                            print("Last possibility avoiding smallest, increases residual")
                        break
                    if n_alloc + dnasz.iloc[0] > target_ub:
                        # Can't add the smallest as it takes us above our upperbound
                        if verbose:
                            print("Can't add smallest due to upperbound")
                        break
                    # Add the smallest
                    idx = 0
                elif not any(dnasz_cs >= target - n_alloc):
                    raise ValueError("Impossible outcome")
                else:
                    # Find where our two criteria meet
                    option_cutoff_idx = np.nonzero(dnasz_cs >= target * soft_upper - n_alloc)[0][0]
                    idx = option_cutoff_idx
                    if (
                        option_cutoff_idx > 0
                        and abs(n_alloc + dnasz.iloc[option_cutoff_idx - 1] - target) < abs(target - (n_alloc + dnasz.iloc[option_cutoff_idx]))
                    ):
                        idx = option_cutoff_idx - 1
                    if n_alloc + dnasz.iloc[option_cutoff_idx] > target_ub:
                        break
                    if verbose:
                        print("Breaking between criteria")
            elif len(grp_barcodes_selected) + 1 >= ub_barcodes:
                # We can only add one more, so add the most populated DNA that makes sense to add
                option_cutoff_idx = np.nonzero(dnasz <= target - n_alloc)[0][-1]
                idx = option_cutoff_idx
                if (
                    option_cutoff_idx < len(dnasz) - 1
                    and abs(n_alloc + dnasz.iloc[option_cutoff_idx + 1] - target) < abs(target - (n_alloc + dnasz.iloc[option_cutoff_idx]))
                    and n_alloc + dnasz.iloc[option_cutoff_idx + 1] <= target_ub
                ):
                    idx = option_cutoff_idx + 1
                if verbose:
                    print("Choosing last barcode to add, so using largest")
            else:
                # idx = rng.integers(sum(is_option), size=1)[0]
                # Only use a barcode from the larger X of what is possible
                idx = rng.integers(int(sum(is_option) * 0.5), sum(is_option), size=1)[0]
                idx = np.nonzero(is_option)[0][idx]
                if verbose:
                    print(f"Randomly selected item #{idx} from {sum(is_option)} options with {dnasz.iloc[idx]} samp")
            if verbose:
                print(f"Adding item {idx}: {dnasz.index[idx]}")
            grp_barcodes_selected.append(dnasz.index[idx])
            n_alloc += dnasz.iloc[idx]
            if n_alloc >= target:
                break
            dnasz_cs[idx:] -= dnasz.iloc[idx]
            indices_used.append(idx)
            len_pre = len(dnasz)
            dnasz_cs.drop(dnasz.index[idx], inplace=True)
            dnasz.drop(dnasz.index[idx], inplace=True)
            len_post = len(dnasz)
            if len_pre == len_post:
                raise ValueError()
            if len(grp_barcodes_selected) >= ub_barcodes:
                break
            if verbose:
                print(f"End of loop with {len(dnasz)} = {len(dnasz_cs)} / {n_barcodes} barcodes remaining")
        if n_alloc >= target_lb:
            barcodes_selected.append(grp_barcodes_selected)
            if verbose:
                print(f"Added {len(grp_barcodes_selected)} barcodes")
        elif verbose:
            print(f"Skipped adding {len(grp_barcodes_selected)} barcodes")
        if verbose:
            print(len(grp_barcodes_selected), "barcodes selected")
            display(grp_barcodes_selected)

    print(len(barcodes_selected))

    barcodes_selected = [b for bs in barcodes_selected for b in bs]

    len(barcodes_selected)

    sel = df["dna_barcode_strip"].isin(barcodes_selected)

    sum(sel)

    df.loc[sel, "species"].nunique()

    # ```
    # Target Current
    # --------------
    # 15211.2 12869
    #  6032.7  4798
    #  3757.0  1618
    # ```
    #
    # ```
    # Target Current
    # --------------
    # 15199.5 12786
    #  6199.2  4644
    #  3757.0  1615
    #
    # ```
    target_samp = val_pc_target / 100 * sum(df["split"].isin(["train", "unk"]))
    target_dna = val_pc_target / 100 * df.loc[df["split"].isin(["train", "unk"]), "dna_barcode_strip"].nunique()
    target_spec = df.loc[df["split"].isin(["train"]), "species"].nunique()

    print("Target Current")
    print("--------------")
    print(f"{target_samp:7.1f} {sum(sel):5d}")
    print(f"{target_dna:7.1f} {len(barcodes_selected):5d}")
    print(f"{target_spec:7.1f} {df.loc[sel, 'species'].nunique():5d}")

    g_val_sz = g_test.size()
    g_val_sz = g_val_sz[g_val_sz < 20]

    small_spec_target = val_pc_target / 100 * (len(single_species) + sum(g_val_sz))
    small_spec_target = round(small_spec_target)
    print(small_spec_target)

    # rng = np.random.default_rng(seed=4)
    df_test = df.loc[df["split"] == "test_seen"]
    n_alloc_total = 0
    barcodes_selected2 = []
    tally_skip_size = 0
    tally_skip_test = 0
    tally_skip_dna = 0
    for species, grp in tqdm(g_test):
        verbose = False
        n = len(grp)
        if n >= 20 or n < 7:
            tally_skip_size += 1
            continue
        n_test = sum(df_test["species"] == species)
        target_test = test_split_fn(n + n_test)
        if n_test > target_test:
            tally_skip_test += 1
            continue
        dnasz = grp.groupby("dna_barcode_strip").size().sort_values()
        if dnasz.iloc[0] > 1:
            tally_skip_dna += 1
            continue
        barcodes_selected2.append(dnasz.index[0])
        n_alloc_total += dnasz.iloc[0]
        if n_alloc_total >= small_spec_target:
            break

    print("tally_skip_size", tally_skip_size)
    print("tally_skip_test", tally_skip_test)
    print("tally_skip_dna", tally_skip_dna)

    print(n_alloc_total, len(barcodes_selected2), small_spec_target)

    sel = df["dna_barcode_strip"].isin(barcodes_selected + barcodes_selected2)

    target_samp = val_pc_target / 100 * sum(df["split"].isin(["train", "unk"]))
    target_dna = val_pc_target / 100 * df.loc[df["split"].isin(["train", "unk"]), "dna_barcode_strip"].nunique()
    target_spec = df.loc[df["split"].isin(["train"]), "species"].nunique()

    print("Target Current")
    print("--------------")
    print(f"{target_samp:7.1f} {sum(sel):5d}")
    print(f"{target_dna:7.1f} {len(barcodes_selected) + len(barcodes_selected2):5d}")
    print(f"{target_spec:7.1f} {df.loc[sel, 'species'].nunique():5d}")

    df.loc[sel, "split"] = "val"

    n_samp_trainvaltest = sum(df["split"].isin(["train", "val", "test_seen", "unk"]))
    n_dna_trainvaltest = df.loc[df["split"].isin(["train", "val", "test_seen", "unk"]), "dna_barcode"].nunique()

    for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%")

    print()
    n_samp_trainvaltest = df["split"].isin(["train", "val", "test_seen", "unk"])
    n_barcodes = df["dna_barcode"].nunique()
    for partition, n_samp in zip(*np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / n_barcodes:5.2f}% {100 * n_samp / n_dna_trainvaltest:7.2f}%")

    print()
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    # ## train
    df.loc[df["split"] == "unk", "split"] = "train"

    n_samp_trainvaltest = sum(df["split"].isin(["train", "val", "test_seen", "unk"]))
    n_dna_trainvaltest = df.loc[df["split"].isin(["train", "val", "test_seen", "unk"]), "dna_barcode"].nunique()

    for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%")

    print()
    n_samp_trainvaltest = df["split"].isin(["train", "val", "test_seen", "unk"])
    n_barcodes = df["dna_barcode"].nunique()
    for partition, n_samp in zip(*np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)):
        print(f"{partition:15s} {n_samp:7d} {100 * n_samp / n_barcodes:5.2f}% {100 * n_samp / n_dna_trainvaltest:7.2f}%")

    print()
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "unk"), ["processid", "genus", "species"]]
    print("samples to place", len(partition_candidates))
    print("species to place", partition_candidates["species"].nunique())

    # # Extras

    # ## species status
    df["species_status"] = "unknown"

    df.loc[df["split"].isin(["train", "val", "test_seen"]), "species_status"] = "seen"

    df.loc[df["is_novel_species"].notna() & df["is_novel_species"], "species_status"] = "pretrain_only"

    df.loc[df["split"].isin(["test_unseen"]), "species_status"] = "unseen"

    df.loc[df["species"].isin(single_species), "species_status"] = "seen_singleton"

    df

    df.drop(columns=['label_was_reworded', 'label_was_manualeditted', 'label_was_inferred', 'label_was_dropped']).to_csv("BIOSCAN-5M_Dataset_v3.0rc7.csv")

    # ## Role
    df["role"] = "none"

    df.loc[df["split"] == "train", "role"] = "key"
    df.loc[df["split"] == "val", "role"] = "query"
    df.loc[df["split"] == "test_seen", "role"] = "query"

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[df["species"].notna() & (df["split"] == "test_unseen"), ["processid", "dna_barcode_strip", "genus", "species"]]
    partition_candidates = partition_candidates.sample(frac=1, random_state=4)

    g_test = partition_candidates.groupby("species", observed=True)

    print("OBSOLETE")

    val_pc_target = 50
    rng = np.random.default_rng(seed=4)
    barcodes_selected = []
    soft_upper = 1.1

    all_kq_barcodes = []
    all_kq_images = []

    for species, grp in tqdm(g_test):
        kq_barcodes = [[], []]
        kq_images = [[], []]
        kq_imcount = [0, 0]

        dnasz = grp.groupby("dna_barcode_strip").size().sort_values(ascending=False)
        if len(dnasz) == 1:
            cutoff = len(grp) // 2
            kq_images = [grp.iloc[:cutoff]["processid"].values, grp.iloc[cutoff:]["processid"].values]
            all_kq_images.append(kq_images)
            print(f"Couldn't split barcodes for {species} as there's only one barcode")
            continue

        for barcode, tally in dnasz.items():
            if len(kq_barcodes[0]) < len(kq_barcodes[1]):
                add_idx = 0
            elif len(kq_barcodes[0]) > len(kq_barcodes[1]):
                add_idx = 1
            elif kq_imcount[0] < kq_imcount[1]:
                add_idx = 0
            elif kq_imcount[0] > kq_imcount[1]:
                add_idx = 1
            else:
                add_idx = rng.integers(0, 2, size=1)[0]
            kq_barcodes[add_idx].append(barcode)
            kq_imcount[add_idx] += tally

        if kq_imcount[0] / kq_imcount[1] < 3 and kq_imcount[1] / kq_imcount[0] < 3:
            all_kq_barcodes.append(kq_barcodes)
            continue

        print(f"Had too much imbalance for {species}: {kq_imcount}")
        kq_barcodes = [[], []]
        for i, (barcode, tally) in enumerate(dnasz.items()):
            if i == 0:
                barcode_to_split = barcode
                continue
            if len(kq_barcodes[0]) < len(kq_barcodes[1]):
                add_idx = 0
            elif len(kq_barcodes[0]) > len(kq_barcodes[1]):
                add_idx = 1
            elif kq_imcount[0] < kq_imcount[1]:
                add_idx = 0
            elif kq_imcount[0] > kq_imcount[1]:
                add_idx = 1
            else:
                add_idx = rng.integers(0, 2, size=1)[0]
            kq_barcodes[add_idx].append(barcode)
            kq_imcount[add_idx] += tally

        images_to_split = grp.loc[grp["dna_barcode_strip"] == barcode_to_split, "processid"]

        delta = abs(kq_imcount[0] - kq_imcount[1])
        if delta >= len(images_to_split) - len(grp) * 0.25:
            if kq_imcount[0] < kq_imcount[1]:
                add_idx = 0
            elif kq_imcount[0] > kq_imcount[1]:
                add_idx = 1
            else:
                # Impossible outcome
                add_idx = rng.integers(0, 2, size=1)[0]
            kq_barcodes[add_idx].append(barcode)
            kq_imcount[add_idx] += tally
            all_kq_barcodes.append(kq_barcodes)
            print(f"As if by magic, the imbalance fixed itself for {species}: {kq_imcount}")
            continue

        remainder = len(images_to_split) - delta
        cutoff = delta + remainder // 2
        if kq_imcount[0] < kq_imcount[1]:
            add_idx = 0
        elif kq_imcount[0] > kq_imcount[1]:
            add_idx = 1
        else:
            add_idx = rng.integers(0, 2, size=1)[0]

        kq_images[add_idx] = grp.iloc[:cutoff]["processid"].values
        kq_images[~add_idx] = grp.iloc[cutoff:]["processid"].values
        all_kq_images.append(kq_images)
        all_kq_barcodes.append(kq_barcodes)

    val_pc_target = 50
    rng = np.random.default_rng(seed=4)
    barcodes_selected = []
    soft_upper = 1.1

    all_kq_barcodes = []
    all_kq_images = []

    for species, grp in tqdm(g_test):
        kq_barcodes = [[], []]
        kq_imcount = [0, 0]

        dnasz = grp.groupby("dna_barcode_strip").size().sort_values()
        if len(dnasz) == 1:
            print(f"Couldn't split barcodes for {species} as there's only one barcode. Adding all to keys.")
            kq_barcodes[0].append(dnasz.index[0])
            all_kq_barcodes.append(kq_barcodes)
            continue

        add_idx = rng.integers(0, 2, size=1)[0]
        kq_barcodes[add_idx].append(dnasz.index[-1])
        kq_imcount[add_idx] += dnasz.iloc[-1]
        dnasz.drop(dnasz.index[-1], inplace=True)
        while len(dnasz) > 0:
            if len(kq_barcodes[0]) < len(kq_barcodes[1]):
                add_idx = 0
            elif len(kq_barcodes[0]) > len(kq_barcodes[1]):
                add_idx = 1
            elif kq_imcount[0] < kq_imcount[1]:
                add_idx = 0
            elif kq_imcount[0] > kq_imcount[1]:
                add_idx = 1
            else:
                add_idx = rng.integers(0, 2, size=1)[0]

            search_min = 0
            search_max = len(dnasz)
            if kq_imcount[add_idx] < kq_imcount[~add_idx]:
                search_min = (len(dnasz) + 1) // 2
            elif kq_imcount[add_idx] > kq_imcount[~add_idx]:
                search_max = len(dnasz) // 2
            if search_min >= search_max:
                search_min = max(0, search_min - 1)
                search_max = min(len(dnasz), search_max + 1)
            barcode_idx = rng.integers(search_min, search_max, size=1)[0]

            kq_barcodes[add_idx].append(dnasz.index[barcode_idx])
            kq_imcount[add_idx] += dnasz.iloc[barcode_idx]
            dnasz.drop(dnasz.index[barcode_idx], inplace=True)

        all_kq_barcodes.append(kq_barcodes)
        # print(kq_imcount)

    len(all_kq_barcodes)

    len(all_kq_images)

    all_k_barcodes = []
    all_q_barcodes = []
    for k_barcodes, q_barcodes in all_kq_barcodes:
        all_k_barcodes.extend(k_barcodes)
        all_q_barcodes.extend(q_barcodes)

    len(all_k_barcodes)

    len(all_q_barcodes)

    df.loc[df["dna_barcode_strip"].isin(all_k_barcodes), "role"] = "key"
    df.loc[df["dna_barcode_strip"].isin(all_q_barcodes), "role"] = "query"

    df.groupby(["split", "role"], dropna=False, observed=True).size()

    # # Save
    for c in df.columns:
        print(c)

    df.columns

    cols_to_save = [
        'processid',
        'sampleid',
        'image_file',
        'chunk',
        'taxon',
        'phylum',
        'class',
        'order',
        'family',
        'subfamily',
        'genus',
        'species',
        'is_novel_species',
        'dna_bin',
        'dna_barcode',
        'split',
        'role',
        'species_status',
        'country',
        'province/state',
        'coord-lat',
        'coord-lon',
        'image_measurement_value',
        'area_fraction',
        'scale_factor',
        'index_bioscan_1M_insect',
        'inferred_ranks',
    ]

    # Save output ----------------------------------------------------------------------
    print(f"Saving to {output_csv}")
    df[cols_to_save].to_csv(output_csv, index=False)
    print("Finished saving.")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Partition the BIOSCAN-5M dataset.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the BIOSCAN dataset.')
    parser.add_argument('output_csv', type=str, default='BIOSCAN-5M_Dataset_partitioned.csv', help='Path to save the partitioned dataset.')
    # Verbosity args ----------------------------------------------------------
    group = parser.add_argument_group("Verbosity")
    group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase the level of verbosity. The default verbosity level is %(default)s.",
    )
    group.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="Decrease the level of verbosity.",
    )
    return parser


def cli():
    parser = get_parser()
    args = parser.parse_args()
    args.verbose -= args.quiet
    del args.quiet
    main(args.csv_file, args.output_csv, verbose=args.verbose)


if __name__ == "__main__":
    cli()
