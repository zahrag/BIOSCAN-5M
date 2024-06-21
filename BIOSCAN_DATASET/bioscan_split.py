#!/usr/bin/env python
# coding: utf-8

"""
BIOSCAN-5M Dataset Partitioning.

:Date: 2024-05-30
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

# Setup
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


def find_novel_species(df, verbose=0):
    """
    Identify records containing novel names in the dataset.

    Records are considered novel based on the contents of their species and
    genus labels.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame with species and genus columns.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    pd.Series
        Boolean series indicating which records contain novel species.
    """
    is_novel_species = df["species"].notna() & df["species"].str.contains(
        r"[mM]+[aA]+[lLiI]+[aA]+[iIlL]+[sSzZ]+[eE]+", regex=True
    )
    is_novel_species |= df["species"].notna() & df["species"].str.contains(r"^[a-z]", regex=True)

    is_novel_genus = df["genus"].notna() & df["genus"].str.contains(
        r"[mM]+[aA]+[lLiI]+[aA]+[iIlL]+[sSzZ]+[eE]+", regex=True
    )
    sel = df["genus"].notna() & df["genus"].str.contains(r"^[a-z]", regex=True)
    sel &= ~(df["genus"].notna() & df["genus"].str.startswith("unclassified"))
    sel &= ~(df["genus"].notna() & df["genus"].str.startswith("unassigned"))
    is_novel_genus |= sel

    is_novel_species |= is_novel_genus
    is_novel_species |= df["species"].notna() & df["species"].str.contains(r"[0-9]", regex=True)
    is_novel_species |= df["species"].notna() & df["species"].str.contains(r"[A-Z][A-Z]", regex=True)

    search_terms = [
        r"\baff\b",
        r"\bn[\b\.\s]+sp\b",
        r"\bsp[\b\s]+ex\b",
        r"\bgen\b",
        r"\bsp\b\.?[\s\w\d]+",
        r"affinis\s+[0-9A-Z]+",
    ]
    search_suffix = ""
    for c in taxon_cols[-2:]:
        for search in search_terms:
            sel = df[c].notna() & df[c].str.contains(search + search_suffix, regex=True)
            if sum(sel) == 0:
                continue
            is_novel_species |= sel
            if verbose >= 1:
                print(c, search)
                display(df.loc[sel, taxon_cols].groupby(taxon_cols, dropna=False, observed=True).size())
                print()

    for c in ["species"]:
        for search in [r"[\_\?]"]:
            sel = df[c].notna() & df[c].str.contains(search + search_suffix, regex=True)
            if sum(sel) == 0:
                continue
            is_novel_species |= sel
            if verbose >= 1:
                print(c, search)
                display(df.loc[sel, taxon_cols].groupby(taxon_cols, dropna=False, observed=True).size())
                print()

    is_novel_species |= df["species"].notna() & df["species"].str.contains(r"[^$][A-Z]", regex=True)
    is_novel_species |= df["species"].notna() & df["species"].str.contains(r"[0-9]", regex=True)
    is_novel_species |= df["species"].notna() & df["species"].str.contains(".", regex=False)

    return is_novel_species


def stratified_dna_image_partition(
    g_test,
    target_fn,
    lower_fn,
    upper_fn,
    dna_upper_fn,
    soft_upper=1.1,
    soft_upper2=1.0,
    center_rand=False,
    top_rand=False,
    barcode_column="dna_barcode_strip",
    seed=None,
    verbose=0,
):
    """
    Partition samples for species into test sets based on DNA barcode counts.

    Samples are moved into the test set at the barcode level, so all samples
    with the same DNA barcode are either in the test set or not in the test set.

    Parameters
    ----------
    g_test : pd.DataFrameGroupBy
        Grouped DataFrame by species. Must have a ``"dna_barcode_strip"`` column.
    target_fn : function
        Function which maps from the number of samples for a species to the
        target number of samples to include in the test set.
        ``target_fn`` must take an integer argument and return an integer or float.
    lower_fn : function
        Function which maps from the number of samples for a species to the
        lower bound of samples to include in the test set.
        If we can not find ``lower_fn(n)`` samples to include in the test set,
        then we will not include the species in the test set.
        ``lower_fn`` must take an integer argument and return an integer or float.
    upper_fn : function
        Function which maps from the number of samples for a species to the
        upper bound of samples to include in the test set.
        We will stop addding samples if the barcode with the fewest samples
        would take us above ``upper_fn(n)``.
        ``upper_fn`` must take an integer argument and return an integer or float.
    dna_upper_fn : function
        Function which maps from the number of DNA barcodes for a species to
        the upper bound of DNA barcodes to include in the test set.
        ``dna_upper_fn`` must take an integer argument and return an integer or float.
    soft_upper : float, default=1.1
        A scale factor for the target number of samples to include in the test
        set. Modifies the target to permit us to select barcodes that would
        take us a little over the target.
    soft_upper2 : float, default=1.1
        A scale factor for the target number of samples to include in the test
        set. Modifies the target to permit us to select barcodes that would
        take us a little over the target for secondary options.
    center_rand : bool, default=False
        If True, only select barcodes from the middle 50% of the barcode
        options by number of samples per barcode.
    top_rand : bool, default=False
        If True, only select barcodes from the top 50% of the barcode
        options by number of samples per barcode.
    barcode_column : str, default="dna_barcode_strip"
        Column name for the DNA barcode, which will be grouped by per species.
    seed : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.
    """
    rng = np.random.default_rng(seed=seed)
    barcodes_selected = []
    if verbose >= 1:
        display(g_test.size())
    for species, grp in tqdm(g_test):
        if verbose >= 3:
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
        dnasz = grp.groupby(barcode_column).size().sort_values()
        if verbose >= 3:
            display(dnasz)
        n_barcodes = len(dnasz)
        # lb_barcodes = dna_lower_fn(n_barcodes)
        ub_barcodes = dna_upper_fn(n_barcodes)
        if verbose >= 3:
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
        while True:
            if verbose >= 4:
                print("n_alloc", n_alloc)
                # display(dnasz)
            # We only want to add samples which keep us below or at the target
            # And cumsum indicates the largest value we can reach with the smallest k entries,
            # so we need to add one of the samples that takes the cumsum to the target.
            is_option = (dnasz <= target - n_alloc).values & (dnasz_cs >= target * soft_upper - n_alloc).values
            if sum(is_option) == 0:
                if verbose >= 4:
                    print("No ideal options")
                # No ideal options, so let's make do
                if n_alloc == 0 and not any(dnasz <= target):
                    # No options and nothing added so far
                    if dnasz.iloc[0] >= lb_samp and dnasz.iloc[0] < n / 2:
                        if verbose >= 4:
                            print("Nothing added, taking first")
                        idx = 0
                    else:
                        break
                elif dnasz.iloc[0] > target * soft_upper2 - n_alloc:
                    # Everything would take us above the remaining target
                    if dnasz.iloc[0] > (target - n_alloc) / 2:
                        # Shouldn't add the smallest because the residual would be larger than it is now
                        if verbose >= 4:
                            print("Last possibility avoiding smallest, increases residual")
                        break
                    if n_alloc + dnasz.iloc[0] > ub_samp:
                        # Can't add the smallest as it takes us above our upperbound
                        if verbose >= 4:
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
                    if option_cutoff_idx > 0 and abs(n_alloc + dnasz.iloc[option_cutoff_idx - 1] - target) < abs(
                        target - (n_alloc + dnasz.iloc[option_cutoff_idx])
                    ):
                        idx = option_cutoff_idx - 1
                    if n_alloc + dnasz.iloc[option_cutoff_idx] > ub_samp:
                        break
                    if verbose >= 4:
                        print("Breaking between criteria")
            elif len(grp_barcodes_selected) + 1 >= ub_barcodes:
                # We can only add one more, so add the most populated DNA that makes sense to add
                option_cutoff_idx = np.nonzero(dnasz <= target - n_alloc)[0][-1]
                idx = option_cutoff_idx
                if (
                    option_cutoff_idx < len(dnasz) - 1
                    and abs(n_alloc + dnasz.iloc[option_cutoff_idx + 1] - target)
                    < abs(target - (n_alloc + dnasz.iloc[option_cutoff_idx]))
                    and n_alloc + dnasz.iloc[option_cutoff_idx + 1] <= ub_samp
                ):
                    idx = option_cutoff_idx + 1
                if verbose >= 4:
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
                if verbose >= 4:
                    print(f"Randomly selected item #{idx} from {n_options} options with {dnasz.iloc[idx]} samp")
            if verbose >= 4:
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
            if verbose >= 4:
                print(f"End of loop with {len(dnasz)} = {len(dnasz_cs)} / {n_barcodes} barcodes remaining")
        if n_alloc >= lb_samp:
            barcodes_selected.append(grp_barcodes_selected)
            if verbose >= 3:
                print(f"Added {len(grp_barcodes_selected)} barcodes")
        elif verbose >= 3:
            print(f"Skipped adding {len(grp_barcodes_selected)} barcodes")
        if verbose >= 3:
            print(len(grp_barcodes_selected), "barcodes selected")
            display(grp_barcodes_selected)
    return barcodes_selected


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


def show_partition_stats(df, show_empty=False):
    out = r"partition          samples         barcodes         species    " + "\n"
    out += "---------------------------------------------------------------\n"
    n_barcode_total = df["dna_barcode"].nunique()
    n_species_total = df["species"].nunique()
    partitions = [
        "pretrain",
        "seen",
        "train",
        "val",
        "test",
        "unseen",
        "key_unseen",
        "val_unseen",
        "test_unseen",
    ]
    for partition in partitions + list(set(df["split"].unique()).difference(partitions)):
        sel = df["split"] == partition
        if not show_empty and sum(sel) == 0:
            continue
        n_barcode = df.loc[sel, "dna_barcode"].nunique()
        n_species = df.loc[sel, "species"].nunique()
        out += f"{partition:17s}"
        out += f" {sum(sel):8d} {100 * sum(sel) / len(df):5.1f}%"
        out += f" {n_barcode:8d} {100 * n_barcode / n_barcode_total:5.1f}%"
        out += f" {n_species:6d} {100 * n_species / n_species_total:5.1f}%"
        out += "\n"
    print("\n" + out)


def main(input_csv, output_csv, verbose=1):
    """
    Partition data into splits.

    Divide the data into the following partitions:

        - pretrain: samples without species labels
        - train: samples with scientific names as species labels
        - val: validation data, ~5% of train data, matching distribution
        - test: test data, ~11% of train data, with flattened distribution
        - key_unseen: key database for unseen species
        - val_unseen: validation data for unseen species, 20% of unseen species
        - test_unseen: test data for unseen species, with flatten distribution
        - other_heldout: remaining samples with placeholder species labels

    Parameters
    ----------
    input_csv : str
        Input CSV file name.
    output_csv : str
        Output CSV file name.
    verbose : int, default=1
        Verbosity level.
    """
    # Load metadata
    if verbose >= 0:
        print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv, dtype=df_dtypes)
    if verbose >= 0:
        print("Finished loading metadata.")
    if verbose >= 1:
        print(f"There are {len(df)} samples in the dataset.")

    df["is_novel_species"] = find_novel_species(df, verbose=verbose - 2)
    df["dna_barcode_strip"] = df["dna_barcode"].str.strip("N")

    df["split"] = "TBD"

    # test_unseen partition ------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating pretrain split")
        print("Adding samples without species labels and samples of placeholder species to pretrain split")

    df.loc[df["species"].isna() | df["is_novel_species"], "split"] = "pretrain"
    df.loc[df["species"].notna() & ~df["is_novel_species"], "split"] = "seen"

    if verbose >= 2:
        show_partition_stats(df)

    # unseen partition -----------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating unseen species set")

    if verbose >= 3:
        partition_candidates = df.loc[
            df["species"].notna() & (df["split"] == "seen"),
            ["processid", "genus", "species"],
        ]
        sp_sz = partition_candidates.groupby("species", observed=True).size()
        print()
        print(sp_sz)
    if verbose >= 4:
        plt.hist(sp_sz, np.arange(100))
        plt.show()
        plt.hist(sp_sz, np.arange(max(sp_sz)), cumulative=True)
        plt.show()
        plt.hist(sp_sz, np.arange(200), cumulative=True)
        plt.show()

    if verbose >= 4:
        print("cut  at_cut  cumm  (%)     n_samp  (%)")
        print("-----------------------------------------")
        for cutoff in range(51):
            sel_sp = sp_sz <= cutoff
            n_sel_sp = sum(sel_sp)
            n_sel_samp = sum(partition_candidates["species"].isin(sp_sz[sel_sp].index))
            print(
                f"{cutoff:3d}  {sum(sp_sz == cutoff):5d}"
                f"  {n_sel_sp:5d}  {100 * n_sel_sp / len(sel_sp):5.2f}%"
                f"  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%"
            )

    if verbose >= 3:
        ge_sz = partition_candidates.groupby("genus", observed=True).size()
        print()
        print(ge_sz)
    if verbose >= 4:
        plt.hist(ge_sz, np.arange(200), cumulative=True)
        plt.show()

    if verbose >= 4:
        print("cut  n_genus (%)     n_samp  (%)")
        print("----------------------------------")
        for cutoff in range(51):
            sel_ge = ge_sz <= cutoff
            n_sel_ge = sum(sel_ge)
            n_sel_samp = sum(partition_candidates["genus"].isin(ge_sz[sel_ge].index))
            print(
                f"{cutoff:3d}  {n_sel_ge:5d}"
                f"  {100 * n_sel_ge / len(ge_sz):5.2f}%"
                f"  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%"
            )

    if verbose >= 4:
        nspecies_per_genus = (
            partition_candidates[["genus", "species"]].drop_duplicates().groupby("genus", observed=True).size()
        )
        plt.hist(nspecies_per_genus, np.arange(40), cumulative=True)
        plt.show()

    if verbose >= 4:
        _sp_sz_trainval_dna = (
            df.loc[df["split"].isin(["seen", "train", "val", "test"])]
            .drop_duplicates("dna_barcode_strip")
            .groupby("species", observed=True, dropna=False)
            .size()
        )
        plt.hist(_sp_sz_trainval_dna, np.arange(40))
        plt.show()
        plt.hist(_sp_sz_trainval_dna, np.arange(40), cumulative=True)
        plt.show()

    if verbose >= 4:
        print("cut  at_cut  cumm  (%)     n_samp  (%)")
        print("-----------------------------------------")
        for cutoff in range(51):
            sel_sp = _sp_sz_trainval_dna <= cutoff
            n_sel_sp = sum(sel_sp)
            n_sel_samp = sum(partition_candidates["species"].isin(_sp_sz_trainval_dna[sel_sp].index))
            print(
                f"{cutoff:3d}  {sum(_sp_sz_trainval_dna == cutoff):5d}"
                f"  {n_sel_sp:5d}  {100 * n_sel_sp / len(sel_sp):5.2f}%"
                f"  {n_sel_samp:6d}  {100 * n_sel_samp / len(partition_candidates):5.2f}%"
            )

    sel_known_genus = df["genus"].isin(df.loc[df["split"] == "seen", "genus"])

    sel_viable = sel_known_genus & (df["split"] == "pretrain") & df["species"].notna()
    if verbose >= 3:
        display(df.loc[sel_viable, taxon_cols].groupby(taxon_cols, dropna=False, observed=True).size())

    g_pt_species = df.loc[sel_viable, ["genus", "species"]].groupby("species", observed=True).size()

    if verbose >= 1:
        print("Placing species whose genus is not in pretrain, with at least 8 observations, into unseen")
    df.loc[df["species"].isin(g_pt_species[g_pt_species >= 8].index), "split"] = "unseen"

    if verbose >= 2:
        show_partition_stats(df)

    if verbose >= 3:
        print("Genera in unseen:")
        print(sorted(df.loc[df["split"] == "unseen", "genus"].unique()))

    # ----------------------------------------------------------------------------------
    # Single sample species
    if verbose >= 2:
        print("Adding samples of singly-annotated non-placeholder species train split")

    partition_candidates = df.loc[
        df["species"].notna() & df["split"].isin(["seen", "TBD"]),
        ["processid", "genus", "species"],
    ]

    # Add all of single species samples to training set
    sp_sz = partition_candidates.groupby("species", observed=True).size()
    single_species = list(sp_sz[sp_sz == 1].index)
    df.loc[df["species"].isin(single_species), "split"] = "train"

    if verbose >= 2:
        show_partition_stats(df)

    # test_seen ------------------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating test_seen split")

    if verbose >= 2:
        partition_candidates = df.loc[
            df["species"].notna() & (df["split"] == "seen"),
            ["processid", "dna_barcode_strip", "genus", "species"],
        ]
        sp_sz = partition_candidates.groupby("species", observed=True).size()
        sp_sztest = sp_sz.apply(test_split_fn)
        print(sum(sp_sztest > 0), "species to place")
        print(sum(sp_sztest), "samples to place")

        df_tmp = pd.concat([sp_sz, sp_sztest], axis=1, keys=["total", "test"])
        df_tmp["train"] = df_tmp["total"] - df_tmp["test"]
        df_tmp["test_pc"] = 100 * df_tmp["test"] / df_tmp["total"]
        df_tmp["train_pc"] = 100 * df_tmp["train"] / df_tmp["total"]
        print(df_tmp)

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[
        df["species"].notna() & (df["split"] == "seen"),
        ["processid", "dna_barcode_strip", "genus", "species"],
    ]
    partition_candidates = partition_candidates.sample(frac=1, random_state=0)

    g_test = partition_candidates.groupby("species", observed=True)

    barcodes_selected = stratified_dna_image_partition(
        g_test,
        test_split_fn,
        lambda x: 3,
        test_split_fn_ub,
        test_split_fn_ub_barcodes,
        soft_upper=1.1,
        top_rand=True,
        seed=1,
        verbose=verbose - 2,
    )
    if verbose >= 2:
        print(len(barcodes_selected), "species selected")

    barcodes_selected = [b for bs in barcodes_selected for b in bs]
    if verbose >= 2:
        print(len(barcodes_selected), "barcodes selected")

    sel = df["dna_barcode_strip"].isin(barcodes_selected)
    if verbose >= 2:
        print(sum(sel), "samples selected")
        print(df.loc[sel, "species"].nunique(), "species selected")

    df.loc[sel, "split"] = "test"

    if verbose >= 2:
        _sp_sz_trainval = (
            df.loc[df["split"].isin(["seen", "train", "val"])].groupby("species", observed=True, dropna=False).size()
        )
        _sp_sz_test = df.loc[df["split"] == "test"].groupby("species", observed=True, dropna=False).size()
        print(
            f"{_sp_sz_test.sum()} of {_sp_sz_trainval.sum()} viable samples"
            f" ({100 * _sp_sz_test.sum() / _sp_sz_trainval.sum():.1f}%) placed in test_seen"
        )

    if verbose >= 3:
        df_tmp2 = pd.concat([_sp_sz_trainval, _sp_sz_test], axis=1, keys=["trainval", "test"])
        df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
        df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
        df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]

        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    if verbose >= 3:
        _sp_sz_trainval_dna = (
            df.loc[df["split"].isin(["seen", "train", "val"])]
            .drop_duplicates("dna_barcode_strip")
            .groupby("species", observed=True, dropna=False)
            .size()
        )
        _sp_sz_test_dna = (
            df.loc[df["split"] == "test"]
            .drop_duplicates("dna_barcode_strip")
            .groupby("species", observed=True, dropna=False)
            .size()
        )
        df_tmp2 = pd.concat([_sp_sz_trainval_dna, _sp_sz_test_dna], axis=1, keys=["trainval", "test"])
        df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
        df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
        df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test_pc"])
        plt.xscale("log")
        plt.show()

    if verbose >= 3:
        n_samp_trainvaltest = sum(df["split"].isin(["seen", "train", "val", "test"]))
        n_dna_trainvaltest = df.loc[df["split"].isin(["seen", "train", "val", "test"]), "dna_barcode"].nunique()
        for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
            print(
                f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 3:
        n_samp_trainvaltest = df["split"].isin(["seen", "train", "val", "test"])
        n_barcodes = df["dna_barcode"].nunique()
        for partition, n_samp in zip(
            *np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)
        ):
            print(
                f"{partition:15s} {n_samp:7d}"
                f" {100 * n_samp / n_barcodes:5.2f}%"
                f" {100 * n_samp / n_dna_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 2:
        show_partition_stats(df)

    # val partition --------------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating validation split")

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[
        df["species"].notna() & (df["split"] == "seen"),
        ["processid", "dna_barcode_strip", "genus", "species"],
    ]
    partition_candidates = partition_candidates.sample(frac=1, random_state=2)

    g_test = partition_candidates.groupby("species", observed=True)

    val_pc_target = 5

    if verbose >= 1:
        print(f"Adding samples to val partition. Target: {val_pc_target}% of samples from non-test seen species.")

    barcodes_selected = stratified_dna_image_partition(
        g_test,
        target_fn=lambda x: 0 if x < 20 else val_pc_target / 100 * x,
        lower_fn=lambda _: 0,
        upper_fn=lambda x: 2 * val_pc_target / 100 * x,
        dna_upper_fn=lambda x: val_pc_target / 100 * x,
        soft_upper=1.1,
        soft_upper2=1.1,
        top_rand=True,
        seed=3,
        verbose=verbose-2
    )

    if verbose >= 2:
        print(len(barcodes_selected), "species selected")

    barcodes_selected = [b for bs in barcodes_selected for b in bs]
    if verbose >= 2:
        print(len(barcodes_selected), "barcodes selected")

    sel = df["dna_barcode_strip"].isin(barcodes_selected)
    if verbose >= 2:
        print(sum(sel), "samples selected")
        print(df.loc[sel, "species"].nunique(), "species selected")

    if verbose >= 2:
        target_samp = val_pc_target / 100 * sum(df["split"].isin(["train", "seen"]))
        target_dna = val_pc_target / 100 * df.loc[df["split"].isin(["train", "seen"]), "dna_barcode_strip"].nunique()
        target_spec = df.loc[df["split"].isin(["train"]), "species"].nunique()
        print()
        print("          Target Current")
        print("------------------------")
        print(f"Samples  {target_samp:7.1f} {sum(sel):5d}")
        print(f"Barcodes {target_dna:7.1f} {len(barcodes_selected):5d}")
        print(f"Species  {target_spec:7.1f} {df.loc[sel, 'species'].nunique():5d}")
        print()

    g_val_sz = g_test.size()
    g_val_sz = g_val_sz[g_val_sz < 20]

    small_spec_target = val_pc_target / 100 * (len(single_species) + sum(g_val_sz))
    small_spec_target = round(small_spec_target)
    if verbose >= 1:
        print("Adding single-sample barcodes from species with 7-19 samples to val split")
        print("Target number of val samples from long-tail species:", small_spec_target)
    df_test = df.loc[df["split"] == "test"]
    n_alloc_total = 0
    barcodes_selected2 = []
    tally_skip_size = 0
    tally_skip_test = 0
    tally_skip_dna = 0
    for species, grp in tqdm(g_test):
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

    if verbose >= 3:
        print("tally_skip_size", tally_skip_size)
        print("tally_skip_test", tally_skip_test)
        print("tally_skip_dna", tally_skip_dna)
        print(n_alloc_total, len(barcodes_selected2), small_spec_target)

    sel = df["dna_barcode_strip"].isin(barcodes_selected + barcodes_selected2)

    if verbose >= 2:
        target_samp = val_pc_target / 100 * sum(df["split"].isin(["train", "seen"]))
        target_dna = val_pc_target / 100 * df.loc[df["split"].isin(["train", "seen"]), "dna_barcode_strip"].nunique()
        target_spec = df.loc[df["split"].isin(["train"]), "species"].nunique()
        print()
        print("          Target Current")
        print("------------------------")
        print(f"Samples  {target_samp:7.1f} {sum(sel):5d}")
        print(f"Barcodes {target_dna:7.1f} {len(barcodes_selected) + len(barcodes_selected2):5d}")
        print(f"Species  {target_spec:7.1f} {df.loc[sel, 'species'].nunique():5d}")
        print()

    df.loc[sel, "split"] = "val"

    if verbose >= 3:
        n_samp_trainvaltest = sum(df["split"].isin(["seen", "train", "val", "test"]))
        n_dna_trainvaltest = df.loc[df["split"].isin(["seen", "train", "val", "test"]), "dna_barcode"].nunique()
        print("seen samples distribution")
        for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
            if partition not in ["seen", "train", "val", "test"]:
                continue
            print(
                f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 3:
        n_samp_trainvaltest = df["split"].isin(["seen", "train", "val", "test"])
        n_barcodes = df["dna_barcode"].nunique()
        print("seen barcodes distribution")
        for partition, n_samp in zip(
            *np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)
        ):
            if partition not in ["seen", "train", "val", "test"]:
                continue
            print(
                f"{partition:15s} {n_samp:7d}"
                f" {100 * n_samp / n_barcodes:5.2f}%"
                f" {100 * n_samp / n_dna_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 2:
        show_partition_stats(df)

    # train partition ------------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating training split")

    df.loc[df["split"] == "seen", "split"] = "train"

    if verbose >= 3:
        n_samp_trainvaltest = sum(df["split"].isin(["seen", "train", "val", "test"]))
        n_dna_trainvaltest = df.loc[df["split"].isin(["seen", "train", "val", "test"]), "dna_barcode"].nunique()
        print("seen samples distribution")
        for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
            if partition not in ["seen", "train", "val", "test"]:
                continue
            print(
                f"{partition:15s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 3:
        n_samp_trainvaltest = df["split"].isin(["seen", "train", "val", "test"])
        n_barcodes = df["dna_barcode"].nunique()
        print("seen barcodes distribution")
        for partition, n_samp in zip(
            *np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)
        ):
            if partition not in ["seen", "train", "val", "test"]:
                continue
            print(
                f"{partition:15s} {n_samp:7d}"
                f" {100 * n_samp / n_barcodes:5.2f}%"
                f" {100 * n_samp / n_dna_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 2:
        show_partition_stats(df)

    # Unseen partitioning --------------------------------------------------------------
    if verbose >= 1:
        print("\nPartitioning of unseen species set")

    # test_unseen partition -------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating test_unseen split")

    if verbose >= 2:
        partition_candidates = df.loc[
            df["species"].notna() & (df["split"] == "unseen"),
            ["processid", "genus", "species"],
        ]
        sp_sz = partition_candidates.groupby("species", observed=True).size()
        sp_sztest = sp_sz.apply(test_split_fn)
        print(sum(sp_sztest > 0), "species to place")
        print(sum(sp_sztest), "samples to place")

        df_tmp = pd.concat([sp_sz, sp_sztest], axis=1, keys=["total", "test"])
        df_tmp["train"] = df_tmp["total"] - df_tmp["test"]
        df_tmp["test_pc"] = 100 * df_tmp["test"] / df_tmp["total"]
        df_tmp["train_pc"] = 100 * df_tmp["train"] / df_tmp["total"]
        print(df_tmp)

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[
        df["species"].notna() & (df["split"] == "unseen"),
        ["processid", "dna_barcode_strip", "genus", "species"],
    ]
    partition_candidates = partition_candidates.sample(frac=1, random_state=4)

    g_test = partition_candidates.groupby("species", observed=True)

    barcodes_selected = stratified_dna_image_partition(
        g_test,
        test_split_fn,
        lambda x: 3,
        test_split_fn_ub,
        test_split_fn_ub_barcodes,
        top_rand=True,
        seed=5,
    )
    if verbose >= 2:
        print(len(barcodes_selected), "species selected")

    barcodes_selected = [b for bs in barcodes_selected for b in bs]
    if verbose >= 2:
        print(len(barcodes_selected), "barcodes selected")

    sel = df["dna_barcode_strip"].isin(barcodes_selected)
    if verbose >= 2:
        print(sum(sel), "samples selected")
        print(df.loc[sel, "species"].nunique(), "species selected")

    df.loc[sel, "split"] = "test_unseen"

    if verbose >= 2:
        _sp_sz_trainval = (
            df.loc[df["split"] == "unseen"].groupby("species", observed=True, dropna=False).size()
        )
        _sp_sz_test = df.loc[df["split"] == "test_unseen"].groupby("species", observed=True, dropna=False).size()
        print(
            f"{_sp_sz_test.sum()} of {_sp_sz_trainval.sum()} viable samples"
            f" ({100 * _sp_sz_test.sum() / _sp_sz_trainval.sum():.1f}%) placed in test_seen"
        )

    if verbose >= 3:
        df_tmp2 = pd.concat([_sp_sz_trainval, _sp_sz_test], axis=1, keys=["trainval", "test"])
        df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
        df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
        df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    if verbose >= 3:
        _sp_sz_trainval_dna = (
            df.loc[df["split"] == "unseen"]
            .drop_duplicates("dna_barcode_strip")
            .groupby("species", observed=True, dropna=False)
            .size()
        )
        _sp_sz_test_dna = (
            df.loc[df["split"] == "test_unseen"]
            .drop_duplicates("dna_barcode_strip")
            .groupby("species", observed=True, dropna=False)
            .size()
        )
        df_tmp2 = pd.concat([_sp_sz_trainval_dna, _sp_sz_test_dna], axis=1, keys=["trainval", "test"])
        df_tmp2.loc[df_tmp2["test"].isna(), "test"] = 0
        df_tmp2["total"] = df_tmp2["trainval"] + df_tmp2["test"]
        df_tmp2["test_pc"] = 100 * df_tmp2["test"] / df_tmp2["total"]
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test"])
        plt.xscale("log")
        plt.show()
        plt.scatter(df_tmp2["total"], df_tmp2["test_pc"])
        plt.xscale("log")
        plt.show()

    if verbose >= 3:
        n_samp_trainvaltest = sum(df["split"].str.contains("unseen"))
        n_dna_trainvaltest = df.loc[df["split"].str.contains("unseen"), "dna_barcode"].nunique()
        for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
            print(
                f"{partition:18s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 3:
        n_samp_trainvaltest = df["split"].str.contains("unseen")
        n_barcodes = df["dna_barcode"].nunique()
        for partition, n_samp in zip(
            *np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)
        ):
            print(
                f"{partition:18s} {n_samp:7d}"
                f" {100 * n_samp / n_barcodes:5.2f}%"
                f" {100 * n_samp / n_dna_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 2:
        show_partition_stats(df)

    # val_unseen partition -------------------------------------------------------------
    if verbose >= 1:
        print("\nCreating val_unseen split")

    # Randomize the order of the data so we select random samples from each species
    partition_candidates = df.loc[
        df["species"].notna() & (df["split"] == "unseen"),
        ["processid", "dna_barcode_strip", "genus", "species"],
    ]
    partition_candidates = partition_candidates.sample(frac=1, random_state=6)

    g_test = partition_candidates.groupby("species", observed=True)

    # val_pc_target = 100 / 3
    val_pc_target = 20

    if verbose >= 1:
        print(
            f"Adding samples to unseen_val partition. Target: {val_pc_target}% of samples from non-test unseen species."
        )

    barcodes_selected = stratified_dna_image_partition(
        g_test,
        target_fn=lambda x: val_pc_target / 100 * x,
        lower_fn=lambda _: 0,
        upper_fn=lambda x: 2 * val_pc_target / 100 * x,
        dna_upper_fn=lambda x: val_pc_target / 100 * x,
        soft_upper=1.1,
        soft_upper2=1.1,
        top_rand=True,
        seed=7,
        verbose=verbose - 2,
    )

    if verbose >= 2:
        print(len(barcodes_selected), "species selected")

    barcodes_selected = [b for bs in barcodes_selected for b in bs]
    if verbose >= 2:
        print(len(barcodes_selected), "barcodes selected")

    sel = df["dna_barcode_strip"].isin(barcodes_selected)
    if verbose >= 2:
        print(sum(sel), "samples selected")
        print(df.loc[sel, "species"].nunique(), "species selected")

    if verbose >= 2:
        target_samp = val_pc_target / 100 * sum(df["split"] == "unseen")
        target_dna = val_pc_target / 100 * df.loc[df["split"] == "unseen", "dna_barcode_strip"].nunique()
        target_spec = df.loc[df["split"] == "unseen", "species"].nunique()
        print()
        print("          Target Current")
        print("------------------------")
        print(f"Samples  {target_samp:7.1f} {sum(sel):5d}")
        print(f"Barcodes {target_dna:7.1f} {len(barcodes_selected):5d}")
        print(f"Species  {target_spec:7.1f} {df.loc[sel, 'species'].nunique():5d}")
        print()

    sel = df["dna_barcode_strip"].isin(barcodes_selected)
    df.loc[sel, "split"] = "val_unseen"

    if verbose >= 3:
        n_samp_trainvaltest = sum(df["split"].str.contains("unseen"))
        n_dna_trainvaltest = df.loc[df["split"].str.contains("unseen"), "dna_barcode"].nunique()
        print("unseen samples distribution")
        for partition, n_samp in zip(*np.unique(df["split"], return_counts=True)):
            if "unseen" not in partition:
                continue
            print(
                f"{partition:18s} {n_samp:7d} {100 * n_samp / len(df):5.2f}% {100 * n_samp / n_samp_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 3:
        n_samp_trainvaltest = df["split"].str.contains("unseen")
        n_barcodes = df["dna_barcode"].nunique()
        print("unseen barcodes distribution")
        for partition, n_samp in zip(
            *np.unique(df[["split", "dna_barcode"]].drop_duplicates()["split"], return_counts=True)
        ):
            if "unseen" not in partition:
                continue
            print(
                f"{partition:18s} {n_samp:7d}"
                f" {100 * n_samp / n_barcodes:5.2f}%"
                f" {100 * n_samp / n_dna_trainvaltest:7.2f}%"
            )
        print()

    if verbose >= 2:
        show_partition_stats(df)

    # Relabel remaining samples --------------------------------------------------------
    if verbose >= 1:
        print("\nCreating key_unseen split")

    df.loc[df["split"] == "unseen", "split"] = "key_unseen"

    if verbose >= 2:
        show_partition_stats(df)

    # other_heldout partition ----------------------------------------------------------
    if verbose >= 1:
        print("\nCreating other_heldout split")

    df.loc[df["species"].notna() & df["split"].isin(["TBD", "pretrain"]), "split"] = "other_heldout"

    if verbose >= 1:
        show_partition_stats(df)

    # Save output ----------------------------------------------------------------------
    df.drop(columns=["dna_barcode_strip", "is_novel_species"], inplace=True)
    if verbose >= 0:
        print(f"Saving to {output_csv}")
    df.to_csv(output_csv, index=False)
    if verbose >= 0:
        print("Finished saving.")


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Partition the BIOSCAN-5M dataset.")
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the CSV file containing the BIOSCAN dataset.",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        default="BIOSCAN-5M_Dataset_partitioned.csv",
        help="Path to save the partitioned dataset.",
    )
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
    main(args.input_csv, args.output_csv, verbose=args.verbose)


if __name__ == "__main__":
    cli()
