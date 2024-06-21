"""
BIOSCAN-5M PyTorch Dataset.

:Date: 2024-06-05
:Authors:
    - Scott C. Lowe <scott.code.lowe@gmail.com>
:Copyright: 2024, Scott C. Lowe
:License: MIT
"""

import os

import pandas as pd
import PIL
from torchvision.datasets.vision import VisionDataset

df_dtypes = {
    "processid": str,
    "sampleid": str,
    "taxon": "category",
    "phylum": "category",
    "class": "category",
    "order": "category",
    "family": "category",
    "subfamily": "category",
    "genus": "category",
    "species": "category",
    "dna_bin": "category",
    "dna_barcode": str,
    "country": "category",
    "province_state": "category",
    "coord-lat": float,
    "coord-lon": float,
    "image_measurement_value": float,
    "area_fraction": float,
    "scale_factor": float,
    "inferred_ranks": "uint8",
    "split": str,
    "index_bioscan_1M_insect": "Int64",
    "chunk": str,
}

df_usecols = [
    "processid",
    "chunk",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species",
    "dna_bin",
    "dna_barcode",
    "split",
]


def get_image_path(row):
    """Get the image path for a row in the metadata DataFrame.

    Parameters
    ----------
    row : pandas.Series
        A row in the metadata DataFrame.

    Returns
    -------
    str
        The path to the image file.
    """
    image_path = row["split"] + os.path.sep
    if pd.notna(row["chunk"]) and row["chunk"]:
        image_path += str(row["chunk"]) + os.path.sep
    image_path += row["processid"] + ".jpg"
    return image_path


class BIOSCAN5M(VisionDataset):
    """
    BIOSCAN-5M Dataset.

    Parameters
    ----------
    root : str
        The root directory, to contain the downloaded tarball file, and
        the image directory, BIOSCAN-5M.

    split : str, default="train"
        The dataset partition.
        One of: ``"pretrain"``, ``"train"``, ``"val"``, ``"test"``,
        ``"key_unseen"``, ``"val_unseen"``, ``"test_unseen"``,
        ``"other_heldout"``, or ``"all"``.
        Set to ``"all"`` to include all splits.

    modality : str or Iterable[str], default=("image", "dna")
        Which data modalities to use. One of, or a list of:
        ``"image"``, ``"dna"``.

    image_package : str, default="cropped_256"
        The package to load images from. One of:
        ``"original_full"``, ``"cropped"``, ``"original_256"``, ``"cropped_256"``.

    reduce_repeated_barcodes : str or bool, default=False
        Whether to reduce the dataset to only one sample per barcodes.

    max_nucleotides : int, default=None
        Maximum number of nucleotides to keep in the DNA barcode.

    target_type : str or Iterable[str], default="species"
        Type of target to use. One of, or a list of:
        ``"phylum"``, ``"class"``, ``"order"``, ``"family"``, ``"subfamily"``,
        ``"genus"``, ``"species"``, ``"dna_bin"``.

    transform : Callable, default=None
        Image transformation pipeline.

    dna_transform : Callable, default=None
        Barcode DNA transformation pipeline.

    target_transform : Callable, default=None
        Label transformation pipeline.
    """

    def __init__(
        self,
        root,
        split="train",
        modality=("image", "dna"),
        image_package="cropped_256",
        reduce_repeated_barcodes=False,
        max_nucleotides=None,
        target_type="species",
        transform=None,
        dna_transform=None,
        target_transform=None,
    ) -> None:
        root = os.path.expanduser(root)
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.metadata = None
        self.root = root
        self.image_package = image_package
        self.image_dir = os.path.join(self.root, "images", self.image_package)
        self.metadata_path = os.path.join(self.root, "metadata", "csv", "BIOSCAN_5M_Insect_Dataset_metadata.csv")

        self.split = split
        self.reduce_repeated_barcodes = reduce_repeated_barcodes
        self.max_nucleotides = max_nucleotides
        self.dna_transform = dna_transform

        if isinstance(modality, str):
            self.modality = [modality]
        else:
            self.modality = list(modality)

        if isinstance(target_type, str):
            self.target_type = [target_type]
        else:
            self.target_type = list(target_type)

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if not self._check_exists():
            raise EnvironmentError(f"{type(self).__name__} dataset not found in {self.image_dir}.")

        self.metadata = self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        sample = self.metadata.iloc[index]
        img_path = os.path.join(self.image_dir, sample["image_path"])
        values = []
        for modality in self.modality:
            if modality == "image":
                X = PIL.Image.open(img_path)
                if self.transform is not None:
                    X = self.transform(X)
            elif modality in ["dna_barcode", "dna", "barcode"]:
                X = sample["dna_barcode"]
                if self.dna_transform is not None:
                    X = self.dna_transform(X)
            else:
                raise ValueError(f"Unfamiliar modality: {modality}")
            values.append(X)

        target = []
        for t in self.target_type:
            target.append(sample[f"{t}_index"])

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        values.append(target)
        return tuple(values)

    def _check_exists(self) -> bool:
        """Check if the dataset is already downloaded and extracted.

        Returns
        -------
        bool
            True if the dataset is already downloaded and extracted, False otherwise.
        """
        check = os.path.isfile(self.metadata_path)
        if "images" in self.modality:
            check &= os.path.isdir(self.image_dir)
        return check

    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from CSV file and prepare it for training.

        Returns
        -------
        pandas.DataFrame
            The metadata DataFrame.
        """
        df = pd.read_csv(self.metadata_path, dtype=df_dtypes, usecols=df_usecols)
        if self.max_nucleotides is not None:
            df["dna_barcode"] = df["dna_barcode"].str[: self.max_nucleotides]
        if self.reduce_repeated_barcodes:
            # Shuffle the data order
            df = df.sample(frac=1, random_state=0)
            # Drop duplicated barcodes
            if self.reduce_repeated_barcodes == "rstrip_Ns":
                df["dna_barcode_strip"] = df["dna_barcode"].str.rstrip("N")
                df = df.drop_duplicates(subset=["dna_barcode_strip"])
                df.drop(columns=["dna_barcode_strip"], inplace=True)
            elif self.reduce_repeated_barcodes == "base":
                df = df.drop_duplicates(subset=["dna_barcode"])
            else:
                raise ValueError(f"Unfamiliar reduce_repeated_barcodes value: {self.reduce_repeated_barcodes}")
            # Re-order the data (reverting the shuffle)
            df = df.sort_index()
        # Filter to just the split of interest
        if self.split is not None and self.split != "all":
            df = df[df["split"] == self.split]
        # Add index columns to use for targets
        label_cols = [
            "phylum",
            "class",
            "order",
            "family",
            "subfamily",
            "genus",
            "species",
            "dna_bin",
        ]
        for c in label_cols:
            df[c + "_index"] = df[c].cat.codes
        # Add path to image file
        df["image_path"] = df.apply(get_image_path, axis=1)
        return df
