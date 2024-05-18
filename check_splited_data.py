import pandas as pd
import argparse
from collections import Counter
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splitted_metadata_path', type=str, default="BIOSCAN_metadata/BIOSCAN_5M_Insect_Dataset_metadata_splitted.tsv",)

    args = parser.parse_args()

    df = pd.read_csv(args.splitted_metadata_path, sep='\t')

    print(df['split'].value_counts())
