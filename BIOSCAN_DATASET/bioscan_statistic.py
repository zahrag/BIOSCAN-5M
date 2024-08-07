from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, size, col, collect_list
import math
from tabulate import tabulate
from bioscan_dataset import BioScan
import pandas as pd
import os
import numpy as np


class BioScanStats:

    def get_unknown_attr(self, df, attr):
        null_count = df.filter(col(attr).isNull()).count()
        return null_count

    def get_unique_groups(self, df, attr, with_unknown=True):

        if with_unknown:
            unique_groups = df.select(F.countDistinct(F.when(col(attr).isNull(),
                                                             "NULL").otherwise(col(attr)))).collect()[0][0]
        else:
            df_filtered = df.filter(col(attr).isNotNull())
            unique_groups = df_filtered.select(F.countDistinct(col(attr))).collect()[0][0]

        return unique_groups

    def get_attr_stats(self, df, attr, n_decimals=4):

        # Group the unique taxa by their corresponding indices
        grouped_df = df.groupBy(attr).agg(collect_list("index").alias("indices"))

        # Add total number of samples of each unique attribute
        grouped_df_with_count = grouped_df.withColumn("num_indices", size(col("indices")))

        # Get total dataset samples
        total_samples = grouped_df_with_count.agg(F.sum("num_indices")).collect()[0][0]

        # Calculate percentage and round to n decimals
        grouped_df_with_percentage = grouped_df_with_count.withColumn(
            "samples_percentage",
            F.round((F.col("num_indices") / total_samples) * 100, n_decimals)
        )
        grouped_df_with_percentage = grouped_df_with_percentage.fillna({attr: 'Unknown'})
        sorted_df = grouped_df_with_percentage.orderBy(F.col("num_indices").desc())

        return sorted_df

    def set_attr_statistics(self, configs):

        """
        This function calculates the statistics of an attribute.
        To get the statistics the attr argument in configuration must be set to genetic, geographic or size attributes.
        This included getting the number of unknown/unlabelled samples and the statistics of unique groups per attribute.
        :param configs: Configurations.
        :return:
        """

        if not configs['attr']:
            print("No taxonomic rank is specified.")
            return

        # Initialize a SparkSession
        spark = SparkSession.builder \
            .appName("Extract Sub-categories") \
            .getOrCreate()

        # Read the CSV file into a DataFrame
        df = spark.read.csv(configs['metadata'], header=True, inferSchema=True)
        df_with_index = df.withColumn("index", monotonically_increasing_id())
        df.show()

        # Count unknown/unlabelled samples
        attr = configs['attr']
        null_count = self.get_unknown_attr(df_with_index, attr)
        print(f"Number of unknown samples of {attr}: ", null_count)

        unique_groups_unknown = self.get_unique_groups(df_with_index, attr, with_unknown=True)
        print(f"Total number of unique groups of {attr} including unknown samples: ", unique_groups_unknown)

        unique_groups = self.get_unique_groups(df_with_index, attr, with_unknown=False)
        print(f"Total number of unique groups of {attr} excluding unknown samples: ", unique_groups)

        # Results
        sorted_df = self.get_attr_stats(df_with_index, attr)
        sorted_df.show(n=unique_groups)
        spark.stop()

    def print_table(self, data_dict, title, print_table=False):

        if not print_table:
            return

        print("\n\n" + "+" + "-" * 143 + "+")
        print(f"\t\t\t\t\t{title}")
        headings = list(data_dict.keys())
        values = list(data_dict.values())
        rows = zip(*values)
        formatted_rows = [
            [f'{val: .2f}' if isinstance(val, float) else str(val) for val in row]
            for row in rows
        ]
        print(tabulate(formatted_rows, headers=headings, tablefmt="grid"))

    def get_imbalance_ratio(self, data_dict):

        data_dict_length = {key: len(value)
                            for key, value in data_dict.items() if key != 'no_data'}

        # If using sorted class
        # clas = list(data_dict_length.keys())
        # imbalance_ratio = data_dict_length[clas[0]] / data_dict_length[clas[-1]]

        max_key, max_specimen = max(data_dict_length.items(), key=lambda item: item[1])
        min_key, min_specimen = min(data_dict_length.items(), key=lambda item: item[1])
        imbalance_ratio = max_specimen / min_specimen

        return imbalance_ratio

    def get_stat_dict(self, dataset, attr_list, dict_type='Genetic Attributes'):

        g_dict = {dict_type: [], 'Categories': [], 'Labelled Samples': [], 'Labelled (%)': [],
                  'Unlabelled Samples': [], 'Unlabelled (%)': [], 'Imbalance Ratio (IR)': []}

        for attr in attr_list:

            lst = dataset.data_list_mapping.get(attr)
            attr_dict = dataset.make_data_dict(lst)

            n_no_data, n_cat = 0, len(attr_dict.keys())
            if np.nan in attr_dict.keys():
                n_no_data, n_cat = len(attr_dict[np.nan]), len(attr_dict.keys()) - 1

            g_dict[dict_type].append(attr)
            g_dict['Categories'].append(n_cat)
            g_dict['Labelled Samples'].append(len(dataset.index) - n_no_data)
            g_dict['Labelled (%)'].append(100 * (len(dataset.index) - n_no_data) / len(dataset.index))
            g_dict['Unlabelled Samples'].append(n_no_data)
            g_dict['Unlabelled (%)'].append(100 * n_no_data / len(dataset.index))
            g_dict['Imbalance Ratio (IR)'].append(self.get_imbalance_ratio(attr_dict))

        return g_dict

    def get_attribute_statistics(self, dataset, level_name, get_attr='genetic'):
        """
        This function shows data statistics from metadata file of the dataset.
        :param dataset: Class BioScan Dataset.
        :param get_attr:
        :return:
        """

        if get_attr == 'genetic':
            list_attr = list({**dataset.taxonomy_groups_list_dict}.keys())
            list_attr.append('dna_bin')
            list_attr.append('dna_barcode')
        elif get_attr == 'geographic':
            list_attr = list({**dataset.geographic_list_dict}.keys())
        elif get_attr == 'size':
            list_attr = list({**dataset.size_list_dict}.keys())
        else:
            raise ValueError('Attribute type is NOT available!')

        g_dict = self.get_stat_dict(dataset, list_attr, dict_type=f'{get_attr.capitalize()} Attributes')

        # Print statistics
        scale = int(math.floor(math.log10(abs(len(dataset.index)))))
        s = 'k' if scale < 6 else 'M'
        n = round(len(dataset.index) / 10 ** scale)
        title = f"{get_attr.capitalize()} Statistics of the BIOSCAN-{n}{s} Insect dataset "
        title += f"having {len(dataset.index)} specimens"
        self.print_table(g_dict, title, print_table=True)

    def get_dataset_statistics(self, configs, dataset):
        """
        This function shows data statistics from metadata file of the dataset.
        :param configs: Configurations.
        :param dataset: Class BioScan Dataset.
        :return:
        """

        if not configs['attr_stat']:
            return

        print(f"\n\nGenerating {configs['attr_stat'].capitalize()}"
              f" statistics of {configs['level_name'][0].capitalize()} {configs['level_name'][1].capitalize()} ...")

        dataset.get_statistics(configs["metadata"], level_name=configs["level_name"])

        print("\n\n"+"-" * 90)
        print(f"\t\t\t\t\t\t\t\tCopyright")
        print("-"*90)
        print("Copyright Holder: CBG Photography Group")
        print("Copyright Institution: Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)")
        print("Photographer: CBG Robotic Imager")
        print("Copyright License: Creative Commons Attribution 3.0 Unported")
        print("Copyright Contact: collectionsBIO@gmail.com")
        print("Copyright Year: 2021")
        print("-"*90 + "\n")

        self.get_attribute_statistics(dataset, configs["level_name"], get_attr=configs['attr_stat'])


def show_statistics(configs):

    """
    This function shows/saves dataset statistics.
    :param configs: Configurations.
    :return:
    """

    dataset = BioScan()
    vis_stats = BioScanStats()
    vis_stats.get_dataset_statistics(configs, dataset)
    vis_stats.set_attr_statistics(configs)
