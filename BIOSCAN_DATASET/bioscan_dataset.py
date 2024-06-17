from torch.utils.data import Dataset
import os
from collections import defaultdict
import dataset_helper


class BioScan(Dataset):
    def __init__(self):
        """
            This class handles getting, setting and showing data statistics ...
            """

    def get_statistics(self, metadata, level_name=None):
        """
        This function sets data attributes read from metadata file of the dataset.
        This includes biological taxonomy information, DNA barcode indexes and RGB image names and chunk numbers.
        :param metadata: Path to the Metadata file (.csv)
        :param level_name: Taxonomic level and corresponding name to extract subset.
        :return:
        """

        self.metadata = metadata
        df = self.read_metadata(metadata)
        self.df = self.get_df(df, level_name=level_name)
        self.index = self.df.index.to_list()
        self.df_categories = self.df.keys().to_list()
        self.n_DatasetAttributes = len(self.df_categories)

        # Biological Taxonomy
        self.taxa_gt_sorted = {'0': 'domain', '1': 'kingdom', '2': 'phylum', '3': 'class', '4': 'order',
                               '5': 'family', '6': 'subfamily', '7': 'tribe', '8': 'genus', '9': 'species',
                               '10': 'subspecies', '11': 'name', '12': 'taxon'}

        self.taxonomy_groups_list_dict = {}
        for taxa in self.taxa_gt_sorted.values():
            if taxa in self.df_categories:
                self.taxonomy_groups_list_dict[taxa] = self.df[taxa].to_list()

        # Barcode and data Indexing
        self.barcode_list_dict = {'dna_barcode': [], 'dna_bin': [], 'processid': [], 'sampleid': []}
        for bar in self.barcode_list_dict.keys():
            if bar in self.df_categories:
                self.barcode_list_dict[bar] = self.df[bar].to_list()
            if f'{bar}_inferred' in self.df_categories:
                self.label_inferred[f'{bar}_inferred'] = self.df[f'{bar}_inferred'].to_list()

        # Organisms' Size
        self.size_list_dict = {'image_measurement_value': [], 'area_fraction': [], 'scale_factor':[]}
        for size_cat in self.size_list_dict.keys():
            if size_cat in self.df_categories:
                self.size_list_dict[size_cat] = self.df[size_cat].to_list()

        # Geographical location associated to the cite of collection
        self.geographic_list_dict = {'country': [], 'province_state': [], 'coord-lat': [], 'coord-lon': []}
        for geo_cat in self.geographic_list_dict.keys():
            if geo_cat in self.df_categories:
                self.geographic_list_dict[geo_cat] = self.df[geo_cat].to_list()

        lat_lon = [[lat, lon] if lat != 'no_data' and lon != 'no_data' else 'no_data'
                   for lat, lon in zip(self.geographic_list_dict.get('coord-lat'),
                                       self.geographic_list_dict.get('coord-lon'))]
        self.geographic_list_dict.update({'lat-lon': lat_lon})

        # Data Chunk
        if 'chunk' in self.df_categories:
            self.chunk_index = self.df['chunk'].to_list()

        if 'index_bioscan_1M_insect' in self.df_categories:
            self.index_1M_insect = self.df['index_bioscan_1M_insect'].to_list()

        if 'inferred_ranks' in self.df_categories:
            self.label_inferred = self.df['inferred_ranks'].to_list()

        self.data_list_mapping = {
            **self.taxonomy_groups_list_dict,
            **self.barcode_list_dict,
            **self.geographic_list_dict,
            **self.size_list_dict,
        }

    def __len__(self):
        return len(self.index)

    def read_metadata(self, metadata):
        """ Read a .tsv type metadata file """

        if os.path.isfile(metadata) and os.path.splitext(metadata)[1] == '.csv':
            df = dataset_helper.read_csv(metadata)
            print(f"Number of Samples of {os.path.basename(metadata)}: {len(df)}\n")
            return df
        else:
            raise ValueError(f'ERROR: Metadata (.csv) does NOT exist in: \n{metadata}!')

    def get_df(self, df, level_name=None):
        """
        Get the subset of metadata.
        :param df: Parent metadata
        :param level_name: 2D string object showing taxonomic group-level, and a name under the group-level.
        """

        if level_name == ['phylum', 'Arthropods']:
            """ When reading all samples of BIOSCAN dataset (exe., to split or to show statistics) """
            return df

        else:
            """ When reading a specific level and name (exe., level: class, and name: Insecta) """
            level, name = level_name
            if level not in df.keys():
                raise ValueError(f'\t\t\tERROR: Taxonomic level {level} is NOT available.')
            df = dataset_helper.keep_row_by_item(df, [name], level)
            if len(df) == 0:
                raise ValueError(f'\t\t\tERROR: There are NO {name} in {level}.')
            print(f"Number of Samples of {level.capitalize()} {name.capitalize()}: {len(df)}\n")
            return df

    def set_statistics(self, configs, split='all'):

        """
        This function sets dataset statistics
        :param configs: Configurations.
        :param split: Split: all, train, validation, test.
        :return:
        """

        self.get_statistics(configs['metadata'], configs['level_name'])

        # Get data list as one of the Biological Taxonomy
        if configs['group_level'] in self.df_categories:
            self.data_list = self.data_list_mapping.get(configs['group_level'])
            if self.data_list is None:
                raise ValueError(f'ERROR: "group_level" is NOT in: \n{self.df_categories}')
        else:
            raise ValueError(f'ERROR: "group_level" is NOT in: \n{self.df_categories}')

        # Get the data dictionary
        self.data_dict = self.make_data_dict(self.data_list)

        # Get numbered labels of the classes
        self.data_idx_label = self.class_to_ids(self.data_dict)

        # Get number of samples per class
        self.n_sample_per_class = self.get_n_sample_class(self.data_dict)

        # Get numbered data samples list
        self.data_list_ids = self.class_list_idx(self.data_list, self.data_idx_label)

    def make_data_dict(self, data_list):
        """
        This function create data dict key:label(exe., order name), value:indexes in data list
        :return:
        """

        data_dict = defaultdict(list)
        if all(isinstance(item, list) or item == 'no_data' for item in data_list):
            for ind, name in enumerate(data_list):
                if name == 'no_data':
                    data_dict[name].append(ind)
                else:
                    data_dict[tuple(name)].append(ind)
        else:
            for ind, name in enumerate(data_list):
                data_dict[name].append(ind)

        sorted_data_dict = dataset_helper.sort_dict_list(data_dict)

        return sorted_data_dict

    def class_to_ids(self, data_dict):

        """
        This function creates a numeric id for a class.
        :param data_dict: Data dictionary corresponding each class to its sample ids
        :return:
        """
        data_idx_label = {}
        data_names = list(data_dict.keys())
        for name in data_names:
            data_idx_label[name] = data_names.index(name)

        return data_idx_label

    def get_n_sample_class(self, data_dict):
        """
        This function compute number of samples per class.
        :param data_dict: Data dictionary corresponding each class to its sample ids
        :return:
        """

        data_samples_list = list(data_dict.values())
        n_sample_per_class = [len(class_samples) for class_samples in data_samples_list]

        return n_sample_per_class

    def class_list_idx(self, data_list, data_idx_label):
        """
        This function creates data list of numbered labels.
        :param data_list: data list of class names.
        :param data_idx_label: numeric ids of class names
        :return:
        """

        data_list_ids = []
        for data in data_list:
            data_list_ids.append(data_idx_label[data])

        return data_list_ids
