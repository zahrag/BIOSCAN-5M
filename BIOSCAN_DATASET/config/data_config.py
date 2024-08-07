import datetime
import argparse


def config_base(parser):

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    parser.add_argument('--date_time',
                        type=str,
                        default=timestamp,
                        help='Data & time of the experiment')

    parser.add_argument('--root',
                        type=str,
                        default='root_cz',
                        help='Set root of environment.')

    parser.add_argument('--level_name',
                        nargs=2,
                        default=['phylum', 'Arthropods'],
                        choices=['phylum', 'Arthropods', 'class', 'Insecta', 'order', 'Diptera', 'family', 'Cecidomyiidae'],
                        help='Two separate strings representing the taxonomic level and name to extract subset.')

    parser.add_argument('--group_level',
                        type=str,
                        default=None,
                        choices=['class', 'order', 'family', 'subfamily', 'genus', 'species', 'taxon'],
                        help='Taxonomic group ranking.')

    parser.add_argument('--attr',
                        type=str,
                        default=None,
                        help='A Dataset Attribute (e.g., order).')

    parser.add_argument('--metadata',
                        type=str,
                        default='BIOSCAN_metadata/BIOSCAN_5M_Insect_Dataset_metadata.csv',
                        help='Path to the metadata file of the dataset.')

    parser.add_argument('--bbox',
                        type=str,
                        default='/BIOSCAN_5M_Insect_bbox.tsv',
                        help='Path to the bounding box file.')

    return parser


def config_run_module(parser):

    parser.add_argument('--download',
                        default=False,
                        action='store_true',
                        help='IF to download dataset?')

    parser.add_argument('--attr_stat',
                        type=str,
                        default=None,
                        choices=['genetic', 'geographic', 'size'],
                        help='Type of statistics, if None skip statistics module.')

    parser.add_argument('--attr_dist',
                        type=str,
                        default=None,
                        choices=['genetic', 'geographic', 'size'],
                        help='Attribute to get distribution, if None skip distribution module.')

    parser.add_argument('--plot_type',
                        type=str,
                        default='heatmap',
                        choices=['bar', 'heatmap'],
                        help='Type of plot.')

    return parser


def config_download(parser):

    if not parser.parse_known_args()[0].download:
        return parser

    parser.add_argument('--ID_mapping_path',
                        type=str,
                        default=None,
                        help='Path to the directory where file ID mapping is saved.')
    parser.add_argument('--file_to_download',
                        type=str,
                        default=None,
                        help='File to download from drive.')
    parser.add_argument('--download_path',
                        type=str,
                        default=None,
                        help='Path to the download directory.')

    return parser


def set_config():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser = config_base(parser)
    parser = config_run_module(parser)
    parser = config_download(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args



