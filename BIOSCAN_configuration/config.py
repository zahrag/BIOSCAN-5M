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
                        default='order',
                        choices=['order', 'family', 'subfamily', 'tribe', 'genus', 'species', 'subspecies', 'name'],
                        help='Taxonomic group ranking.')

    parser.add_argument('--metadata',
                        type=str,
                        default='',
                        help='Path to the metadata file of the dataset.')

    return parser


def config_run_module(parser):

    parser.add_argument('--download',
                        default=False,
                        action='store_true',
                        help='IF to download dataset?')

    parser.add_argument('--attr_stat',
                        type=str,
                        default=None,
                        choices=['genetic', 'habitat', 'size'],
                        help='Type of statistics, if None skip statistics module.')

    parser.add_argument('--attr_dist',
                        type=str,
                        default=None,
                        choices=['class', 'order', 'family', 'subfamily', 'genus', 'species',
                                 'country', 'province/state', 'coord-lat', 'coord-lon'],
                        help='Attribute to get distribution, if None skip distribution module.')

    parser.add_argument('--lat_lon_map',
                        default=False,
                        action='store_true',
                        help='IF generate world map from longitude and latitude coordinates.')

    parser.add_argument('--split',
                        default=False,
                        action='store_true',
                        help='IF split dataset, if None skip split module.')

    parser.add_argument('--print_split_statistics',
                        default=False,
                        action='store_true',
                        help='IF print split statistics, if None skip module.')

    parser.add_argument('--plot_split_statistics',
                        default=False,
                        action='store_true',
                        help='IF plot split statistics,if None skip module.')

    parser.add_argument('--print_split_distribution',
                        default=False,
                        action='store_true',
                        help='IF get split distributions, if None skip module.')

    parser.add_argument('--plot_type',
                        type=str,
                        default='heatmap',
                        choices=['bar', 'heatmap'],
                        help='Type of plot.')

    parser.add_argument('--bioscan_bert',
                        default=False,
                        action='store_true',
                        help='IF run bioscan_bert experiments, if None skip module.')

    parser.add_argument('--bioscan_clustering',
                        default=False,
                        action='store_true',
                        help='IF run zero-shot clustering experiments, if None skip module.')

    parser.add_argument('--bioscan_clip',
                        default=False,
                        action='store_true',
                        help='IF bioscan-clip experiments, if None skip module.')

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


def config_split(parser):

    if not parser.parse_known_args()[0].split:
        return parser

    parser.add_argument(
        "-s",
        "--min-species-size",
        dest="min_species_size",
        type=int,
        help="minimum number of samples in species to be considered for training",
        default=10,
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
    parser.add_argument("-o", "--output", dest="output", type=str, help="path to output TSV file",
                        default='data/BioScan_1M/splits.tsv')

    return parser


def config_bioscan_bert(parser):

    if not parser.parse_known_args()[0].barcode_bert:
        return parser


def config_bioscan_clustering(parser):

    if not parser.parse_known_args()[0].bioscan_clustering:
        return parser


def config_bioscan_clip(parser):
    if not parser.parse_known_args()[0].bioscan_clip:
        return parser


def set_config():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser = config_base(parser)
    parser = config_run_module(parser)
    parser = config_download(parser)
    parser = config_split(parser)

    parser = config_bioscan_bert(parser)
    parser = config_bioscan_clustering(parser)
    parser = config_bioscan_clip(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args



