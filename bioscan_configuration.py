import torch
import datetime
import argparse


def config_base(parser):
    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    parser.add_argument('--date_time', type=str, default=timestamp,
                        help='Data & time of the experiment')
    parser.add_argument('--root', type=str, default='root_cz',
                        help='Set root of environment.')
    parser.add_argument('--level_name', nargs=2, default=None,
                        choices=['class', 'Insecta', 'order', 'Diptera', 'family', 'Cecidomyiidae'],
                        help='Two separate strings representing the taxonomic level and name to extract subset.')
    parser.add_argument('--group_level', type=str, default='order',
                        choices=['order', 'family', 'subfamily', 'tribe', 'genus', 'species', 'subspecies', 'name'],
                        help='Taxonomic group ranking for experiments.')
    parser.add_argument('--dataset_name', type=str, default='bioscan_dataset',
                        help='Name of the dataset.')
    parser.add_argument('--data_structure', type=str, default=None,
                        help='If using 1M_insect dataset structure, 113 chunks of data (part1:part113)?')
    parser.add_argument('--data_format', type=str, default='folder', choices=['folder', 'hdf5'],
                        help='Dataset format.')
    parser.add_argument('--read_all_samples', default=False, action='store_true',
                        help='IF reading all samples of metadata file?')

    return parser


def config_path(parser):

    parser.add_argument('--download_path', type=str, default=None,
                        help='Path to the download directory.')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the individual RGB images (if any).')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to the dataset file.")
    parser.add_argument('--hdf5_path', type=str, default=None,
                        help='Path to the HDF5 files of the Original images.')
    parser.add_argument('--metadata_path_1M', type=str, default=None,
                        help='Path to the metadata file of BIOSCAN-1M Insect.')
    parser.add_argument('--metadata_path_6M', type=str, default=None,
                        help='Path to the metadata file of BIOSCAN-6M Insect.')
    parser.add_argument('--resized_image_path', type=str, default=None,
                        help='Path to the resized images.')
    parser.add_argument('--resized_hdf5_path', type=str, default=None,
                        help='Path to the resized HDF5.')
    parser.add_argument('--cropped_image_path', type=str, default=None,
                        help="Path to the cropped images.")
    parser.add_argument('--cropped_hdf5_path', type=str, default=None,
                        help='Path to the HDF5 files of the cropped images.', )
    parser.add_argument('--resized_cropped_image_path', type=str, default=None,
                        help='Path to the resized cropped images.')
    parser.add_argument('--resized_cropped_hdf5_path', type=str, default=None,
                        help='Path to the HDF5 files of the resized cropped images.')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to save results.')

    return parser


def config_run_module(parser):
    parser.add_argument('--download', default=False, action='store_true',
                        help='IF to download dataset?')
    parser.add_argument('--lat_lon_map', default=False, action='store_true',
                        help='IF generate world map from longitude and latitude coordinates.')
    parser.add_argument('--attr_stat', type=str, default=None, choices=['genetic', 'habitat', 'size'],
                        help='Type of statistics, if None skip statistics module.')
    parser.add_argument('--attr_dist', type=str, default=None,
                        choices=['class', 'order', 'family', 'subfamily', 'genus', 'species',
                                 'country', 'province/state', 'coord-lat', 'coord-lon'],
                        help='Attribute to get distribution, if None skip distribution module.')
    parser.add_argument('--crop_image', default=False, action='store_true',
                        help='IF crop dataset images.')
    parser.add_argument('--resize_image', default=False, action='store_true',
                        help='IF resize dataset images.')
    parser.add_argument('--split_1M_Insect', default=False, action='store_true',
                        help='IF split dataset with stratifies class-based approach used in BIOSCAN-1M Insect paper.')
    parser.add_argument('--split_bioscan_clip', default=False, action='store_true',
                        help='IF split dataset with stratifies class-based approach used in BIOSCAN-CLIP paper.')
    parser.add_argument('--print_split_statistics', default=False, action='store_true',
                        help='IF print split statistics?')
    parser.add_argument('--plot_split_statistics', default=False, action='store_true',
                        help='IF plot split statistics?')
    parser.add_argument('--exp_1M_Insect', default=False, action='store_true',
                        help='IF run experiments performed and presented in BIOSCAN-1M Insect paper.')
    parser.add_argument('--exp_zero_shot_clustering', default=False, action='store_true',
                        help='IF run zero-shot clustering experiments.')
    parser.add_argument('--exp_barcode_bert', default=False, action='store_true',
                        help='IF run Barcode-BERT experiments.')
    parser.add_argument('--exp_bioscan_clip', default=False, action='store_true',
                        help='IF run BIOSCAN-CLIP experiments.')

    return parser


def config_download(parser):

    if not parser.parse_known_args()[0].download:
        return parser

    parser.add_argument('--ID_mapping_path', type=str, default=None,
                        help='Path to the directory where file ID mapping is saved.')
    parser.add_argument('--file_to_download', type=str, default=None,
                        help='File to download from drive.')

    return parser


def config_crop_resize(parser):

    if not parser.parse_known_args()[0].crop_image and not parser.parse_known_args()[0].resize_image:
        return parser

    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to the crop model checkpoint.')
    parser.add_argument('--crop_ratio', type=float, default=1.4,
                        help='Scale the bbox to crop large or small area.')
    parser.add_argument('--equal_extend', default=True, action='store_true',
                        help='IF extending cropped images in the height and width of same length?')
    parser.add_argument('--background_color_R', type=int, default=240,
                        help='Set the background color channel Red.')
    parser.add_argument('--background_color_G', type=int, default=240,
                        help='Set the background color channel Green.')
    parser.add_argument('--background_color_B', type=int, default=240,
                        help='Set the background color channel Blue.')
    parser.add_argument('--resize_dimension', type=int, default=256,
                        help='Dimension to resize images.')
    parser.add_argument('--save_binary', default=True, action='store_true',
                        help='IF save resized cropped images in binary mode in HDF5 to save space.')

    return parser


def config_exp_1m_insect(parser):

    if not parser.parse_known_args()[0].exp_1M_Insect:
        return parser

    parser.add_argument('--experiment_names', type=str,
                        default=['large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
                                 'large_insect_order', 'medium_insect_order', 'small_insect_order'],
                        help='Name of all experiments conducted with BIOSCAN_1M_Insect Dataset paper.')
    parser.add_argument('--exp_name', type=str, default='small_insect_order',
                        choices=['large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
                                 'large_insect_order', 'medium_insect_order', 'small_insect_order'],
                        help='Name of the experiment performed and presented in BIOSCAN_1M_Insect paper.')
    parser.add_argument('--loader', default=False, action='store_true', help='Whether to create dataloader?')
    parser.add_argument('--train', default=False, action='store_true', help='Whether to train the model?')
    parser.add_argument('--test', default=False, action='store_true', help='Whether to test the model?')
    parser.add_argument('--n_epochs', type=int, default=100, help='Maximum number of epochs to train model.')
    parser.add_argument('--epoch_decay', nargs='+', type=int, default=[20, 25])
    parser.add_argument('--mu', type=float, default=0.0001, help='weight decay parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate to use')
    parser.add_argument('--batch_size', type=int, default=32, help='default is 32')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--k', nargs='+', default=[1, 3, 5, 10], type=int,
                        help='value of k for computing the top-k loss and computing top-k accuracy')
    parser.add_argument('--no_transform', default=False, action='store_true',
                        help='IF using dataloader transformation?')
    parser.add_argument('--pretrained', default=True, action='store_true',
                        help='IF using pretrained weights.')
    parser.add_argument('--vit_pretrained', type=str, default=None,
                        help='Path to the checkpoint.')

    parser.add_argument('--best_model', type=str, default='',
                        help='directory where best results saved (inference/test mode).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Set the seed for reproducibility')
    parser.add_argument('--loss', type=str, choices=['CE', 'Focal'],
                        help='decide which loss to use during training', default='CE')
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                            'mobilenet_v2', 'inception_v3', 'alexnet', 'squeezenet',
                                            'shufflenet', 'wide_resnet50_2', 'wide_resnet101_2',
                                            'vgg11', 'mobilenet_v3_large', 'mobilenet_v3_small',
                                            'inception_resnet_v2', 'inception_v4', 'efficientnet_b0',
                                            'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                            'efficientnet_b4', 'vit_base_patch16_224'], default='resnet50',
                        help='choose the model you want to train on')
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available(),
                        help='IF using GPU.')
    return parser


def config_split_1M_Insect(parser):

    if not parser.parse_known_args()[0].split_1M_Insect:
        return parser

    parser.add_argument('--method', type=str, default='stratified_class_based', help='Split mechanism approach.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train ratio for train set.')
    parser.add_argument('--validation_ratio', type=float, default=0.1, help='Validation ratio for train set.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio for train set.')
    parser.add_argument('--chunk_length', type=int, default=10000, help='Chunk length: number of images of each patch.')
    parser.add_argument('--chunk_num', type=int, default=0, help='set the data chunk number.')
    parser.add_argument('--max_num_sample', type=int, default=50000, choices=[50000, 200000],
                        help='Number of samples of each subset.')
    return parser


def config_split_bioscan_clip(parser):

    if not parser.parse_known_args()[0].split_bioscan_clip:
        return parser

    parser.add_argument('--method', type=str, default='', help='Split mechanism approach.')

    return parser


def config_zero_shot_clustering(parser):

    if not parser.parse_known_args()[0].exp_zero_shot_clustering:
        return parser

    parser.add_argument('--experiment_names', type=str,
                        default=['', '', ''],
                        help='Name of all experiments conducted with zero-shot clustering approach.')

    return parser


def config_barcode_bert(parser):

    if not parser.parse_known_args()[0].exp_barcode_bert:
        return parser

    parser.add_argument('--experiment_names', type=str,
                        default=['', '', ''],
                        help='Name of all experiments conducted with Barcode-BERT approach.')

    return parser


def config_bioscan_clip(parser):

    if not parser.parse_known_args()[0].exp_bioscan_clip:
        return parser

    parser.add_argument('--experiment_names', type=str,
                        default=['', '', ''],
                        help='Name of all experiments conducted with BIOSCAN-CLIP approach.')

    return parser


def set_config():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser = config_base(parser)
    parser = config_path(parser)
    parser = config_run_module(parser)
    parser = config_crop_resize(parser)
    parser = config_split_1M_Insect(parser)
    parser = config_split_bioscan_clip(parser)
    parser = config_exp_1m_insect(parser)
    parser = config_zero_shot_clustering(parser)
    parser = config_barcode_bert(parser)
    parser = config_bioscan_clip(parser)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args



