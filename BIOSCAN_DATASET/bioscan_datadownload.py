
import os
import wget
from dataset_helper import make_directory
import itertools
import gdown


def read_id_mapping(id_mapping_path=""):
    """ Read ID-Mapping-File """
    if not os.path.isfile(id_mapping_path):
        print(f"ID Mapping File is not available to read in:\n{id_mapping_path}")
        return
    file_id_mapping = {}
    with open(id_mapping_path) as fp:
        for i, line in enumerate(fp):
            a_string = line.strip()
            info = [word for word in a_string]
            sep_index = info.index(':')
            file_id_mapping[''.join(info[:sep_index])] = ''.join(info[sep_index + 1:])
    return file_id_mapping


def wget_download(parent_folder_id, file_id, file_name, download_path=""):
    """ Set download url of the files on the drive """
    make_directory(download_path)
    url = f"https://drive.google.com/uc?id={file_id}"
    wget.download(url, out=f"{download_path}/{file_name}")


def gdown_download(file_id, file_name, download_path=""):
    """ Download file from drive using gdown """
    make_directory(download_path)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, f"{download_path}/{file_name}")


def make_download(configs):
    """
    This function downloads the files of the BIOSCAN_1M_Insect dataset from GoogleDrive.
    :param configs: Configurations.
    :return:
    """

    if not configs['download']:
        print("No file is selected to download!")
        return

    id_mapping_path = configs['ID_mapping_path']
    download_path = configs['download_path']
    file_selected = configs["file_to_download"]

    parent_main = "1ft17GpcC_xDhx5lOBhYAEtox0sdZchjA"

    parent_original_full = "1li1UCT483OXb7AO_-6SOecTvIfrlKhMz"
    original_full_zip = ['BIOSCAN_5M_original_full.01.zip',
                         'BIOSCAN_5M_original_full.02.zip',
                         'BIOSCAN_5M_original_full.03.zip',
                         'BIOSCAN_5M_original_full.04.zip',
                         'BIOSCAN_5M_original_full.05.zip']

    parent_cropped = "1M2riGzS3x6d2ARUHlWXPOdiFKH8rRXxN"
    cropped_zip = ['BIOSCAN_5M_cropped.01.zip',
                   'BIOSCAN_5M_cropped.02.zip']

    parent_metadata = "1TLVw0P4MT_5lPrgjMCMREiP8KW-V4nTb"
    metadata_zip = ['BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip']

    parent_original_256 = "1YWnmoraPAGcQKwL9Y7F4WC1o60pMQhO8"
    original_256_zip = ['BIOSCAN_5M_original_256.zip',
                        'BIOSCAN_5M_original_256_pretrain.zip',
                        'BIOSCAN_5M_original_256_train.zip',
                        'BIOSCAN_5M_original_256_eval.zip']

    parent_cropped_256 = "1tqpzBoTAfENTY9Ndx_h9DFl1XzHHixg_"
    cropped_256_zip = ['BIOSCAN_5M_cropped_256.zip',
                       'BIOSCAN_5M_cropped_256_pretrain.zip',
                       'BIOSCAN_5M_cropped_256_train.zip',
                       'BIOSCAN_5M_cropped_256_eval.zip']

    parent_bbox = "1i6mSf5P6nmc228RUOfVwer6TVjZXUzeP"
    bbox_tsv = ['BIOSCAN_5M_Insect_bbox.tsv']

    files_list = [metadata_zip, original_full_zip, cropped_zip, original_256_zip, cropped_256_zip, bbox_tsv]
    files_list = list(itertools.chain(*files_list))

    if file_selected not in files_list:
        raise RuntimeError(f"File: {file_selected} is not available for download!")

    file_id_mapping = read_id_mapping(id_mapping_path=id_mapping_path)

    if file_selected in original_256_zip:
        parent_folder_id = parent_original_256

    elif file_selected in cropped_256_zip:
        parent_folder_id = parent_cropped_256

    elif file_selected in metadata_zip:
        parent_folder_id = parent_metadata

    elif file_selected in original_full_zip:
        parent_folder_id = parent_original_full

    elif file_selected in cropped_zip:
        parent_folder_id = parent_cropped

    elif file_selected in bbox_tsv:
        parent_folder_id = parent_bbox

    # wget_download(parent_folder_id, file_id_mapping[file_selected], file_selected, download_path=download_path)
    gdown_download(file_id_mapping[file_selected], file_selected, download_path=download_path)




