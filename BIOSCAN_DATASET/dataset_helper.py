import numpy as np
import tarfile
import zipfile
import os
import shutil
import h5py
import pandas as pd
import io
from PIL import Image
import csv
from pathlib import Path


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_directory(path):
    shutil.rmtree(path, ignore_errors=False, onerror=None)

def path_exist(path):
    directory_path = Path(path)
    if directory_path.exists():
        print(f"Directory {path} does not exist")
    if not os.path.exists(path):
        print(f"Directory {path} does not exist")
    return directory_path.exists()


def file_exist(file_path):
    file_path_obj = Path(file_path)
    if file_path_obj.exists() and file_path_obj.is_file():
        print(f"File {file_path} exists")
    else:
        print(f"File {file_path} does not exist")
    return file_path_obj.exists() and file_path_obj.is_file()

def create_zip(source_folder=None, output_zip=None, package_type=None, part_name=None):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(
                    f"bioscan/images/{package_type}/{part_name}", file)
                zipf.write(file_path, arcname=arcname)


def extract_zip(zip_file=None, path=None):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)

def create_hdf5(date_time, dataset_name='', path='', data_typ='Original Full Size', author='Zahra Gharaee'):

    with h5py.File(path, 'w') as hdf5:
        dataset = hdf5.create_group(dataset_name)
        dataset.attrs['Description'] = f'BIOSCAN_1M Insect Dataset: {data_typ} Images'
        dataset.attrs['Copyright Holder'] = 'CBG Photography Group'
        dataset.attrs['Copyright Institution'] = 'Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)'
        dataset.attrs['Photographer'] = 'CBG Robotic Imager'
        dataset.attrs['Copyright License'] = 'Creative Commons-Attribution Non-Commercial Share-Alike'
        dataset.attrs['Copyright Contact'] = 'collectionsBIO@gmail.com'
        dataset.attrs['Copyright Year'] = '2021'
        dataset.attrs['Author'] = author
        dataset.attrs['Date'] = date_time

    return dataset


def write_in_hdf5(hdf5, image, image_name, image_dir=None, save_binary=False):
    """
    This function writes an image in a HDF5 file.
    :param hdf5: HDF5 file to write image in.
    :param image: Image as data array.
    :param image_name: Name which image is archived with.
    :param image_dir: Directory where the image is saved.
    :param save_binary: If save as binary data (compressed) to save space?
    :return:
    """

    if save_binary:
        if image_dir is not None:
            with open(image_dir, 'rb') as img_f:
                binary_data = img_f.read()
            image_data = np.asarray(binary_data)
        else:
            binary_data_io = io.BytesIO()
            image.save(binary_data_io, format='JPEG')
            binary_data = binary_data_io.getvalue()
            image_data = np.frombuffer(binary_data, dtype=np.uint8)
    else:
        image_data = np.array(image)

    hdf5.create_dataset(image_name, data=image_data)


def read_from_hdf5(hdf5, image_name, saved_as_binary=False):
    """
    This function reads an image from a HDF5 file.
    :param hdf5: The Hdf5 file to read from.
    :param image_name: The image to read.
    :param saved_as_binary: If data is saved as binary?
    :return:
    """

    if saved_as_binary:
        data = np.array(hdf5[image_name])
        image = Image.open(io.BytesIO(data))
    else:
        data = np.asarray(hdf5[image_name])
        image = Image.fromarray(data)

    return image


def resize_image(input_file, output_file, resize_dimension=256):
    command = f'convert "{input_file}" -resize x{resize_dimension} "{output_file}"'
    os.system(command)


def extract_prefix(filename):
    parts = filename.split('.')
    return parts[0]


def extract_format(filename):
    parts = filename.split('.')
    return parts[-1]


def keep_row_by_item(df, items_to_keep, column=None):
    """ keep based on specified item list (items) """
    df_updated = df[df[column].isin(items_to_keep)]
    df_updated.reset_index(inplace=True, drop=True)
    return df_updated


def make_tsv(df, name=None, path=None):
    df_ = pd.DataFrame(df)
    df_.reset_index(inplace=True, drop=True)
    df_.to_csv(os.path.join(path, name), sep='\t', index=False)


def read_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    return df


def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def read_tsv_large(tsv_file):
    try:
        csv.field_size_limit(10 ** 9)
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False, quoting=csv.QUOTE_NONE)
        print(df.head())
        return df
    except Exception as error:
        print(f"Reading the file:\n{tsv_file} gives the error:\n{error}")


def sort_dict(data_dict):
    """ data_dict is a dictionary of keys (strings) and values (numbers),
    and you want to sort it based on the values (numbers). """
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_data_dict


def sort_dict_list(data_dict):
    """data_dict is a dictionary of keys (strings) and values are lists or collections of samples.
    It calculates the number of samples per class, then sorts the indices based on these counts in descending order.
    sorted_data_dict where the classes are sorted based on the number of samples they contain."""
    n_sample_per_class = [len(class_samples) for class_samples in list(data_dict.values())]
    indexed_list = list(enumerate(n_sample_per_class))
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    original_indices_sorted = [x[0] for x in sorted_list]

    class_names = list(data_dict.keys())
    sorted_class_names = [class_names[ind] for ind in original_indices_sorted]

    sorted_data_dict = {}
    for name in sorted_class_names:
        sorted_data_dict[name] = data_dict[name]

    return sorted_data_dict
