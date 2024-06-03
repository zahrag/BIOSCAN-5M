import shutil
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def copy_files_to_subfolder_based_on_prefix(folder):
    source_base_path = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized'
    target_base_path = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized/new_org'
    current_folder = os.path.join(source_base_path, folder)
    if os.path.exists(current_folder) and os.path.isdir(current_folder):
        for filename in os.listdir(current_folder):
            file_path = os.path.join(current_folder, filename)
            if os.path.isfile(file_path):
                subfolder = filename[:2]
                subfolder_path = os.path.join(target_base_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                shutil.copy(file_path, subfolder_path)

def main():
    source_base_path = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized'
    target_base_path = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized/new_org'
    os.makedirs(target_base_path, exist_ok=True)

    folders = [f'part{i}' for i in range(1, 115)]
    pool = Pool(processes=cpu_count())
    for _ in tqdm(pool.imap_unordered(copy_files_to_subfolder_based_on_prefix, folders), total=len(folders)):
        pass

    pool.close()
    pool.join()

    print("Done")

if __name__ == "__main__":
    main()
