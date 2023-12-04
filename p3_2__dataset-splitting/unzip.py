import os
import shutil


def unpack_archive(file_path, destination_path):
    if os.path.exists(file_path):
        shutil.unpack_archive(file_path, destination_path, 'zip')
        print(f'The folder has been unzipped to "{destination_path}".')
    else:
        print(f'The folder at "{file_path}" does not exist.')


def move_folder(cur_path, new_path):
    if os.path.exists(cur_path):
        shutil.move(cur_path, new_path)
        print(f'The folder has been moved to "{new_path}".')
    else:
        print(f'The folder at "{cur_path}" does not exist.')


def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'The folder at "{path}" has been removed.')
    else:
        print(f'The folder at "{path}" does not exist.')


def main():
    zip_file_path = "D:/NEU/CSYE7105/archive.zip"
    extract_folder = "D:/NEU/CSYE7105/new-plant-diseases-dataset"
    nested = "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    # Unzip
    unpack_archive(zip_file_path, extract_folder)

    # Move and rename current train and test
    move_folder(extract_folder + nested + "/train", extract_folder + "/train-old")
    move_folder(extract_folder + nested + "/valid", extract_folder + "/valid")
    move_folder(extract_folder + "/test/test", extract_folder + "/test-old")

    # Remove blank folders
    remove_folder(extract_folder + "/New Plant Diseases Dataset(Augmented)")
    remove_folder(extract_folder + "/test")


main()
