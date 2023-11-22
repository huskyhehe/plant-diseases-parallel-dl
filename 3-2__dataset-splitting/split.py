import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(dataset_path, test_size):
    train_old_dir = dataset_path + "/train-old"

    train_dir = dataset_path + "/train"
    test_dir = dataset_path + "/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create new 'train' and 'test' directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all labels/directories in the 'train-old' set
    labels = os.listdir(train_old_dir)

    for label in labels:
        # Create corresponding label directories in new 'train' and 'test' sets
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

        # List all files under each label in the 'train-old' set
        files = os.listdir(os.path.join(train_old_dir, label))

        # Split files to keep 90% in new 'train' and 10% in new 'test'
        # Adjust the test_size to 10% of the original dataset, which is 12.5% of the training set
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        # Copy selected files to the new 'train' and 'test' directories
        for file in train_files:
            src_path = os.path.join(train_old_dir, label, file)
            dest_path = os.path.join(train_dir, label, file)
            shutil.copy(src_path, dest_path)

        for file in test_files:
            src_path = os.path.join(train_old_dir, label, file)
            dest_path = os.path.join(test_dir, label, file)
            shutil.copy(src_path, dest_path)


def get_image_count(subset):
    count = 0
    for category in os.listdir(subset):
        for file in os.listdir(os.path.join(subset, category)):
            if file.lower().endswith('.jpg'):
                count += 1
    return count


def main():
    dataset_path = "D:/NEU/CSYE7105/new-plant-diseases-dataset"
    train_old_count = get_image_count(dataset_path + "/train-old")
    print(f"Train set count before splitting: {train_old_count}")

    test_size = (1 / 0.8) * 0.1
    split_dataset(dataset_path, test_size)

    train_count = get_image_count(dataset_path + "/train")
    print(f"Train set count after splitting: {train_count}")


main()
