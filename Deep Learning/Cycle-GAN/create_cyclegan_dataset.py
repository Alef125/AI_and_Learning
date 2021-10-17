"""Create datasets for training and testing."""
import csv
import os
import random
import cyclegan_datasets_info


def files_with_this_suffix(folder_name, full_dir=True, suffix=".jpg"):
    """
    :param folder_name: The full path of the folder.
    :param full_dir: Whether to return the full path or not.
    :param suffix: Filter by suffix.
    :return: The list of file_names in the folder with given suffix.
    """
    files_list = os.listdir(folder_name)
    files_list_with_this_suffix = []
    if full_dir:
        for item in files_list:
            if item.endswith(suffix):
                files_list_with_this_suffix.append(os.path.join(folder_name, item))
    else:
        for item in files_list:
            if item.endswith(suffix):
                files_list_with_this_suffix.append(item)
    return files_list_with_this_suffix


def create_dataset_csv(dataset_name, do_shuffle):
    """
    :param dataset_name: The name of the dataset in cyclegan_dataset.
    :param do_shuffle: Whether to shuffle images when creating the dataset.
    :return: Nothing :)). A CSV information file is generated in the path that which pairs images in groups A & B
    """
    image_path_a = cyclegan_datasets_info.get_dataset_info(dataset_name, 'PATH_TO_GROUP_A')
    image_path_b = cyclegan_datasets_info.get_dataset_info(dataset_name, 'PATH_TO_GROUP_B')
    images_type = cyclegan_datasets_info.get_dataset_info(dataset_name, 'IMAGE_TYPE')
    images_list_a = files_with_this_suffix(image_path_a, full_dir=True, suffix=images_type)
    images_list_b = files_with_this_suffix(image_path_b, full_dir=True, suffix=images_type)

    output_path = cyclegan_datasets_info.get_dataset_info(dataset_name, 'PATH_TO_CSV')
    num_rows = cyclegan_datasets_info.get_dataset_info(dataset_name, 'SIZE')

    all_data_tuples = []
    for i in range(num_rows):
        all_data_tuples.append((images_list_a[i % len(images_list_a)],
                                images_list_b[i % len(images_list_b)]))
    if do_shuffle:
        random.shuffle(all_data_tuples)

    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))


""" --------- main --------- """
create_dataset_csv(dataset_name="FERG_train", do_shuffle=False)
