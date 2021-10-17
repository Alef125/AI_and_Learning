"""Create datasets for training"""
import csv
import os
import random
import json


def get_files_with_this_suffix(folder_name, full_dir=True, suffix=".jpg"):
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


def make_random_pairs_for_images(classified_images_dict, num_rows):
    all_data_tuples = []
    classes = classified_images_dict.keys()
    classes_length = {the_class: len(classified_images_dict[the_class]) for the_class in classes}
    classes_numbers = dict(zip(classes, range(len(classes))))  # to save numerical
    for i in range(num_rows):  # ToDo: with for, some images may not be seen. make it wiser
        class1, class2 = random.sample(classes, 2)
        all_data_tuples.append((classified_images_dict[class1][i % classes_length[class1]],
                                classes_numbers[class1],
                                classified_images_dict[class2][i % classes_length[class2]],
                                classes_numbers[class2]))
    return all_data_tuples


def create_train_dataset_csv(dataset_name, classes, classified_train_images_path, images_type,
                             path_to_csv, dataset_size):
    """
    :param dataset_name: The name of the dataset in cyclegan_dataset.
    :param classes: list of classes for this dataset
    :param classified_train_images_path: directory that there is num_classes folders classified containing images
    :param path_to_csv: where csv file should be written
    :param images_type: image_type, .jpg or .png
    :param dataset_size: number of rows you want to be in the csv file (number of pairs)
    :return: Nothing :)). A CSV information file is generated in the path that which pairs images in groups A & B
    Remark: It is assumed that images in different classes are not pair (from a single character)
    """
    classified_images_list = {the_class: get_files_with_this_suffix(
        folder_name=os.path.join(classified_train_images_path, the_class),
        full_dir=True,
        suffix=images_type
    )
        for the_class in classes}

    all_data_tuples = make_random_pairs_for_images(classified_images_list, num_rows=dataset_size)

    with open(path_to_csv, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))

    print("csv completely written for dataset %s" % dataset_name)


def create_test_csv(train_images_path, images_type, path_to_csv):
    images_list = get_files_with_this_suffix(folder_name=train_images_path,
                                             full_dir=True,
                                             suffix=images_type)
    with open(path_to_csv, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for image_name in images_list:
            csv_writer.writerow(list([image_name]))


""" --------- main --------- """
configs_filename = "./config/configs.json"
with open(configs_filename) as configs_file:
    configs = json.load(configs_file)
create_train_dataset_csv(dataset_name="FERG_train",
                         classes=configs['classes'], classified_train_images_path=configs['path_to_train_images'],
                         images_type=configs['image_type'], path_to_csv=configs['path_to_train_csv'],
                         dataset_size=configs['dataset_size'])
create_test_csv(train_images_path=configs['path_to_test_images'],
                images_type=configs['image_type'],
                path_to_csv=configs['path_to_test_csv'])
