# Filename: transfer_learning_cnn_util.py
# Dependencies: transfer_learning_cnn_util.py, collections, keras, math, os, pathlib, random, shutil, sys, tensorflow, warnings
# Author: Jean-Michel Boudreau
# Date: October 21, 2019

'''
Module containing all function required by transfer_learning_cnn.py driver.
'''

# Import libraries
from keras.preprocessing.image import ImageDataGenerator
import math
import os
import pathlib
import random
import shutil
import sys
from collections import defaultdict
from os.path import abspath, join, basename, exists
import tensorflow as tf
import warnings
warnings.simplefilter("ignore")  

'''
Queries the user for whether data exists and is already split. All cases are 
handled:

If data does not exist, asks to download the flowers data set. If data exists
and is not split, asks for absolute path of directory containing all images. 
In either case, a query for the user is given asking to split data into 
training and validation subsets or to simply train on all images.

If data exists and is split, asks for absolute path of training and validation
directories containing subdirectories (named after the classes).
 '''
def clean_data_exists(img_height, img_width, batch_size):
    query_data_cleaned = query_yes_no("Data exists in seperate training and validation directories?")\
    # Yes
    if query_data_cleaned == True:
        # User provides training and validation directory manually
        training_dir = input("Provide the absolute path of the training data: \n")
        validation_dir = input("Provide the absolute path of the validation data: \n")
        # Generates batches of data from training directory
        train_generator = datagen_flow(
                training_dir,
                img_height,
                img_width,
                batch_size)
        # Generates batches of data from validation directory
        validation_generator = datagen_flow(
                validation_dir,
                img_height,
                img_width,
                batch_size)
    # No
    elif query_data_cleaned == False:
        # Query to download and/or specify absolute path of directory 
        # containing dataset
        data_dir = get_data_path()
        # Construct train_generator and validaiton_generator vars for use in
        # CNN
        train_generator, validation_generator = query_data_split(data_dir,
                                                                 batch_size,
                                                                 img_height,
                                                                 img_width)
    # Return generator for training and validation directories each containing
    # subdirectories filled with images for each class
    return train_generator, validation_generator


'''
Creates target directory generator (for either training or validation)
'''
def datagen_flow(target_dir, img_height, img_width, batch_size):
                # Rescale images. The 1./255 is to convert from uint8 to float32 in range [0,1].
                target_datagen = ImageDataGenerator(rescale=1./255)
                # Generates batches of data from target directory
                target_generator = target_datagen.flow_from_directory(
                # target_directory
                directory = target_dir,
                # Size of batches
                batch_size=batch_size,
                # Shuffle data
                shuffle=True,
                # All images will be resized
                target_size=(img_height, 
                             img_width))
                # Return generator for directory containing subdirectories 
                # filled with images for each class
                return target_generator

'''
Queries whether to download flower photos dataset from Google or if a dataset 
exists. If the latter, asks to provide the absolute path of the unsplit
dataset. Returns  the dataset directory path in the format of the system's path
flvaor using pathlib
'''
def get_data_path():
    # Query user to download flower photos dataset from Google so as to not be 
    # invasive
    query_dl = query_yes_no("Download archive of creative-commons licensed flower photos from Google?")
    # User agrees to download dataset
    if query_dl == True:
        # Download dataset from Google
        data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                                 fname='flower_photos', untar=True)
        # Absolute path of dataset
        data_dir = pathlib.Path(data_dir)
        # Remove tar file
        (data_dir.parent / "flower_photos.tar.gz").unlink()
    elif query_dl == False:
        # User specified absolute path of dataset
        data_dir = pathlib.Path(
                input("Provide the absolute path of the directory containing the subdirectories (named after classes) for images: \n"))
    # Return path of dataset directory
    return data_dir

'''
Query to split data into training and, (if applicable) validation 
subsets or to train on all images.
'''
def query_data_split(data_dir, batch_size, img_height, img_width, 
                     validation_pct=0.25):
    # Query user if automatic data splitting into trainining/validation subsets via
    # stratified sampling 
    query_split_data = query_yes_no("Split images into stratified samples for training and validation? This will replace the original dataset directory with a new one")
    # User agrees to split data via stratified fashion
    if query_split_data == True:
        # Path to directory for stratified data
        split_data_dir_name = data_dir.name + '_split'
        split_data_dir = data_dir.parents[0] / split_data_dir_name
        # Path to subdirectory for training data
        training_dir_path = split_data_dir /'training'
        training_dir = str(training_dir_path)
        # Path to subdirectory for validation data
        validation_dir =  str(split_data_dir / 'validation')
        # Split dataset into respective subsets
        split_dataset_into_test_and_train_sets(data_dir, 
                                               training_dir, 
                                               validation_dir, 
                                               validation_pct)
        # Generates batches of data from validation directory
        validation_generator = datagen_flow(
                validation_dir,
                img_height,
                img_width,
                batch_size)
        # Remove emptied-out original directory
        shutil.rmtree(data_dir)
    # User does not agree to split data in stratified fashion
    elif query_split_data == False:
        # Query user if all data should be used to train
        query_train_all = query_yes_no("Train on all images?")
        # All data should be used
        if query_train_all == True:
            # Use all data for training
            training_dir_path = data_dir
            training_dir = str(training_dir_path)
            # Do not use validation_generator
            validation_generator = None
        # All data should not be used
        elif query_train_all == False:
                sys.exit("No data to train on. Exiting... \n")
    # Generates batches of data from training directory
    train_generator = datagen_flow(
            training_dir,
            img_height,
            img_width,
            batch_size)
     # Return generator for training and validation directories each containing
     # subdirectories filled with images for each class
    return train_generator, validation_generator

'''
A helper function to query [Y/n] inputs from user
'''
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

'''
Splits a specified directory, containing subdirectories named after classes 
that respectively hold images of their class, into training and validation 
directories arranged in the same manner and saves this directory in the parent 
directory to the one specified. Deletes the specified directory following the
split
'''         
def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct,
                                           stratify=True, seed=None):
    prev_state = None
    if seed:
        prev_state = random.getstate()
        random.seed(seed)

    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory", testing_data_dir)
    else:
        print(
            testing_data_dir, "not empty, did not remove contents")

    if training_data_dir.count('/') > 1:
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory", training_data_dir)
    else:
        print(training_data_dir, "not empty, did not remove contents")

    files_per_class = defaultdict(list)

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = basename(subdir)

        # Don't create a subdirectory for the root directories
        if category_name in map(basename, [all_data_dir, training_data_dir, testing_data_dir]):
            continue

        # filtered past top-level dirs, now we're in a category dir
        files_per_class[category_name].extend([join(abspath(subdir), file) for file in files])

    # keep track of train/validation split for each category
    split_per_category = defaultdict(lambda: defaultdict(int))
    # create train/validation directories for each class
    class_directories_by_type = defaultdict(lambda: defaultdict(str))
    for category in files_per_class.keys():
        training_data_category_dir = join(training_data_dir, category)
        if not exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)
        class_directories_by_type['train'][category] = training_data_category_dir

        testing_data_category_dir = join(testing_data_dir, category)
        if not exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)
        class_directories_by_type['validation'][category] = testing_data_category_dir

    if stratify:
        for category, files in files_per_class.items():

            random.shuffle(files)
            last_index = math.ceil(len(files) * testing_data_pct)
            # print('files upto index {} to val'.format(last_index))
            # print('category {} train/validation: {}/{}'.format(category, len(files[:last_index]),
            #                                                    len(files[last_index:])))
            for file in files[:last_index]:
                testing_data_category_dir = class_directories_by_type['validation'][category]
                # print('moving {} to {}'.format(file, join(testing_data_category_dir, basename(file))))
                shutil.move(file, join(testing_data_category_dir, basename(file)))
                split_per_category['validation'][category] += 1
            for file in files[last_index:]:
                training_data_category_dir = class_directories_by_type['train'][category]
                # print('moving {} to {}'.format(file, join(training_data_category_dir, basename(file))))
                shutil.move(file, join(training_data_category_dir, basename(file)))
                split_per_category['train'][category] += 1

    else:  # not stratified, move a fraction of all files to validation
        files = []
        for file_list in files_per_class.values():
            files.extend(file_list)

        random.shuffle(files)
        last_index = math.ceil(len(files) * testing_data_pct)
        for file in files[:last_index]:
            category = get_containing_folder_name(file)
            directory = class_directories_by_type['validation'][category]
            shutil.move(file, join(directory, basename(file)))
            split_per_category['validation'][category] += 1
        for file in files[last_index:]:
            category = get_containing_folder_name(file)
            directory = class_directories_by_type['train'][category]
            shutil.move(file, join(directory, basename(file)))
            split_per_category['train'][category] += 1

    if seed:
        random.setstate(prev_state)
    return split_per_category