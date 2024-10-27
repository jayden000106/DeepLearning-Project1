import csv
import random
import numpy as np
from PIL import Image

def read_result_file(filepath):
    image_ids, labels = [], []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            image_ids.append(row[0])
            labels.append(int(row[1]))

    return image_ids, labels


def split_train_and_test(indices, ratio):
    test_size = int(len(indices) * ratio)
    random.shuffle(indices)

    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    return train_indices, test_indices


def load_and_preprocess_image(file_path, target_size = (128, 128)):
    image = Image.open(file_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image) / 255.0