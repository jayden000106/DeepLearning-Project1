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


def split_train_and_test(indices, labels, ratio):
    from collections import defaultdict

    label_to_indices = defaultdict(list)
    for idx in indices:
        label = labels[idx]
        label_to_indices[label].append(idx)

    train_indices = []
    test_indices = []

    for label, idxs in label_to_indices.items():
        random.shuffle(idxs)
        split = int(len(idxs) * ratio)
        test_indices.extend(idxs[:split])
        train_indices.extend(idxs[split:])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return train_indices, test_indices


def load_and_preprocess_image(file_path, target_size = (128, 128)):
    image = Image.open(file_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image) / 255.0