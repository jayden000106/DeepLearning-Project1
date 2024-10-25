import os
from src.preprocessing import read_result_file, split_train_and_test
from src.cnn import CNN

result_path = os.path.join(os.path.dirname(__file__), "../assets/results/G1020.csv")

image_ids, labels = read_result_file(result_path)
indices = list(range(len(image_ids)))

train_indices, test_indices = split_train_and_test(indices, 0.2)

cnn = CNN(image_ids, labels, train_indices, test_indices)
print(cnn.train_indices)
print(cnn.test_indices)