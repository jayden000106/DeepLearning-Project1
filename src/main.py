import os
from src.preprocessing import read_result_file, split_train_and_test

result_path = os.path.join(os.path.dirname(__file__), "../assets/results/G1020.csv")

image_ids, labels = read_result_file(result_path)
indices = list(range(len(image_ids)))

train_indices, test_indices = split_train_and_test(indices, 0.2)

print(train_indices)
print(test_indices)