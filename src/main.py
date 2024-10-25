import os
from src.preprocessing import read_result_file

result_path = os.path.join(os.path.dirname(__file__), "../assets/results/G1020.csv")

image_ids, labels = read_result_file(result_path)
print(image_ids)
print(labels)