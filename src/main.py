import os
import numpy as np
from src.preprocessing import read_result_file, split_train_and_test, load_and_preprocess_image
from src.cnn import SimpleCNN

result_path = os.path.join(os.path.dirname(__file__), "../assets/results/G1020.csv")

image_ids, labels = read_result_file(result_path)
indices = list(range(len(image_ids)))

train_indices, test_indices = split_train_and_test(indices, labels, ratio=0.2)

train_images = np.array([load_and_preprocess_image(os.path.join(os.path.dirname(__file__), "../assets/images/" + image_ids[i])) for i in train_indices])
train_labels = np.array([labels[i] for i in train_indices]).reshape(-1, 1)

test_images = np.array([load_and_preprocess_image(os.path.join(os.path.dirname(__file__), "../assets/images/" + image_ids[i])) for i in test_indices])
test_labels = np.array([labels[i] for i in test_indices]).reshape(-1, 1)

train_images = train_images.transpose(0, 3, 1, 2)
test_images = test_images.transpose(0, 3, 1, 2)

cnn = SimpleCNN()

epochs = 15
learning_rate = 0.01
batch_size = 4

for epoch in range(epochs):
    for i in range(0, len(train_images), batch_size):
        X_batch = train_images[i:i + batch_size]
        y_batch = train_labels[i:i + batch_size]

        grads = cnn.gradient(X_batch, y_batch)
        for key in cnn.params.keys():
            cnn.params[key] -= learning_rate * grads[key]

    loss = cnn.loss(train_images, train_labels)
    accuracy = cnn.accuracy(train_images, train_labels)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

test_accuracy = cnn.accuracy(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")