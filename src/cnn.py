class CNN:
    def __init__(self, image_ids, labels, train_indices, test_indices):
        self.image_ids = image_ids
        self.labels = labels
        self.train_indices = train_indices
        self.test_indices = test_indices