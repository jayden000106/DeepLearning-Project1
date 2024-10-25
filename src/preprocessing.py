import csv

def read_result_file(filepath):
    image_ids, labels = [], []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            image_ids.append(row[0])
            labels.append(int(row[1]))

    return image_ids, labels
