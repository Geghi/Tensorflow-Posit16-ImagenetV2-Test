import os
import pandas as pd


def get_label(path):
    parts = path.split("/")
    img_label = parts[-2]
    return int(img_label)


MODELS_RESULTS_DIR = "Completed-Models/"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


for subdir, dirs, files in os.walk(MODELS_RESULTS_DIR):
    for file in files:
        if 'misclassified' in file:
            print(subdir, file)
            misclassified_images = pd.read_csv(subdir + '/' + file)
            previous_label = None
            label_counter = 0
            for index, row in misclassified_images.iterrows():
                file_path = row['Path']
                new_label = get_label(file_path)
                if previous_label == new_label:
                    label_counter = label_counter + 1
                    # Print the class label for which none of the images has been correctly classified.
                    if label_counter == 9:
                        print(new_label)
                else:
                    label_counter = 0
                    previous_label = new_label
