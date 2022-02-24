import os
import pandas as pd
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_vgg16
from tensorflow.keras.applications.vgg16 import decode_predictions as dd_vgg16

# VGG19
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
from tensorflow.keras.applications.vgg16 import decode_predictions as dd_vgg19

# InceptionV3
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inceptionv3
from tensorflow.keras.applications.inception_v3 import decode_predictions as dd_inceptionv3

# InceptionResnetV2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as pp_inceptionresnetv2
from tensorflow.keras.applications.inception_resnet_v2 import decode_predictions as dd_inceptionresnetv2

# ResNet50
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet
from tensorflow.keras.applications.resnet50 import decode_predictions as dd_resnet


# Initialize Models
# Pretrained models
def get_model(pretrained_model_name):
    if pretrained_model_name == 'VGG16':
        pretrained_model = VGG16(weights='imagenet')
    elif pretrained_model_name == 'VGG19':
        pretrained_model = VGG19(weights='imagenet')
    elif pretrained_model_name == 'ResNet50':
        pretrained_model = ResNet50(weights='imagenet')
    elif pretrained_model_name == 'InceptionV3':
        pretrained_model = InceptionV3(weights='imagenet')
    elif pretrained_model_name == 'InceptionResNetV2':
        pretrained_model = InceptionResNetV2(weights='imagenet')
    else:
        print("Invalid Model Name!")
        exit(1)
    return pretrained_model


def build_dir_tree(models):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    for pretrained_model in models:
        if not os.path.exists(MODELS_DIR + pretrained_model):
            os.makedirs(MODELS_DIR + pretrained_model)


def set_format_weights(format_name):
    if format_name == 'float16':
        K.set_floatx('float16')
    else:
        K.set_floatx('float32')


def convert_image(img, format_name):
    if format_name == 'posit160':
        img = tf.convert_to_tensor(img, dtype=tf.posit160)
    elif format_name == 'float32':
        img = tf.convert_to_tensor(img, dtype=tf.float32)
    elif format_name == 'float16':
        img = tf.convert_to_tensor(img, dtype=tf.float16)
    elif format_name == 'bfloat16':
        img = tf.convert_to_tensor(img, dtype=tf.bfloat16)
    else:
        print("Invalid Format")
        exit(1)
    img = tf.expand_dims(img, axis=0)
    return img


def load_image_from_dataset(file_path, size):
    img = image.load_img(file_path, target_size=size)
    img = image.img_to_array(img)
    return img


def get_label(file_path):
    parts = file_path.split("/")
    img_label = parts[-2]
    return int(img_label)


def predict_data(images, labels):
    hit = 0
    top_n_hit = 0
    for i, test_image in enumerate(images):
        class_number = labels[i]

        test_image = models_dictionary.get(model_name)[0](test_image)
        probs = model.predict(test_image)
        predicted_class_number = probs.argmax(axis=-1)[0]
        top_n_labels = np.argsort(probs, axis=1)[:, -top_n:]
        # label = models_dictionary.get(model_name)[1](probs)[0][0]

        if class_number == predicted_class_number:
            hit = hit + 1
        if class_number in top_n_labels:
            top_n_hit = top_n_hit + 1
    return hit, top_n_hit


# DATASET_DIR = 'imagenetv2-top-images/imagenetv2-top-images-format-val/'
DATASET_DIR = 'imagenetv2-matched-frequency-format-val/'
MODELS_DIR = 'FinalResults/'
batch_size = 50
top_n = 5

models_dictionary = {
    "VGG16": [pp_vgg16, dd_vgg16, (224, 224)],
    "VGG19": [pp_vgg19, dd_vgg19, (224, 224)],
    "ResNet50": [pp_resnet, dd_resnet, (224, 224)],
    "InceptionResNetV2": [pp_inceptionresnetv2, dd_inceptionresnetv2, (299, 299)],
    "InceptionV3": [pp_inceptionv3, dd_inceptionv3, (299, 299)]
}

formats = ["float32", "bfloat16", "float16", "posit160"]
model_names = ["VGG16", "VGG19", "ResNet50", "InceptionResNetV2", "InceptionV3"]


if __name__ == '__main__':

    build_dir_tree(model_names)

    # save mappings between class number and class name to check if the prediction is correct.
    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    for format_type in formats:
        print("Data Type Name: ", format_type)

        results = []

        set_format_weights(format_type)

        for model_name in model_names:
            print("Model Name: ", model_name)
            model = None
            model = get_model(model_name)

            if format_type is not 'float32':
                ws = model.get_weights()
                wsp = [w.astype(format_type) for w in ws]
                model.set_weights(wsp)

            print(np.unique([w.dtype for w in model.get_weights()]))

            test_images = []
            test_labels = []
            count = 0
            hits = 0
            top_n_hits = 0
            total_images_completed = 0
            paths = []
            misclassified_paths = []

            dim = models_dictionary.get(model_name)[2]

            for subdir, dirs, files in os.walk(DATASET_DIR):
                for file in files:
                    count = count + 1
                    path = os.path.join(subdir, file)
                    paths.append(path)
                    test_labels.append(get_label(path))
                    test_images.append(convert_image(load_image_from_dataset(path, dim), format_type))

                    if count >= batch_size:
                        # Intermediate result
                        batch_hits, batch_top_n_hits = predict_data(test_images, test_labels)
                        hits = hits + batch_hits
                        top_n_hits = top_n_hits + batch_top_n_hits
                        total_images_completed = total_images_completed + batch_size
                        test_images = []
                        test_labels = []
                        paths = []
                        print("\nCorrectly Classified: ", hits, " over ", total_images_completed, " images")
                        print("\nTOP_N_ACCURACY:\nCorrectly Classified: ", top_n_hits, " over ", total_images_completed, " images")
                        # break  # Remove comment to stop after one batch_size Cycle and debug results.
                        count = 0

                '''if count >= batch_size:
                    count = 0
                    break'''

            if len(misclassified_paths) > 0:
                miss_df = pd.DataFrame(misclassified_paths)
                miss_df.to_csv(MODELS_DIR + model_name + '/' + format_type + '_misclassified_images.csv',
                               header=["Path", "Real Class Name", "Predicted Class Name", "Predicted Class Number"])

            print("\nTotal Correctly Classified Images: ", hits, " over: 10000 images")
            print("\nTOP_N_ACCURACY:\nTotal Correctly Classified Images: ", top_n_hits, " over: 10000 images")
            accuracy = hits * 100 / 10000
            top_n_accuracy = top_n_hits * 100 / 10000
            print("Accuracy :", accuracy, "%")
            print("TOP_N_Accuracy :", top_n_accuracy, "%")

            results.append([model_name, accuracy, top_n_accuracy])

        print(results)

        # Save Final Results
        results_df = pd.DataFrame(results)
        filename = format_type + "_"
        for name in model_names:
            filename = filename + name[0:3] + "_"

        results_df.to_csv(MODELS_DIR + filename + 'top_' + str(top_n) + '.csv', header=["Model Name", "Accuracy", "Top_N_Accuracy"])

    print("Completed!")

