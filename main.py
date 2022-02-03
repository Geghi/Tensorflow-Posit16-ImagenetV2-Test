import os
import pandas as pd
import json

import tensorflow as tf
from tensorflow.keras.preprocessing import image

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


def build_dir_tree(models):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    for pretrained_model in models:
        if not os.path.exists(MODELS_DIR + pretrained_model):
            os.makedirs(MODELS_DIR + pretrained_model)


def convert_image(img, format_name):
    if format_name == 'Posit160':
        img = tf.convert_to_tensor(img, dtype=tf.posit160)
    elif format_name == 'FP32':
        img = tf.convert_to_tensor(img, dtype=tf.float32)
    elif format_name == 'FP16':
        img = tf.convert_to_tensor(img, dtype=tf.float16)
    elif format_name == 'BFLOAT16':
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


def predict_data(images, labels, class2label):
    hit = 0
    for i, test_image in enumerate(images):
        class_number = labels[i]
        class_id, class_name = class2label[str(class_number)]
        # print("\nREAL__IMG Class Number: ", class_number, "\tClass Index: ", class_id, "\tClass Name: ", class_name)

        test_image = models_dictionary.get(model_name)[1](test_image)
        probs = model.predict(test_image)
        label = models_dictionary.get(model_name)[2](probs)[0][0]
        predicted_class_number = probs.argmax(axis=-1)[0]

        # print('PREDICTED Class Number: ', predicted_class_number, '\tClass Index: ', label[0], '\tClass Name: ',
        # label[1], '\tProbability: ', label[2] * 100, '%')
        if class_number == predicted_class_number:
            hit = hit + 1
        else:
            misclassified_paths.append([paths[i], class_name, label[1], predicted_class_number])
    return hit


# DATASET_DIR = 'imagenetv2-top-images/imagenetv2-top-images-format-val/'
DATASET_DIR = 'imagenetv2-matched-frequency-format-val/'
MODELS_DIR = 'models/'
batch_size = 50


# Create Models Array
VGG16_pre_trained = VGG16(weights='imagenet')
VGG19_pre_trained = VGG19(weights='imagenet')
ResNet50_pre_trained = ResNet50(weights='imagenet')
InceptionV3_pre_trained = InceptionV3(weights='imagenet')
InceptionResNetV2_pre_trained = InceptionResNetV2(weights='imagenet')

models_dictionary = {
    "VGG16": [VGG16_pre_trained, pp_vgg16, dd_vgg16, (224, 224)],
    "VGG19": [VGG19_pre_trained, pp_vgg19, dd_vgg19, (224, 224)],
    "ResNet50": [ResNet50_pre_trained, pp_resnet, dd_resnet, (224, 224)],
    "InceptionResNetV2": [InceptionResNetV2_pre_trained, pp_inceptionresnetv2, dd_inceptionresnetv2, (299, 299)],
    "InceptionV3": [InceptionV3_pre_trained, pp_inceptionv3, dd_inceptionv3, (299, 299)]
}

formats = ["Posit160", "FP32", "BFLOAT16", "FP16"]
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

        for model_name in model_names:
            print("Model Name: ", model_name)

            test_images = []
            test_labels = []
            count = 0
            hits = 0
            total_images_completed = 0
            paths = []
            misclassified_paths = []
            model = models_dictionary.get(model_name)[0]
            dim = models_dictionary.get(model_name)[3]

            for subdir, dirs, files in os.walk(DATASET_DIR):
                for file in files:
                    count = count + 1
                    path = os.path.join(subdir, file)
                    paths.append(path)
                    test_labels.append(get_label(path))
                    test_images.append(convert_image(load_image_from_dataset(path, dim), format_type))

                if count >= batch_size:
                    # Intermediate result
                    hits = hits + predict_data(test_images, test_labels, class_idx)
                    total_images_completed = total_images_completed + batch_size
                    test_images = []
                    test_labels = []
                    paths = []
                    count = 0
                    print("\nCorrectly Classified: ", hits, " over ", total_images_completed, " images")
                    # break  # Remove comment to stop after one batch_size Cycle and debug results.

            miss_df = pd.DataFrame(misclassified_paths)
            miss_df.to_csv(MODELS_DIR + model_name + '/' + format_type + '_misclassified_images.csv',
                           header=["Path", "Real Class Name", "Predicted Class Name", "Predicted Class Number"])

            print("\nTotal Correctly Classified Images: ", hits, " over: 10000 images")
            accuracy = hits * 100 / 10000
            print("Accuracy :", accuracy, "%")

            results.append([model_name, accuracy])

        print(results)

        # Save Final Results
        results_df = pd.DataFrame(results)
        filename = format_type + "_"
        for name in model_names:
            filename = filename + name[0:3] + "_"

        results_df.to_csv(MODELS_DIR + filename + 'res.csv', header=["Model Name", "Accuracy"])

    print("Completed!")
