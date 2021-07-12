import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from avatar_models.model import CNN_Encoder
import os
from config.util import get_config
import jsonlines
import pickle
import pandas as pd
import json
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split



def load_preprocessed_vqa_data():
    """
    :return: X_train: VQA Training Data
            X_val: VQA Validation Data
            tokenizer: the question tokenizer for our Attention VQA model
            label_encoder: encodes a question for our Attention VQA model
            question_vector_train: endcoded questions for training for our Attention VQA model
            question_vector_val: encoded questions for validation for our Attention VQA model
    """

    conf = get_config()
    vqa_conf = conf["vqa"]["attention"]
    pretrained_dir = vqa_conf["pretrained_dir"]

    serialized_tokenizer = os.path.join(pretrained_dir,
                                        "tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    label_encoder_serialized = os.path.join(pretrained_dir,
                                            "label_encoder.pickle")

    with open(label_encoder_serialized, 'rb') as handle:
        label_encoder = pickle.load(handle)

    serialized_question_vector_train = os.path.join(pretrained_dir,
                                                    "question_vector_train.pickle")

    with open(serialized_question_vector_train, 'rb') as handle:
        question_vector_train = pickle.load(handle)

    serialized_question_vector_val = os.path.join(pretrained_dir,
                                                  "question_vector_val.pickle")

    with open(serialized_question_vector_val, 'rb') as handle:
        question_vector_val = pickle.load(handle)

    x_train_path = os.path.join(pretrained_dir,
                                "X_train.csv")
    x_val_path = os.path.join(pretrained_dir,
                              "X_val.csv")

    X_train = pd.read_csv(x_train_path)
    X_val = pd.read_csv(x_val_path)

    return X_train, X_val, tokenizer, label_encoder, question_vector_train, question_vector_val


def get_ade20_vqa_data(file_name="ade20k_vqa.jsonl"):
    """
    Get the general project configpretrained_dir = conf["captioning"]["pretrained_dir"]
    :return:
    """
    conf = get_config()
    vqa_file = conf["ade20k_vqa_dir"]
    file = os.path.join(vqa_file, file_name)
    print(f"Reading {file}")
    with jsonlines.open(file) as reader:
        data = [i for i in iter(reader)]
    return data


def get_ade20_qa_cleaned(file_name="ade20k_qa_cleaned.json"):
    """
    Get the general project configpretrained_dir = conf["captioning"]["pretrained_dir"]
    :return:
    """
    conf = get_config()
    vqa_file = conf["ade20k_dir"]
    file = os.path.join(vqa_file, file_name)

    with open(file) as content:
        data = json.load(content)
    return data


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    # avoids distortions
    # img = tf.image.resize_with_pad(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def load_image_with_pad(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, (299, 299))
    # avoids distortions
    img = tf.image.resize_with_pad(img, 299, 299)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def get_image_feature_extractor():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    return tf.keras.Model(new_input, hidden_layer)


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result / 2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def get_pretrained_image_encoder():
    # Get the Image Encoder trained on captioning with frozen weights

    encoder = CNN_Encoder(embedding_dim=256)

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captioning/checkpoints/train/")
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Restoring:", ckpt_manager.latest_checkpoint)
    else:
        print("Not able to restore pretrained Image Encoder")

    for l in encoder.layers:
        l.trainable = False
    return encoder


def add_image_path_qa_data(
        path_save="/home/rafi/PycharmProjects/sose21-pm-language-and-vision-g1/data/ade20k_vqa/ade20k_qa_cleaned_with_image_path.json"):
    """
    The qa pairs for vqa does not have the proper paths to images, this function tries to fix the issue.
    :param path_save: Path to the new file with corrected image paths
    :return:
    """
    vqa_yes_no = get_ade20_vqa_data()
    qa_cleaned = get_ade20_qa_cleaned()
    # get Key values of the form : "ADE_train_00005297": "training/c/cathedral/indoor/ADE_train_00005297.jpg"
    vqa_path_dict = {re.search(r'.*/(.*?).jpg', line["image_path"]).group(1): line["image_path"] for line in vqa_yes_no}

    corrected_qa_paths = {}
    for k in qa_cleaned.keys():
        if k in vqa_path_dict.keys():
            path = vqa_path_dict[k]
            corrected_qa_paths[path] = qa_cleaned[k]

    print(f"Saving {len(corrected_qa_paths)} corrrected paths to {path_save}")
    with open(path_save, 'w') as outfile:
        json.dump(corrected_qa_paths, outfile)


def merge_synthetic_qa(save_dir=None, save_filename_root="merged_synthetic_vqa"):
    """
    Creates 4 different files: merged_synthetic_vqa.jsonl, merged_synthetic_vqa_test.jsonl, merged_synthetic_vqa_train.jsonl,
     merged_synthetic_vqa_splits.json
    The merged_synthetic_vqa_splits.json file is a dictionary containing training and test data in one file
    The *.jsonlines file has the same content in a flat structure (one line, one question answer pair)
    :param save_dir: The directory where to create the merged synthetic data. If none, it will use the default ADE20 directory
    :param save_filename_root: the root name of the different files that are created
    :return:
    """

    if save_dir is None:
        save_dir = get_config()["ade20k_dir"]

    path_save = os.path.join(save_dir, save_filename_root)

    qa_cleaned = get_ade20_qa_cleaned("ade20k_qa_cleaned_with_image_path.json")
    vqa = get_ade20_vqa_data()
    merged_vqa = {k: qa_cleaned[k] for k in qa_cleaned.keys()}
    print("Start merging")
    for row in tqdm(vqa):
        key = row["image_path"]
        if key in merged_vqa.keys():
            merged_vqa[key].append({"question": row["question"], "answer": row["answer"]})


    print(f"Save {path_save}.jsonl")
    jsonl = []
    with jsonlines.open(path_save+".jsonl", 'w') as outfile:
        for key in tqdm(merged_vqa.keys()):
            for row in merged_vqa[key]:
                jsonl.append({"image_path": key, "question": row["question"], "answer": row["answer"]})
                outfile.write({"image_path": key, "question": row["question"], "answer": row["answer"]})


    vqa_dataset = pd.DataFrame(jsonl)
    train_subset, test_subset = train_test_split(vqa_dataset, random_state=42)
    training_subset = train_subset.to_dict()
    testing_subset = test_subset.to_dict()
    final_dataset={"training" : training_subset, "testing" : testing_subset}

    with jsonlines.open(path_save+"_test.jsonl", 'w') as outfile:
        print("Saving:", path_save + "_test.jsonl")
        for key in tqdm(test_subset["image_path"].keys()):
            outfile.write({"image_path": test_subset["image_path"][key], "question": test_subset["question"][key], "answer": test_subset["answer"][key]})

    with jsonlines.open(path_save+"_train.jsonl", 'w') as outfile:
        print("Saving:", path_save + "_train.jsonl")
        for key in tqdm(test_subset["image_path"].keys()):
            outfile.write({"image_path": test_subset["image_path"][key], "question": test_subset["question"][key], "answer": test_subset["answer"][key]})

    with open(path_save+"_splits.json", "w") as f:
        print("Saving:", path_save+"_splits.json")
        json.dump(final_dataset, f)


def resize_img_to_array(img, img_shape=(244, 244)):
    """
    REshape an Image to the given Size
    :param img: Image as Numpy array
    :param img_shape: The new shape of the image, tuple of height and width
    :return:
    """
    # Code from https://gist.github.com/yinleon/8a6bf8c2f3226a521680f930f3d4339f
    img_array = np.array(
        img.resize(
            img_shape,
            Image.ANTIALIAS
        )
    )

    return img_array


def image_grid(fn_images: list,
               text: list = [],
               top: int = 12,
               per_row: int = 2,
               dim: tuple = (150, 150)):
    """
    # Code from https://gist.github.com/yinleon/8a6bf8c2f3226a521680f930f3d4339f
    fn_images is a list of image paths.
    text is a list of annotations.
    top is how many images you want to display
    per_row is the number of images to show per row.
    """
    img_width, img_height = dim
    for i in range(len(fn_images[:top])):
        if i % per_row == 0:
            _, ax = plt.subplots(1, per_row,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(15, 5))
        j = i % per_row
        image = Image.open(fn_images[i])
        image = resize_img_to_array(image,
                                    img_shape=(img_width,
                                               img_height))
        ax[j].imshow(image)
        ax[j].axis('off')
        if text:
            offset = -10
            start = -30
            for t in text[i]:
                start += offset
                ax[j].annotate(t,
                               (0, 0), (0, start),
                               xycoords='axes fraction',
                               textcoords='offset points',
                               va='top')

if __name__ == "__main__":
    merge_synthetic_qa()
    pass
