import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.model import CNN_Encoder
import os
import json
import jsonlines

def get_config():
    """
    Get the general project config
    :return:
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)
    conf["glove_embeddings"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", conf["glove_embeddings"] )
    if not os.path.exists(conf["glove_embeddings"]):
        os.makedirs(conf["glove_embeddings"])
        print("Created directory:", conf["glove_embeddings"])


    conf["captioning"]["pretrained_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../",
                                            conf["captioning"]["pretrained_dir"])
    pretrained_dir = conf["captioning"]["pretrained_dir"]
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
        print("Created directory:", pretrained_dir)
    checkpoints_dir = os.path.join(pretrained_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print("Created directory:", checkpoints_dir)
    return conf

def get_ade20_vqa_data():
    """
    Get the general project configpretrained_dir = conf["captioning"]["pretrained_dir"]
    :return:
    """
    conf = get_config()
    vqa_file = conf["ade20k_dir"]
    file = os.path.join(vqa_file,"ade20k_vqa.jsonl")

    with jsonlines.open(file) as reader:
        data = [ i for i in iter(reader)]
    return data


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    #avoids distortions
    #img = tf.image.resize_with_pad(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def load_image_with_pad(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, (299, 299))
    #avoids distortions
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