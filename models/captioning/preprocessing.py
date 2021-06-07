# From https://www.tensorflow.org/tutorials/text/image_captioning
# You need to downlod the MSCOO dataset and the annotaions, change the variable base_dir accordingly...
# If you want to debug (step in where the loss is calculated), comment out the annotation @tf.function

import tensorflow as tf
import pickle
import collections
import random
import numpy as np
import os
import json
import models.utils.util as util
import math
from models.utils.util import get_config

from tqdm import tqdm

# Caches the images, store the Glove embeddings as Numpy array, initialize the tokenizer and
# prepare the data (path to captions and vectorized captions for training)
if __name__ == "__main__":

    conf = get_config()
    captioning_conf = conf["captioning"]
    VOCAB_SIZE = captioning_conf["vocab_size"]
    EMBEDDING_DIM = captioning_conf["embedding_dim"]
    BATCH_SIZE = captioning_conf["batch_size"]
    PRETRAINED_DIR = captioning_conf["pretrained_dir"]
    USE_GLOVE = captioning_conf["use_glove"]
    PAD_IMAGE = captioning_conf["pad_image"]

    base_dir = conf["ms_coco_dir"]
    annotation_folder = 'annotations'
    annotation_file = os.path.join(base_dir, annotation_folder, 'captions_train2017.json')
    if not os.path.exists(os.path.join(base_dir, annotation_folder)):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(base_dir),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                                 extract=True)
    # Download image files
    # CHANGE THIS!
    image_folder = 'train2017'
    PATH = os.path.join(base_dir, image_folder)
    if not os.path.exists(os.path.join(base_dir, image_folder)):
        image_zip = tf.keras.utils.get_file('train2017.zip',
                                            cache_subdir=os.path.abspath(base_dir),
                                            origin='http://images.cocodataset.org/zips/train2017.zip',
                                            extract=True)
        os.remove(image_zip)


    with open(annotation_file, 'r') as f:
        annotations = json.load(f)



    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = os.path.join(PATH, '%012d.jpg' % (val['image_id']))
        image_path_to_caption[image_path].append(caption)


    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    # lead to 30,000 examples.
    train_image_paths = image_paths[:]
    #train_image_paths = image_paths[:6000]
    print(len(train_image_paths))


    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))


    # print(train_captions[0])
    # Image.open(img_name_vector[0])

    image_features_extract_model = util.get_image_feature_extractor()

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # if padding, the image does not get distorted when resizing it
    if PAD_IMAGE:
        image_dataset = image_dataset.map(
            util.load_image_with_pad, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    else:
        image_dataset = image_dataset.map(
            util.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

    index_feature = True

    for f in os.listdir(PATH):
        if f.endswith(".npy"):
            index_feature = False
            break

    if index_feature:
        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())


    # Find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)


    # Choose the top 5000 words from the vocabulary
    serialized_tokenizer = os.path.join(PRETRAINED_DIR, "tokenizer.pickle")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    tokenizer.max_length = max_length
    with open(serialized_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if USE_GLOVE:
        GLOVE_FILE = conf["glove_embeddings"]
        embeddings_index = {}
        if os.path.isfile(GLOVE_FILE):

            with open(GLOVE_FILE) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs

        embedding_matrix = np.zeros((VOCAB_SIZE+1, EMBEDDING_DIM))
        missed = 0
        hit = 0
        # idx 0 used for padding is skipped
        for word_idx in range(1, VOCAB_SIZE+1):
            word = tokenizer.index_word[word_idx]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[word_idx] = embedding_vector[:EMBEDDING_DIM]
                hit = hit + 1
            else:
                # Xavier / Glodot Init for Numpy
                # See https://stackoverflow.com/questions/62249084/what-is-the-numpy-equivalent-of-tensorflow-xavier-initializer-for-cnn
                np.random.seed(0)
                scale = 1 / max(1., (2 + 2) / 2.)
                limit = math.sqrt(3.0 * scale)
                embedding_matrix[word_idx] = np.random.uniform(-limit, limit, size=(EMBEDDING_DIM))
                missed = missed + 1
        print("hit embeds: ", hit)
        print("missed embeds: ", missed)

        np.save(os.path.join(PRETRAINED_DIR, f"glove{EMBEDDING_DIM}_V{VOCAB_SIZE + 1}"), embedding_matrix)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    # slice_index = int(len(img_keys) * 0.8)
    # img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]
    img_name_train_keys = img_keys

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    file_name = os.path.join(PRETRAINED_DIR, "img_name_train.pickle")
    with open(file_name, "wb") as open_file:
        pickle.dump(img_name_train, open_file)

    file_name = os.path.join(PRETRAINED_DIR, "cap_train.pickle")
    with open(file_name, "wb") as open_file:
        pickle.dump(cap_train, open_file)
