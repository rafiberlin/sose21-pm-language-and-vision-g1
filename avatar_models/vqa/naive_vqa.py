import tensorflow as tf
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import random as rn
import avatar_models.utils.util as util
from config.util import get_config, create_directory_structure
from avatar_models.utils.util import load_preprocessed_vqa_data
#Beware : VQA uses pictures from MS COCO 2014 => some pictures disapeared in MS COCO 2017...

# Lets try to make this implementation work: https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180
# https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention
# Official API for retrieval: https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaDemo.py




def get_image_tensor(img, ques):
    #path = img.decode('utf-8').replace(imageDirectory,imageNumpyDirectory).replace('.jpg',"") +'.npy'
    #img_tensor = np.load(path)
    img_tensor = np.load(img.decode('utf-8')+'.npy')
    return img_tensor, ques

def create_dataset(image_paths, question_vector, answer_vector, batch_size=8):
    # WHen enumerate is called, it return a tuple. in the element, there is another tuple of img, question
    # in the second element you get the actual answer...


    dataset_input = tf.data.Dataset.from_tensor_slices((image_paths, question_vector.astype(np.float32)))
    dataset_output = tf.data.Dataset.from_tensor_slices((answer_vector.astype(np.float32)))
    # using map to load the numpy files in parallel
    dataset_input = dataset_input.map(
        lambda img, ques: tf.numpy_function(get_image_tensor, [img, ques], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffling and batching
    # dataset_input = dataset_input.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_input = dataset_input.batch(batch_size)
    dataset_output = dataset_output.batch(batch_size)  # .repeat()

    dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def build_naive_vqa_model(image_embed_size, feature_map_number, answer_number, question_max_length, vocab_size):
    image_input = tf.keras.layers.Input(shape=(feature_map_number, feature_map_number, image_embed_size))
    question_input = tf.keras.layers.Input(shape=(question_max_length,))

    image_conv_layer1 = tf.keras.layers.Conv2D(filters=4096, kernel_size=8, strides=1, padding="valid",
                                               activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=45))(image_input)

    image_flatten = tf.keras.layers.Flatten()(image_conv_layer1)

    image_dense_1 = tf.keras.layers.Dense(4096, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.he_uniform(seed=54))(image_flatten)

    image_dense_2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.he_uniform(seed=32))(image_dense_1)

    # Input 2 Pathway
    question_emb = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=300,
                                             name="Embedding_Layer",
                                             embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1,
                                                                                                       seed=23))(
        question_input)

    question_lstm = tf.keras.layers.LSTM(1024,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                                         recurrent_initializer=tf.keras.initializers.orthogonal(seed=54),
                                         bias_initializer=tf.keras.initializers.zeros())(question_emb)

    question_flatten = tf.keras.layers.Flatten(name="Flatten_lstm")(question_lstm)

    image_question = tf.keras.layers.Multiply()([image_dense_2, question_flatten])

    image_question_dense_1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu,
                                                   kernel_initializer=tf.keras.initializers.he_uniform(seed=19))(
        image_question)

    image_question_dense_2 = tf.keras.layers.Dense(1000, activation=tf.nn.relu,
                                                   kernel_initializer=tf.keras.initializers.he_uniform(seed=28))(
        image_question_dense_1)

    output = tf.keras.layers.Dense(answer_number, activation=tf.nn.softmax,
                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=15))(
        image_question_dense_2)

    # Create Model
    model = tf.keras.models.Model(inputs=[image_input, question_input], outputs=output)
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model





def cache_vqa_images(image_paths_train):

    image_encoder = util.get_pretrained_image_encoder()


    def cache_vqa_image_feature(img_name, vqa_dir="vqa_cache"):
        img_tensor =  np.load(img_name + '.npy')
        img_tensor = image_encoder(img_tensor)
        img_tensor = tf.reshape(img_tensor, (8, 8 , -1))
        last_char_index = img_name.rfind(os.path.sep)
        coco_train = img_name[:last_char_index]
        name = img_name[last_char_index + 1:]
        save_path = os.path.join(coco_train, vqa_dir, name)
        np.save(save_path, img_tensor.numpy())

    for i , path in tqdm(enumerate(image_paths_train)):
        cache_vqa_image_feature(path)

# The preprocessed vqa data and model can be downloaded from:
#https://drive.google.com/file/d/1jF_bPICe490BMaWyTpoy9kEH9PWdX77l/view?usp=sharing
#must be unzipped at this level (a directory named checkpoinst will be at the same lvel as naive_vqa.py)
if __name__ == "__main__":

    conf = get_config()
    vqa_conf = conf["vqa"]["naive"]
    pretrained_dir = vqa_conf["pretrained_dir"]
    VQA_ANNOTATIONS_DIR = conf["vqa_dir"]
    MS_COCO_DIR = conf["ms_coco_dir"]

    X_train, X_val, tokenizer, label_encoder, question_vector_train, question_vector_val = load_preprocessed_vqa_data()
    coco_train = os.path.join(MS_COCO_DIR, "train2017")
    coco_train_cache = os.path.join(MS_COCO_DIR, "train2017", "vqa_cache")
    create_directory_structure(coco_train_cache)

    image_paths_train = X_train['image_id'].apply(lambda x:  os.path.join(coco_train, '%012d.jpg' % (x))).values
    image_paths_train_cache  =X_train['image_id'].apply(lambda x: os.path.join(coco_train_cache, '%012d.jpg' % (x))).values
    image_paths_val = X_val['image_id'].apply(lambda x:  os.path.join(coco_train, '%012d.jpg' % (x))).values
    image_paths_val_cache = X_val['image_id'].apply(lambda x: os.path.join(coco_train_cache, '%012d.jpg' % (x))).values

    question_max_length = question_vector_train.shape[1]
    vocabulary_size = len(tokenizer.word_index)
    ans_vocab = {l: i for i, l in enumerate(label_encoder.classes_)}
    number_of_answers = len(ans_vocab)

    answer_vector_train = label_encoder.fit_transform(X_train['multiple_choice_answer'].apply(lambda x: x).values)
    answer_vector_val = label_encoder.transform(X_val['multiple_choice_answer'].apply(lambda x: x).values)

    #Just apply it once and comment out
    if not os.path.exists(coco_train_cache):
        os.makedirs(coco_train_cache)
    #cache_vqa_images(image_paths_train)
    #cache_vqa_images(image_paths_val)


    #TODO Finish the implementation of the naive model based on
    # 3_Modeling.ipynb from https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention


    vqa = build_naive_vqa_model(256, 8, number_of_answers, question_max_length, vocabulary_size)

    BATCH_SIZE = 64

    train_dataset = create_dataset(image_paths_train_cache, question_vector_train, answer_vector_train, BATCH_SIZE)
    val_dataset = create_dataset(image_paths_val_cache, question_vector_val, answer_vector_val, BATCH_SIZE)


    all_image_dict = {}
    # for i, e in enumerate(dataset):
    #     pass

    np.random.seed(42)

    ##fixing tensorflow RS
    tf.random.set_seed(32)

    ##python RS
    rn.seed(12)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=4),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(pretrained_dir, 'checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(pretrained_dir,'logs')),
    ]

    vqa.fit(train_dataset, epochs = 20, validation_data = val_dataset, callbacks = cb)
