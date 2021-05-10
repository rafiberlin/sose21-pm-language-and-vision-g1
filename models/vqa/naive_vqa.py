import tensorflow as tf
from models.model import CNN_Encoder, BahdanauAttention
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import random as rn
from PIL import Image

#Beware : VQA uses pictures from MS COCO 2014 => some pictures disapeared in MS COCO 2017...
VQA_ANNOTATIONS_DIR = "/home/rafi/_datasets/VQA/"
MS_COCO_DIR = '/home/rafi/_datasets/MSCOCO/'

def get_pretrained_image_encoder():
    # Get the Image Encoder trained on captioning with frozen weights

    encoder = CNN_Encoder(embedding_dim=256)

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captioning/checkpoints/train/")
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        print("Not able to restore pretrained Image Encoder")

    for l in encoder.layers:
        l.trainable = False
    return encoder


# encoder = get_pretrained_image_encoder()
#
# annotation_file = os.path.join(VQA_ANNOTATIONS_DIR, "v2_mscoco_train2014_annotations.json")
# question_file = os.path.join(VQA_ANNOTATIONS_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
#
# with open(annotation_file, 'r') as f:
#     annotations = json.load(f)["annotations"]
#
# with open(question_file, 'r') as f:
#     questions = json.load(f)["questions"]
#
# #Get the first image
# image_id = annotations[0]["image_id"]
# coco_train = os.path.join(MS_COCO_DIR, "train2017")
# image_path = os.path.join( coco_train, '%012d.jpg' % (image_id))

# I have to check that but I think each line in questions matches the corresponding lines in annotations (answers)


#img = Image.open(image_path)
#img.show()


# Lets try to make this implementation work: https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180
# https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention
# Official API for retrieval: https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaDemo.py



def load_preprocessed_data ():

    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)


    label_encoder_serialized =  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/label_encoder.pickle")

    with open(label_encoder_serialized, 'rb') as handle:
        label_encoder = pickle.load(handle)


    serialized_question_vector_train = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/question_vector_train.pickle")

    with open(serialized_question_vector_train, 'rb') as handle:
        question_vector_train = pickle.load(handle)

    serialized_question_vector_val = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/question_vector_val.pickle")

    with open(serialized_question_vector_val, 'rb') as handle:
        question_vector_val  = pickle.load(handle)

    x_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/X_train.csv")
    x_val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/X_val.csv")


    X_train = pd.read_csv(x_train_path)
    X_val = pd.read_csv(x_val_path)

    return X_train, X_val, tokenizer, label_encoder, question_vector_train, question_vector_val

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

    image_conv_layer1 = tf.keras.layers.Conv2D(filters=4096, kernel_size=7, strides=1, padding="valid",
                                               activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=45))(image_input)

    image_flatten = tf.keras.layers.Flatten()(image_conv_layer1)

    image_dense_1 = tf.keras.layers.Dense(4096, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.he_uniform(seed=54))(image_flatten)

    image_dense_2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.he_uniform(seed=32))(image_dense_1)

    # Input 2 Pathway
    question_emb = tf.keras.layers.Embedding(input_dim=len(vocab_size) + 1, output_dim=300,
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

    image_encoder = get_pretrained_image_encoder()


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

# The preprocessed vqa data can be downloaded from:
#https://drive.google.com/file/d/1jF_bPICe490BMaWyTpoy9kEH9PWdX77l/view?usp=sharing
#must be unzipped at this level (a directory named checkpoinst will be at the same lvel as naive_vqa.py)
if __name__ == "__main__":

    X_train, X_val, tokenizer, label_encoder, question_vector_train, question_vector_val = load_preprocessed_data()
    coco_train = os.path.join(MS_COCO_DIR, "train2017")
    coco_train_cache = os.path.join(MS_COCO_DIR, "train2017", "vqa_cache")
    image_paths_train = X_train['image_id'].apply(lambda x:  os.path.join(coco_train, '%012d.jpg' % (x))).values
    image_paths_train_cache  =X_train['image_id'].apply(lambda x: os.path.join(coco_train_cache, '%012d.jpg' % (x))).values
    image_paths_val = X_val['image_id'].apply(lambda x:  os.path.join(coco_train, '%012d.jpg' % (x))).values
    image_paths_val_cache = X_val['image_id'].apply(lambda x: os.path.join(coco_train_cache, '%012d.jpg' % (x))).values

    question_max_length = question_vector_train.shape[1]
    vocabulary_size = tokenizer.word_index
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

    train_dataset = create_dataset(image_paths_train_cache, question_vector_train, answer_vector_train)
    val_dataset = create_dataset(image_paths_val_cache, question_vector_val, answer_vector_val)


    all_image_dict = {}
    # for i, e in enumerate(dataset):
    #     pass

    np.random.seed(42)

    ##fixing tensorflow RS
    tf.random.set_seed(32)

    ##python RS
    rn.seed(12)

    #Just apply it once and comment out
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    vqa.fit(train_dataset, epochs = 20, validation_data = val_dataset, callbacks = cb)
