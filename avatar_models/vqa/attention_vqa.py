import tensorflow as tf
from avatar_models.model import CoattentionModel
import os
from config.util import get_config,create_directory_structure
from avatar_models.utils.util import load_preprocessed_vqa_data
import pickle
import numpy as np
import pandas as pd
import random as rn

#Beware : VQA uses pictures from MS COCO 2014 => some pictures disapeared in MS COCO 2017...




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





def get_image_tensor(img, ques):
    #path = img.decode('utf-8').replace(imageDirectory,imageNumpyDirectory).replace('.jpg',"") +'.npy'
    #img_tensor = np.load(path)
    img_tensor = np.load(img.decode('utf-8')+'.npy')
    # quick fix: the feature maps were saved as 8*8*256 => the model expects only to dimensions as input for the images...
    return img_tensor.reshape((FEAT_MAP_WIDTH*FEAT_MAP_WIDTH, -1)), ques

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

def build_co_attention_model(image_embedding, feature_maps, number_of_answers, question_max_length, vocabulary_size):
    image_input = tf.keras.layers.Input(shape = (feature_maps*feature_maps, image_embedding))

    question_input = tf.keras.layers.Input(shape=(question_max_length,))

    output = CoattentionModel(number_of_answers, vocabulary_size, image_embedding)(image_input,question_input)#num_embeddings = len(ques_vocab), num_classes = len(ans_vocab), embed_dim = 512

    model = tf.keras.models.Model(inputs = [image_input, question_input], outputs = output)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model





# The preprocessed vqa data and model can be downloaded from:
#https://drive.google.com/file/d/1jF_bPICe490BMaWyTpoy9kEH9PWdX77l/view?usp=sharing
#must be unzipped at this level (a directory named checkpoinst will be at the same lvel as naive_vqa.py)
if __name__ == "__main__":
    BATCH_SIZE = 128
    IMG_EMBED_SIZE = 256
    FEAT_MAP_WIDTH = 8

    conf = get_config()
    vqa_conf = conf["vqa"]["attention"]
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




    #TODO Finish the implementation of the naive model based on
    # 3_Modeling.ipynb from https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention


    vqa = build_co_attention_model(IMG_EMBED_SIZE, FEAT_MAP_WIDTH, number_of_answers, question_max_length, vocabulary_size)



    train_dataset = create_dataset(image_paths_train_cache, question_vector_train, answer_vector_train, BATCH_SIZE)
    val_dataset = create_dataset(image_paths_val_cache, question_vector_val, answer_vector_val, BATCH_SIZE)


    all_image_dict = {}

    np.random.seed(42)

    ##fixing tensorflow RS
    tf.random.set_seed(32)

    ##python RS
    rn.seed(12)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy'),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(pretrained_dir, 'checkpoints/attention_model.{epoch:02d}-{val_loss:.2f}.h5')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(pretrained_dir,'logs')),
    ]

    #vqa.load_weights("./checkpoints/attention_model.08-0.39.h5")
    #vqa.save(".test")
    # tf.keras.avatar_models.load_model(".test/")
    #tf.config.experimental_run_functions_eagerly(True)
    vqa.fit(train_dataset, epochs = 20, validation_data = val_dataset, callbacks = cb)

