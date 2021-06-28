from tensorflow import keras
import tensorflow as tf
import os
import models.utils.util as util
from models.vqa.attention_vqa import build_co_attention_model
import pickle
import numpy as np
from models.vqa.create_preprocessed_questions import preprocess_english, preprocess_english_add_tokens
from config.util import get_config

class TrainedVQA:

    def __init__(self, model_path, tokenizer_path, label_encoder_path):
        #self.model = keras.models.load_model(model_path)
        #self.model = build_co_attention_model(256, 8, 1000, 24, 11953)
        #https://drive.google.com/file/d/1EWMHAafdAba2wUv56bvdg8UrV9h9rw6V/view?usp=sharing
        self.model = build_co_attention_model(256, 8, 2000, 24, 12147)
        self.model.load_weights(model_path)
        #self.model.save_weights("attention_model_best.tf")
        self.inception_v3 = util.get_image_feature_extractor()
        self.image_caption_processing = util.get_pretrained_image_encoder()
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_len_question = self.model.input[1].shape[1]
        self.answer_number = self.model.output.shape[1]
        with open(label_encoder_path, 'rb') as handle:
            self.label_encoder = pickle.load(handle)


    def __process_image(self, image_url):
        last_char_index = image_url.rfind("/")
        # Can now handle file on the computer without downloading them from a url...
        url_shards = image_url.split("://")
        image_path = image_url
        if len(url_shards) == 2:
            image_path = url_shards[1]
        if not os.path.isfile(image_path) and not os.path.isfile(image_url):
            image_name = image_url[last_char_index + 1:]
            image_path = tf.keras.utils.get_file(image_name, origin=image_url)

        image = tf.expand_dims(util.load_image(image_path)[0], 0)
        image = self.inception_v3(image)
        image = self.image_caption_processing(image)
        # reshape to 1 image * feature maps (squared) * image dimensions
        return tf.reshape(image, (1 , 64 ,-1))
    def __process_question(self, question):
        preprocessed_question = preprocess_english_add_tokens(preprocess_english(question))
        question_seq = self.tokenizer.texts_to_sequences([preprocessed_question])
        if len(question_seq) > self.max_len_question:
            question_seq = question_seq[:self.max_len_question]
            print(f"Warning your question is too long!Max length is {str(self.max_len_question)}")
        vectorized_question_seq = tf.keras.preprocessing.sequence.pad_sequences(question_seq, maxlen=self.max_len_question,
                                                                                padding='post')
        return vectorized_question_seq

    def infer(self, tuple_input):
        image_url, question_string = tuple_input
        vector_question = self.__process_question(question_string)
        processed_image = self.__process_image(image_url)
        input = (processed_image, vector_question)

        inference = self.model.predict(input)
        decision = tf.nn.softmax(inference)
        # get the column id whithe the highest probability mass
        id = tf.math.argmax(decision, 1).numpy()
        one_hot_decision = np.zeros(self.answer_number)
        one_hot_decision[id] = 1
        label = self.label_encoder.inverse_transform(np.expand_dims(one_hot_decision, 0))

        return label

def get_eval_vqa_model():

    conf_vqa  = get_config()["vqa"]["attention"]
    pretrained_dir = conf_vqa["pretrained_dir"]
    serialized_tokenizer = os.path.join(pretrained_dir,
                                        "tokenizer.pickle")

    model_path = os.path.join(pretrained_dir,
                 "checkpoints/attention_model_best.h5")

    label_encoder_serialized =  os.path.join(pretrained_dir,
                                        "label_encoder.pickle")

    vqa = TrainedVQA(model_path, serialized_tokenizer, label_encoder_serialized)

    return vqa

# The preprocessed vqa data and model can be downloaded from:
# https://drive.google.com/file/d/1qNIb24KBYJZs6AJOEkuq4TNJXSRVV-fa/view?usp=sharing
# must be unzipped at this level (a directory named checkpoinst will be at the same lvel as naive_vqa.py

# If the h5 file can t be loaded, here is  a file to the weights as tensorflow format (tf)
# https://drive.google.com/file/d/1Sr7dTPV54LQW0UZGJZTqLUtLzBIrRfuy/view?usp=sharing
if __name__ == "__main__":

    image_url = 'https://tensorflow.org/images/surf.jpg'
    #question = "What do you see?" # answer: surfboard
    #question = "What is it?"  # answer: surfboard
    #question = "Who is it?"  # answer: surfer
    #question = "Is there a man?" # answer: yes
    question = "Is this a man or a dog?"  # answer: man
    vqa = get_eval_vqa_model()
    label = vqa.infer((image_url, question))
    print("question: ", question)
    print("answer: ", label[0])



