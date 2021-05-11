from tensorflow import keras
import tensorflow as tf
import os
import models.utils.util as util
import pickle
from models.vqa.create_preprocessed_questions import preprocess_english, preprocess_english_add_tokens


def get_eval_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "checkpoints/model.06-1.80.h5")

    model = keras.models.load_model(path)

    return model

# The preprocessed vqa data and model can be downloaded from:
#https://drive.google.com/file/d/1jF_bPICe490BMaWyTpoy9kEH9PWdX77l/view?usp=sharing
#must be unzipped at this level (a directory named checkpoinst will be at the same lvel as naive_vqa.py

if __name__ == "__main__":
    model = get_eval_model()
    image_extractor = util.get_pretrained_image_encoder()

    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    image_url = 'https://tensorflow.org/images/surf.jpg'
    question = "Is there a man?"
    preprocessed_question = preprocess_english_add_tokens(preprocess_english(question))
    # image_url = 'http://localhost:8000/a/atrium/home/ADE_train_00001860.jpg'
    last_char_index = image_url.rfind("/")
    image_name = image_url[last_char_index + 1:]
    image_path = tf.keras.utils.get_file(image_name, origin=image_url)
    inception_v3 = util.get_image_feature_extractor()
    images = tf.expand_dims(util.load_image(image_path)[0], 0)
    images = inception_v3(images)
    images = image_extractor(images)
    question_seq = tokenizer.texts_to_sequences([preprocessed_question])
    vectorized_question_seq = tf.keras.preprocessing.sequence.pad_sequences(question_seq, maxlen=model.input[1].shape[1], padding='post')
    input = (images, vectorized_question_seq)
    inference = model.predict(input)
    print(inference)


    label_encoder_serialized =  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/label_encoder.pickle")

    with open(label_encoder_serialized, 'rb') as handle:
        label_encoder = pickle.load(handle)
    #TODO put the inference into a softmax to get the most probably answer and use the label encoder to get the answer as a string...
