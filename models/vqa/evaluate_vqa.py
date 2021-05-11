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
    images = tf.expand_dims(util.load_image(image_path)[0], 0)
    question_seq = tokenizer.texts_to_sequences(preprocessed_question)
    vectorized_question_seq = tf.keras.preprocessing.sequence.pad_sequences(question_seq, padding='post')
    input = (images, vectorized_question_seq)
    inference = model.predict(input)
    print(inference)


