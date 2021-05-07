# From https://www.tensorflow.org/tutorials/text/image_captioning
# You need to downlod the MSCOO dataset and the annotaions, change the variable base_dir accordingly...
# If you want to debug (step in where the loss is calculated), comment out the annotation @tf.function

import tensorflow as tf
import pickle
import numpy as np
import os
import models.utils.util as util
from models.model import RNN_Decoder, CNN_Encoder, BahdanauAttention




def get_eval_model():
    # Choose the top 5000 words from the vocabulary
    VOCAB_SIZE = 5000
    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/train/tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
        max_length = tokenizer.max_length

    embedding_dim = 256
    units = 512
    vocab_size = VOCAB_SIZE + 1
    attention_features_shape = 64

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    # checkpoint_path = "./checkpoints/train"

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/train/")
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    image_features_extract_model = util.get_image_feature_extractor()

    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(util.load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                     -1,
                                                     img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input,
                                                             features,
                                                             hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            predicted_id = tf.math.argmax(predictions[0]).numpy()
            # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    return evaluate


# model to unpack at this level:
# https://drive.google.com/file/d/1d2ZH7699DDStrJt5EOsFneT1XWGgcNRj/view?usp=sharing
# at the same level as this scrip, you should see directory called checkpoints/
if __name__ == "__main__":
    model = get_eval_model()

    image_url = 'https://tensorflow.org/images/surf.jpg'
    # image_url = 'http://localhost:8000/a/atrium/home/ADE_train_00001860.jpg'
    last_char_index = image_url.rfind("/")
    image_name = image_url[last_char_index + 1:]
    image_path = tf.keras.utils.get_file(image_name, origin=image_url)

    result, _ = model(image_path)

    print('Prediction Caption:', ' '.join(result))
