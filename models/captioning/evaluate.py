# From https://www.tensorflow.org/tutorials/text/image_captioning
# You need to downlod the MSCOO dataset and the annotaions, change the variable base_dir accordingly...
# If you want to debug (step in where the loss is calculated), comment out the annotation @tf.function

import tensorflow as tf
import pickle

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image



def get_eval_model():




    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path



    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    BATCH_SIZE = 8


    # Choose the top 5000 words from the vocabulary
    VOCAB_SIZE = 5000
    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/train/tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
        max_length = tokenizer.max_length

    # Create the tokenized vectors





    embedding_dim = 256
    units = 512
    vocab_size = VOCAB_SIZE + 1
    features_shape = 2048
    attention_features_shape = 64



    class BahdanauAttention(tf.keras.Model):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, features, hidden):
            # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

            # hidden shape == (batch_size, hidden_size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
            hidden_with_time_axis = tf.expand_dims(hidden, 1)

            # attention_hidden_layer shape == (batch_size, 64, units)
            attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                                 self.W2(hidden_with_time_axis)))

            # score shape == (batch_size, 64, 1)
            # This gives you an unnormalized score for each image feature.
            score = self.V(attention_hidden_layer)

            # attention_weights shape == (batch_size, 64, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * features
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights


    # In[24]:


    class CNN_Encoder(tf.keras.Model):
        # Since you have already extracted the features and dumped it
        # This encoder passes those features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(CNN_Encoder, self).__init__()
            # shape after fc == (batch_size, 64, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)

        def call(self, x):
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x


    # In[25]:


    class RNN_Decoder(tf.keras.Model):
        def __init__(self, embedding_dim, units, vocab_size):
            super(RNN_Decoder, self).__init__()
            self.units = units

            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.fc1 = tf.keras.layers.Dense(self.units)
            self.fc2 = tf.keras.layers.Dense(vocab_size)

            self.attention = BahdanauAttention(self.units)

        def call(self, x, features, hidden):
            # defining attention as a separate model
            context_vector, attention_weights = self.attention(features, hidden)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # shape == (batch_size, max_length, hidden_size)
            x = self.fc1(output)

            # x shape == (batch_size * max_length, hidden_size)
            x = tf.reshape(x, (-1, x.shape[2]))

            # output shape == (batch_size * max_length, vocab)
            x = self.fc2(x)

            return x, state, attention_weights

        def reset_state(self, batch_size):
            return tf.zeros((batch_size, self.units))



    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()


    #checkpoint_path = "./checkpoints/train"

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/train/")
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()



    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
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
            #predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot




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

    return evaluate

# model to unpack at this level:
#https://drive.google.com/file/d/1d2ZH7699DDStrJt5EOsFneT1XWGgcNRj/view?usp=sharing
# at the same level as this scrip, you should see directory called checkpoints/
if __name__ == "__main__":
    model = get_eval_model()

    image_url = 'https://tensorflow.org/images/surf.jpg'
    #image_url = 'http://localhost:8000/a/atrium/home/ADE_train_00001860.jpg'
    last_char_index = image_url.rfind("/")
    image_name = image_url[last_char_index+1:]
    image_path = tf.keras.utils.get_file(image_name, origin=image_url)

    result, _ = model(image_path)


    print('Prediction Caption:', ' '.join(result))