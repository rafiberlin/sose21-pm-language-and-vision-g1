# From https://www.tensorflow.org/tutorials/text/image_captioning
# You need to downlod the MSCOO dataset and the annotaions, change the variable base_dir accordingly...
# If you want to debug (step in where the loss is calculated), comment out the annotation @tf.function

import tensorflow as tf
import pickle
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
import random
import numpy as np
import os
import time
from models.model import RNN_Decoder, CNN_Encoder
from models.utils.util import get_config
from tqdm import tqdm



if __name__ == "__main__":


    conf = get_config()
    captioning_conf = conf["captioning"]
    VOCAB_SIZE = captioning_conf["vocab_size"]
    EMBEDDING_DIM = captioning_conf["embedding_dim"]
    BATCH_SIZE = captioning_conf["batch_size"]
    PRETRAINED_DIR = captioning_conf["pretrained_dir"]
    USE_GLOVE = captioning_conf["use_glove"]
    EPOCHS = captioning_conf["epochs"]
    LR = captioning_conf["lr"]
    CLIP = captioning_conf["clip"]
    BUFFER_SIZE = captioning_conf["buffer_size"]
    HORIZONTAL_FLIP = captioning_conf["horizontal_flip"]
    serialized_tokenizer = os.path.join(PRETRAINED_DIR, "tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
        max_length = tokenizer.max_length

    img_name_train_file = os.path.join(PRETRAINED_DIR, "img_name_train.pickle")
    with open(img_name_train_file, 'rb') as handle:
        img_name_train = pickle.load(handle)


    cap_train_file = os.path.join(PRETRAINED_DIR, "cap_train.pickle")
    with open(cap_train_file, 'rb') as handle:
        cap_train = pickle.load(handle)

    zipped = list(zip(img_name_train, cap_train))
    random.shuffle(zipped)
    img_name_train, cap_train = zip(*zipped)



    units = EMBEDDING_DIM*2
    vocab_size = VOCAB_SIZE + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    # Load the numpy files
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        # Random horizontal flip
        if HORIZONTAL_FLIP:
            flipped_coin = np.random.rand()
            if flipped_coin < 0.5:
                img_tensor = np.flip(img_tensor, 1)

        return img_tensor, cap

    img_name_train = tf.convert_to_tensor(img_name_train)
    cap_train = tf.convert_to_tensor(cap_train)
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    # ## Model
    #
    # Fun fact: the decoder below is identical to the one in the example for [Neural Machine Translation with Attention](../sequences/nmt_with_attention.ipynb).
    #
    # The model architecture is inspired by the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.
    #
    # * In this example, you extract the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
    # * You squash that to a shape of (64, 2048).
    # * This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
    # * The RNN (here GRU) attends over the image to predict the next word.
    encoder = CNN_Encoder(EMBEDDING_DIM)
    decoder = RNN_Decoder(EMBEDDING_DIM, units, vocab_size)


    if USE_GLOVE:
        glove_npy_file = os.path.join(PRETRAINED_DIR, f'glove{EMBEDDING_DIM}_V{VOCAB_SIZE + 1}.npy')
        embedding_matrix = np.load(glove_npy_file)
        embedding_layer = Embedding(
            vocab_size,
            EMBEDDING_DIM,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=True,
        )
        decoder.embedding = embedding_layer


    optimizer = tf.keras.optimizers.Adam(learning_rate=LR,clipnorm=CLIP)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    # ## Checkpoint

    checkpoint_path = os.path.join(PRETRAINED_DIR, "checkpoints")
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restoring checkpoint", ckpt_manager.latest_checkpoint)

    # ## Training
    #
    # * You extract the features stored in the respective `.npy` files and then pass those features through the encoder.
    # * The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
    # * The decoder returns the predictions and the decoder hidden state.
    # * The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
    # * Use teacher forcing to decide the next input to the decoder.
    # * Teacher forcing is the technique where the target word is passed as the next input to the decoder.
    # * The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
    #

    loss_plot = []


    # In[31]:


    @tf.function
    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss


    for epoch in tqdm(range(start_epoch, EPOCHS)):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in tqdm(enumerate(dataset)):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        # if epoch % 5 == 0:
        ckpt_manager.save()

        print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
        for i, l in enumerate(loss_plot):
            print("losses_", start_epoch+i, l)
