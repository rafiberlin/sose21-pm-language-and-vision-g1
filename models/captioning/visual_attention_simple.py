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
import models.utils.util as util
from models.model import RNN_Decoder, CNN_Encoder, BahdanauAttention



# ## Download and prepare the MS-COCO dataset
#
# You will use the [MS-COCO dataset](http://cocodataset.org/#home) to train our model. The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. The code below downloads and extracts the dataset automatically.
#
# **Caution: large download ahead**. You'll use the training set, which is a 13GB file.

# In[3]:
# Download caption annotation files
# CHANGE THIS!
base_dir = '/home/rafi/_datasets/MSCOCO/'
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

# ## Optional: limit the size of the training set
# To speed up training for this tutorial, you'll use a subset of 30,000 captions and their corresponding images to train our model. Choosing to use more data would result in improved captioning quality.

# In[4]:


with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# In[5]:


# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = os.path.join(PATH, '%012d.jpg' % (val['image_id']))
    image_path_to_caption[image_path].append(caption)

# In[6]:


image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

# Select the first 6000 image_paths from the shuffled set.
# Approximately each image id has 5 captions associated with it, so that will
# lead to 30,000 examples.
train_image_paths = image_paths[:]
# train_image_paths = image_paths[:]
print(len(train_image_paths))

# In[7]:


train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

# In[8]:


print(train_captions[0])
Image.open(img_name_vector[0])


# ## Preprocess the images using InceptionV3
# Next, you will use InceptionV3 (which is pretrained on Imagenet) to classify each image. You will extract features from the last convolutional layer.
#
# First, you will convert the images into InceptionV3's expected format by:
# * Resizing the image to 299px by 299px
# * [Preprocess the images](https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage) using the [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input) method to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.

# In[9]:




# ## Initialize InceptionV3 and load the pretrained Imagenet weights
#
# Now you'll create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. The shape of the output of this layer is ```8x8x2048```. You use the last convolutional layer because you are using attention in this example. You don't perform this initialization during training because it could become a bottleneck.
#
# * You forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
# * After all the images are passed through the network, you save the dictionary to disk.
#

# In[10]:


image_features_extract_model = util.get_image_feature_extractor()

BATCH_SIZE = 128

from tqdm import tqdm

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
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
VOCAB_SIZE = 5000

serialized_tokenizer = "./checkpoints/train/tokenizer.pickle"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# In[17]:

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)
tokenizer.max_length = max_length
with open(serialized_tokenizer, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Split the data into training and testing

# In[18]:


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

# img_name_val = []
# cap_val = []
# for imgv in img_name_val_keys:
#     capv_len = len(img_to_cap_vector[imgv])
#     img_name_val.extend([imgv] * capv_len)
#     cap_val.extend(img_to_cap_vector[imgv])

# In[19]:

zipped = list(zip(img_name_train, cap_train))
random.shuffle(zipped)
img_name_train, cap_train = zip(*zipped)

# len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

# ## Create a tf.data dataset for training
#

#  Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.

# In[20]:


# Feel free to change these parameters according to your system's configuration


BUFFER_SIZE = 29952
embedding_dim = 256
units = 512
vocab_size = VOCAB_SIZE + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
LR=0.0005
CLIP=0.0005

# In[21]:


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# In[22]:

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

# In[23]:



encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# In[27]:


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

# In[28]:


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

# In[29]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

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

# In[30]:


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
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


# In[32]:


EPOCHS = 20

for epoch in tqdm(range(start_epoch, EPOCHS)):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
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
