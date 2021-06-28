# From https://www.tensorflow.org/tutorials/text/image_captioning
# You need to downlod the MSCOO dataset and the annotaions, change the variable base_dir accordingly...
# If you want to debug (step in where the loss is calculated), comment out the annotation @tf.function

import tensorflow as tf
import pickle
import numpy as np
import os
import avatar_models.utils.util as util
from avatar_models.model import RNN_Decoder, CNN_Encoder
from config.util import get_config

class CaptionWithAttention():

    def __init__(self):
        # Choose the top 5000 words from the vocabulary

        conf = get_config()

        captioning_conf = conf["captioning"]["attention"]
        tf_gpu = captioning_conf["tensorflow_gpu_name"]
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        if type(tf_gpu) is str and tf_gpu.startswith("/physical_device:GPU:"):
            tf.config.set_visible_devices([ d for d in physical_devices if d[0] == tf_gpu], 'GPU')
            print("Tensorflow GPU Name Supported", tf_gpu)
        else:
            tf.config.set_visible_devices([], 'GPU')
            print("No GPU Support for Tensorflow")

        VOCAB_SIZE = captioning_conf["vocab_size"]
        EMBEDDING_DIM = captioning_conf["embedding_dim"]
        PRETRAINED_DIR = captioning_conf["pretrained_dir"]
        self.BEAM_SIZE = captioning_conf["beam_size"]
        self.PAD_IMAGES = captioning_conf["pad_images"]

        serialized_tokenizer = os.path.join(PRETRAINED_DIR,
                                            "tokenizer.pickle")
        with open(serialized_tokenizer, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            self.max_length = self.tokenizer.max_length

        units = EMBEDDING_DIM*2
        vocab_size = VOCAB_SIZE + 1

        self.encoder = CNN_Encoder(EMBEDDING_DIM)
        self.decoder = RNN_Decoder(EMBEDDING_DIM, units, vocab_size)

        optimizer = tf.keras.optimizers.Adam()

        # checkpoint_path = "./checkpoints/train"

        checkpoint_path = os.path.join(PRETRAINED_DIR, "checkpoints")
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder,
                                   optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print("Restore :", ckpt_manager.latest_checkpoint)
        else:
            print("WARNING : no model loaded!")
        self.image_features_extract_model = util.get_image_feature_extractor()

    def infer(self, image):

        beam_size = self.BEAM_SIZE

        hidden = self.decoder.reset_state(batch_size=beam_size)
        if self.PAD_IMAGES:
            temp_input = tf.expand_dims(util.load_image_with_pad(image)[0], 0)
        else:
            temp_input = tf.expand_dims(util.load_image(image)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.stack([tf.reshape(img_tensor_val, (
            -1,
            img_tensor_val.shape[3])) for i in range(beam_size)])

        features = self.encoder(img_tensor_val)
        END_IDX = self.tokenizer.word_index['<end>']
        start_tensor = tf.stack([[self.tokenizer.word_index['<start>']] for i in range(beam_size)])
        predictions, hidden, _ = self.decoder(start_tensor,
                                         features,
                                         hidden)
        predictions = tf.nn.log_softmax(predictions)

        candidates_log_prob, candidate_indices = tf.math.top_k(predictions, k=beam_size)

        input_list = [[candidate_indices[0][i]] for i in range(beam_size)]
        previous_predictions = tf.stack([[candidates_log_prob[0][i]] for i in range(beam_size)])
        preds = {i: np.zeros(self.max_length, dtype=int) for i in range(beam_size)}
        for i in range(beam_size):
            preds[i][0] = candidate_indices[0][i]

        dec_input = tf.stack(input_list)

        candidates = []

        for step in range(1, self.max_length):

            predictions, hidden, _ = self.decoder(dec_input,
                                             features,
                                             hidden)

            predictions = tf.nn.log_softmax(predictions)

            candidates_log_prob, candidate_indices = tf.math.top_k(predictions, k=beam_size)
            candidate_indices = tf.reshape(candidate_indices, -1)
            # Calculates the likelihood of all candidates so far
            candidates_log_prob = tf.reshape(candidates_log_prob + previous_predictions, -1)
            current_top_candidates, current_top_candidates_idx = tf.math.top_k(candidates_log_prob, k=beam_size)

            # Do the mapping best candidate and "source" of the best candidates
            k_idx = tf.gather(candidate_indices, current_top_candidates_idx)
            prev_idx = tf.cast(tf.math.floor(current_top_candidates_idx / beam_size), tf.int32)

            # Modify the hidden states accordingly
            hidden = tf.gather(hidden, prev_idx, axis=0)

            # Overwrite the previous predictions due to the new best candidates
            preds = {i: preds[prev_idx.numpy()[i]].copy() for i in range(prev_idx.shape[0])}
            stop_idx = []
            for i in range(k_idx.shape[0]):
                preds[i][step] = k_idx[i]
                if k_idx[i] == END_IDX:
                    stop_idx.append(i)

            # remove all finished captions and adjust all tensors accordingly...
            if len(stop_idx):
                for i in reversed(sorted(stop_idx)):
                    candidate = preds.pop(i)
                    loss = current_top_candidates[i]
                    length = int(tf.where(candidate == END_IDX)) + 1
                    normalized_loss = loss / float(length)
                    candidates.append((candidate, normalized_loss))
                beam_size = beam_size - len(stop_idx)
                if beam_size > 0:
                    left_idx = tf.convert_to_tensor([i for i in range(k_idx.shape[0]) if i not in stop_idx])
                    k_idx = tf.convert_to_tensor([k_idx[i] for i in range(k_idx.shape[0]) if i not in stop_idx])
                    current_top_candidates = tf.convert_to_tensor(
                        [current_top_candidates[i] for i in range(current_top_candidates.shape[0]) if
                         i not in stop_idx])
                    hidden = tf.gather(hidden, left_idx)
                    features = tf.gather(features, left_idx)
                    # now that the finished sentences have been removed, we need to update the predictions dict accordingly
                    for i, key in enumerate(sorted(preds.keys())):
                        preds[i] = preds.pop(key)
                else:
                    break  # No sequences unfinished

            dec_input = tf.expand_dims(tf.identity(k_idx), axis=1)
            previous_predictions = tf.expand_dims(current_top_candidates, axis=1)

        if len(candidates) > 0:
            result, _ = max(candidates, key=lambda c: c[1])
        else:
            result = preds[0]

        result = [self.tokenizer.index_word[i] for i in result if i != 0]
        if result[-1] == "<end>":
            result = result[:-1]

        caption = ' '.join(result)
        return caption

    # def evaluate(image):
    #     attention_features_shape = 64
    #     attention_plot = np.zeros((max_length, attention_features_shape))
    #
    #     hidden = decoder.reset_state(batch_size=1)
    #
    #     temp_input = tf.expand_dims(util.load_image(image)[0], 0)
    #     img_tensor_val = image_features_extract_model(temp_input)
    #     img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
    #                                                  -1,
    #                                                  img_tensor_val.shape[3]))
    #
    #     features = encoder(img_tensor_val)
    #
    #     dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    #     result = []
    #     result_id = []
    #
    #     for i in range(max_length):
    #         predictions, hidden, attention_weights = decoder(dec_input,
    #                                                          features,
    #                                                          hidden)
    #         predictions = tf.nn.log_softmax(predictions)
    #         attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
    #         predicted_id = tf.math.argmax(predictions[0]).numpy()
    #         result_id.append(predicted_id)
    #         # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
    #         result.append(tokenizer.index_word[predicted_id])
    #
    #         if tokenizer.index_word[predicted_id] == '<end>':
    #             return result, attention_plot
    #
    #         dec_input = tf.expand_dims([predicted_id], 0)
    #
    #     attention_plot = attention_plot[:len(result), :]
    #     print(result_id)
    #     return result, attention_plot



# model to unpack at this level:
# https://drive.google.com/file/d/1d2ZH7699DDStrJt5EOsFneT1XWGgcNRj/view?usp=sharing
# at the same level as this scrip, you should see directory called checkpoints/
if __name__ == "__main__":


    model = CaptionWithAttention()
    image_url = 'https://tensorflow.org/images/surf.jpg'
    # image_url = 'http://localhost:8000/a/atrium/home/ADE_train_00001860.jpg'
    last_char_index = image_url.rfind("/")
    url_shards = image_url.split("://")
    image_path = image_url
    if len(url_shards) == 2:
        image_path = url_shards[1]
    if not os.path.isfile(image_path) and not os.path.isfile(image_url):
        image_name = image_url[last_char_index + 1:]
        image_path = tf.keras.utils.get_file(image_name, origin=image_url)

    result = model.infer(image_path)

    print('Prediction Caption:', result)

