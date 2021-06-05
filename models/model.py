import tensorflow as tf


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


class CoattentionModel(tf.keras.layers.Layer):
    def __init__(self, num_classes, question_vocab_size, image_embedding):  # , num_embeddings, num_classes, embed_dim=512, k=30
        super().__init__()
        self.num_classes = num_classes
        self.image_embedding = image_embedding
        self.dropout = 0.3
        self.question_vocab_size = question_vocab_size

        self.image_dense = tf.keras.layers.Dense(self.image_embedding,
                                                 kernel_initializer=tf.keras.initializers.glorot_normal(seed=15))
        self.image_corr = tf.keras.layers.Dense(self.image_embedding,
                                                kernel_initializer=tf.keras.initializers.glorot_normal(seed=29))

        self.image_atten_dense = tf.keras.layers.Dense(self.image_embedding,
                                                       kernel_initializer=tf.keras.initializers.glorot_uniform(seed=17))
        self.question_atten_dens = tf.keras.layers.Dense(self.image_embedding,
                                                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                             seed=28))
        self.question_atten_dropout = tf.keras.layers.Dropout(self.dropout)
        self.image_atten_dropout = tf.keras.layers.Dropout(self.dropout)

        self.ques_atten = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=21))

        self.img_atten = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=33))

        self.embed = tf.keras.layers.Embedding(self.question_vocab_size, self.image_embedding,
                                               embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0,
                                                                                                         stddev=1,
                                                                                                         seed=23))

        self.unigram_conv = tf.keras.layers.Conv1D(filters=self.image_embedding, kernel_size=1, strides=1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=41))
        self.bigram_conv = tf.keras.layers.Conv1D(filters=self.image_embedding, kernel_size=2, strides=1, padding='same',
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=58),
                                                  dilation_rate=2)
        self.trigram_conv = tf.keras.layers.Conv1D(filters=self.image_embedding, kernel_size=3, strides=1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=89),
                                                   dilation_rate=2)
        self.max_pool = tf.keras.layers.MaxPool2D((3, 1))
        self.phrase_dropout = tf.keras.layers.Dropout(self.dropout)

        self.lstm = tf.keras.layers.LSTM(units=self.image_embedding, return_sequences=True, dropout=self.dropout,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                                         recurrent_initializer=tf.keras.initializers.orthogonal(seed=54))

        self.tanh = tf.keras.layers.Activation('tanh')
        self.softmax = tf.keras.layers.Activation('softmax')

        self.W_w_dropout = tf.keras.layers.Dropout(self.dropout)
        self.W_p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.W_s_dropout = tf.keras.layers.Dropout(self.dropout)

        self.W_w = tf.keras.layers.Dense(units=self.image_embedding,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=32),
                                         input_shape=(self.image_embedding,))
        self.W_p = tf.keras.layers.Dense(units=self.image_embedding,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=49),
                                         input_shape=(2 * self.image_embedding,))
        self.W_s = tf.keras.layers.Dense(units=self.image_embedding,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(seed=31),
                                         input_shape=(2 * self.image_embedding,))

        self.fc1_Dense = tf.keras.layers.Dense(units=2 * self.image_embedding, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=84))
        self.fc1_dropout = tf.keras.layers.Dropout(self.dropout)

        self.fc = tf.keras.layers.Dense(units=self.num_classes, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=91),
                                        input_shape=(self.image_embedding,))

        return

    def get_config(self):
        #config = super(tf.keras.layers.Layer).get_config().copy()
        config = {
            "num_classes": self.num_classes,
            "image_embedding":self.image_embedding,
            "question_vocab_size": self.question_vocab_size
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, image, question):  # Image: B x 196 x 512
        image = self.image_dense(image)
        image = self.tanh(image)

        words = self.embed(question)  # Words: B x L x 51
        unigrams = tf.expand_dims(self.tanh(self.unigram_conv(words)), 1)  # B x L x 512
        bigrams = tf.expand_dims(self.tanh(self.bigram_conv(words)), 1)  # B x L x 512
        trigrams = tf.expand_dims(self.tanh(self.trigram_conv(words)), 1)  # B x L x 512

        phrase = tf.squeeze(self.max_pool(tf.concat((unigrams, bigrams, trigrams), 1)), axis=1)  # B x L x 512
        phrase = self.tanh(phrase)
        phrase = self.phrase_dropout(phrase)

        hidden = None
        sentence = self.lstm(phrase)  # B x L x 512
        v_word, q_word = self.co_attention(image, words)
        v_phrase, q_phrase = self.co_attention(image, phrase)
        v_sent, q_sent = self.co_attention(image, sentence)

        h_w = self.tanh(self.W_w(self.W_w_dropout(q_word + v_word)))
        h_p = self.tanh(self.W_p(self.W_p_dropout(tf.concat(((q_phrase + v_phrase), h_w), axis=1))))
        h_s = self.tanh(self.W_s(self.W_s_dropout(tf.concat(((q_sent + v_sent), h_p), axis=1))))

        fc1 = self.fc1_Dense(self.fc1_dropout(h_s))
        logits = self.fc(fc1)

        return logits

    def co_attention(self, img_feat, ques_feat):  # V : B x 512 x 196, Q : B x L x 512

        img_corr = self.image_corr(img_feat)
        weight_matrix = tf.keras.backend.batch_dot(ques_feat, img_corr, axes=(2, 2))
        weight_matrix = self.tanh(weight_matrix)

        ques_embed = self.image_atten_dense(ques_feat)
        img_embed = self.question_atten_dens(img_feat)

        transform_img = tf.keras.backend.batch_dot(weight_matrix, img_embed)

        ques_atten_sum = self.tanh(transform_img + ques_embed)
        ques_atten_sum = self.question_atten_dropout(ques_atten_sum)
        ques_atten = self.ques_atten(ques_atten_sum)
        ques_atten = tf.keras.layers.Reshape((ques_atten.shape[1],))(ques_atten)
        ques_atten = self.softmax(ques_atten)

        # atten for image feature
        transform_ques = tf.keras.backend.batch_dot(weight_matrix, ques_embed, axes=(1, 1))
        img_atten_sum = self.tanh(transform_ques + img_embed)
        img_atten_sum = self.image_atten_dropout(img_atten_sum)
        img_atten = self.img_atten(img_atten_sum)
        img_atten = tf.keras.layers.Reshape((img_atten.shape[1],))(img_atten)
        img_atten = self.softmax(img_atten)

        ques_atten = tf.keras.layers.Reshape((1, ques_atten.shape[1]))(ques_atten)
        img_atten = tf.keras.layers.Reshape((1, img_atten.shape[1]))(img_atten)

        ques_atten_feat = tf.keras.backend.batch_dot(ques_atten, ques_feat)
        ques_atten_feat = tf.keras.layers.Reshape((ques_atten_feat.shape[-1],))(ques_atten_feat)

        img_atten_feat = tf.keras.backend.batch_dot(img_atten, img_feat)
        img_atten_feat = tf.keras.layers.Reshape((img_atten_feat.shape[-1],))(img_atten_feat)

        return img_atten_feat, ques_atten_feat