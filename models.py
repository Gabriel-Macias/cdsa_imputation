import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# Position encoder
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = np.expand_dims(pos_encoding, axis = 2) # new line to have (batch, time, measure, d_model)

    return tf.cast(pos_encoding, dtype = tf.float32)

@tf.function
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, time_dim) or (batch_size, 1, 1, n_measure)

def scaled_dot_product_attention(q, k, v, mask = None):
    # (batch_size, num_heads, time, depth) > depth * n_heads = m * d
    matmul_qk = tf.matmul(q, k, transpose_b = True)  # (..., t, t) or (..., m, m)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., t, t)

    output = tf.matmul(attention_weights, v)  # (batch, n_heads, time, depth)

    return output, attention_weights


class MultiHeadAttention(tfl.Layer):
    def __init__(self, d_model, num_heads, extra_dim = 1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.extra_dim = extra_dim

        assert (d_model * extra_dim) % self.num_heads == 0

        self.depth = (d_model * extra_dim) // self.num_heads

        self.wq = tfl.Dense(d_model)
        self.wk = tfl.Dense(d_model)
        self.wv = tfl.Dense(d_model)

        self.dense = tfl.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, attention_dim, depth)
        input: (batch, time, m * d)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "d_model": self.d_model,
            'num_heads': self.num_heads,
            "extra_dim": self.extra_dim})
        return config

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, time, m, d)
        k = self.wk(k)  # (batch_size, time, m, d)
        v = self.wv(v)  # (batch_size, time, m, d)

        q = tf.reshape(q, (batch_size, -1, self.d_model * self.extra_dim)) # (batch_size, time, m * d)
        k = tf.reshape(k, (batch_size, -1, self.d_model * self.extra_dim)) # (batch_size, time, m * d)
        v = tf.reshape(v, (batch_size, -1, self.d_model * self.extra_dim)) # (batch_size, time, m * d)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, time, depth) > depth * n_heads = m * d
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, time, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, time, depth)

        # scaled_attention.shape  ==  (b, h, t, depth)
        # attention_weights.shape ==  (b, h, t, t)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (b, t, h, dep)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.extra_dim * self.d_model))  # (b, t, m * d)
        concat_attention = tf.reshape(concat_attention, (batch_size, -1, self.extra_dim, self.d_model)) # (b, t, m, d)

        output = self.dense(concat_attention)  # (batch_size, t, m, d)

        return output, attention_weights

def point_wise_feed_forward_network(output_dim, hidden_dim):
    return tf.keras.Sequential([
        tfl.Dense(hidden_dim, activation = 'selu'),  # (batch, ..., hidden_dim)
        tfl.Dense(output_dim)                        # (batch, ..., output_dim)
    ])


class EncoderLayer(tfl.Layer):
    def __init__(self, time_dim, m_dim, d_model, num_heads, dff, rate = 0.1, imputation_mode = False):
        super(EncoderLayer, self).__init__()

        self.imp = imputation_mode
        self.d_dim = d_model
        self.t_dim = time_dim
        self.m_dim = m_dim
        self.dff = dff
        self.num_heads = num_heads
        self.rate = rate

        self.mha_t = MultiHeadAttention(d_model, num_heads, m_dim)
        self.mha_m = MultiHeadAttention(d_model, num_heads, time_dim)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tfl.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tfl.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = tfl.Dropout(rate)
        self.dropout2 = tfl.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "time_dim": self.t_dim,
            'm_dim': self.m_dim,
            'd_model': self.d_dim,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
            "imputation_mode": self.imp})
        return config

    def call(self, x, training):
        # x.shape = (batch_size, time_dim, n_measurements, model_dim)

        # Create the reshaped matrices X_T, X_M
        x_shape = tf.shape(x)
        x_m = tf.transpose(x, perm = [0, 2, 1, 3])  # (batch_size, n_measurements, time_dim, model_dim)

        if self.imp:
            mask_t = tf.eye(x_shape[1])[tf.newaxis, tf.newaxis, ...] # (batch, n_heads, t, t)
            mask_m = tf.eye(x_shape[2])[tf.newaxis, tf.newaxis, ...] # (batch, n_heads, m, m)
        else:
            mask_t = None
            mask_m = None

        attn_output, _ = self.mha_t(x, x, x, mask_t)                  # (batch_size, t, m, d)
        attn_output = tf.transpose(attn_output, perm = [0, 2, 1, 3])  # (batch_size, m, t, d)
        attn_output, _ = self.mha_m(attn_output, x_m, x_m, mask_m)    # (batch_size, m, t, d), (batch_size, h, m, m)

        attn_output = tf.transpose(attn_output, perm = [0, 2, 1, 3])  # (batch_size, t, m, d)

        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)  # (B, T, M, D)

        ffn_output = self.ffn(out1)  # (B, T, M, D)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (B, T, M, D)

        return out2

class Encoder(tfl.Layer):
    def __init__(self, num_layers, time_dim, m_dim, d_model, num_heads, dff, max_time_step, rate = 0.1, imputation_mode = False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.imp_mode = imputation_mode
        self.num_heads = num_heads
        self.dff = dff
        self.t_dim = time_dim
        self.m_dim = m_dim
        self.max_t_step = max_time_step
        self.rate = rate

        self.embedding = tfl.Dense(d_model)
        self.pos_encoding = positional_encoding(max_time_step, self.d_model)

        self.enc_layers = [EncoderLayer(time_dim, m_dim, d_model, num_heads, dff, rate, imputation_mode) for _ in range(num_layers)]

        self.dropout = tfl.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_layers": self.num_layers,
            "time_dim": self.t_dim,
            'm_dim': self.m_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            "max_time_step": self.max_t_step,
            'rate': self.rate,
            "imputation_mode": self.imp_mode})
        return config

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        x = tf.expand_dims(x, axis = -1) # (batch_size, time_dim, n_measurements, 1)
        x = self.embedding(x) # (batch_size, time_dim, n_measurements, d_model)

        x += self.pos_encoding[:, :seq_len, :, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (batch_size, time, n_signal, d_model)


def get_imp_model(model_dim, n_layers, n_heads, dff, time_dim, n_signals):
    opt = Adam(CustomSchedule(model_dim), beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)
    # opt = Adam(learning_rate = 1e-2, clipnorm = 1.0)

    imp_input = tfl.Input(shape = (time_dim, n_signals))

    x = Encoder(num_layers = n_layers, d_model = model_dim, time_dim = time_dim, m_dim = n_signals, num_heads = n_heads,
                dff = dff, max_time_step = time_dim, rate = 0.1, imputation_mode = True)(
        imp_input)  # (batch, time, measure, d_model)

    # x = tfl.Reshape((time_dim, model_dim * n_signals))(x) # (batch, time, measure * d_model)
    output_layer = tfl.Dense(1)(x)  # (batch, time, measure, 1)
    imp_model = Model(inputs = imp_input, outputs = output_layer)

    imp_model.compile(optimizer = opt, loss = imputation_rmse_loss)
    print("The number of parameters in the model: {:,d}".format(imp_model.count_params()))

    return imp_model

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class lr_scheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr = 1e-3):
        self.initial_value = initial_lr
    def on_epoch_end(self, epoch, logs = {}):
        if (epoch + 1) > 5:
            old_lr = self.model.optimizer.learning_rate.read_value()
            new_lr = self.initial_value / np.sqrt(epoch)
            print("Epoch: {}. Reducing learning rate from {:2.6f} to {:2.6f}\n".format(epoch + 1, old_lr, new_lr))
            self.model.optimizer.learning_rate.assign(new_lr)

# ================================== Loss used for the imputation part
def imputation_rmse_loss(y_true, y_pred):
    real_mask = tf.cast(tf.not_equal(y_true, 0), dtype = tf.float32)
    square_diffs = real_mask * ((y_true - y_pred) ** 2)
    return tf.sqrt(tf.reduce_sum(square_diffs) / tf.reduce_sum(real_mask))