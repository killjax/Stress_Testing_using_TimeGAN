"""utils.py (Modified for Static + Temporal Features)

(1) train_test_divide: Divide train and test data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: Vectorized random vector generator.
(5) batch_generator: Mini-batch generator for both static and temporal features.
(6) batch_generator1: Mini-batch generator for temporal features.
"""

import numpy as np
import tensorflow as tf


def train_test_divide(
    data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8, seed=None
):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
      - seed: optional integer random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Divide train/test index (original data)
    no = len(data_x)
    idx = rng.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [np.asarray(data_x[i], dtype=np.float32) for i in train_idx]
    test_x = [np.asarray(data_x[i], dtype=np.float32) for i in test_idx]
    train_t = [int(data_t[i]) for i in train_idx]
    test_t = [int(data_t[i]) for i in test_idx]

    # Divide train/test index (synthetic data)
    no_hat = len(data_x_hat)
    idx_hat = rng.permutation(no_hat)
    train_idx_hat = idx_hat[: int(no_hat * train_rate)]
    test_idx_hat = idx_hat[int(no_hat * train_rate) :]

    train_x_hat = [np.asarray(data_x_hat[i], dtype=np.float32) for i in train_idx_hat]
    test_x_hat = [np.asarray(data_x_hat[i], dtype=np.float32) for i in test_idx_hat]
    train_t_hat = [int(data_t_hat[i]) for i in train_idx_hat]
    test_t_hat = [int(data_t_hat[i]) for i in test_idx_hat]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data (temporal)

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        seq_len_i = len(data[i])
        max_seq_len = max(max_seq_len, seq_len_i)
        time.append(seq_len_i)

    return time, max_seq_len


@tf.keras.utils.register_keras_serializable()
class LayerNormLSTMCell(tf.keras.layers.LSTMCell):
    """LSTMCell with Layer Normalization."""

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            **kwargs,
        )
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        super().build(input_shape)
        self.ln_i = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_c = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_o = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_cell_state = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        z = tf.matmul(inputs, self.kernel)
        z += tf.matmul(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = tf.nn.bias_add(z, self.bias)
        z_i, z_f, z_c, z_o = tf.split(z, 4, axis=1)
        z_i = self.ln_i(z_i)
        z_f = self.ln_f(z_f)
        z_c = self.ln_c(z_c)
        z_o = self.ln_o(z_o)
        i = self.recurrent_activation(z_i)
        f = self.recurrent_activation(z_f)
        c = self.activation(z_c)
        o = self.recurrent_activation(z_o)
        new_c = f * c_tm1 + i * c
        new_c_norm = self.ln_cell_state(new_c)
        new_h = o * self.activation(new_c_norm)
        return new_h, [new_h, new_c]

    def get_config(self):
        config = super().get_config()
        config.update({"layer_norm_epsilon": self.layer_norm_epsilon})
        return config


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.

    Args:
      - module_name: gru, lstm, or lstmLN

    Returns:
      - cell: RNN Cell
    """
    assert module_name in ["gru", "lstm", "lstmLN"]

    if module_name == "gru":
        cell = tf.keras.layers.GRUCell(units=hidden_dim, activation="tanh")
    elif module_name == "lstm":
        cell = tf.keras.layers.LSTMCell(units=hidden_dim, activation="tanh")
    elif module_name == "lstmLN":
        cell = LayerNormLSTMCell(units=hidden_dim, activation="tanh")
    return cell


def rnn_layer(module_name, hidden_dim, return_sequences=True):
    """Returns a cuDNN-accelerated Keras RNN layer (GRU/LSTM) or LayerNormLSTM RNN.

    Using native `tf.keras.layers.GRU` and `tf.keras.layers.LSTM` enables fused
    cuDNN GPU kernels (50x-100x faster than wrapping GRUCell/LSTMCell in `layers.RNN`).

    Args:
      - module_name: 'gru', 'lstm', or 'lstmLN'
      - hidden_dim: number of hidden units
      - return_sequences: whether to return the full sequence or last output

    Returns:
      - layer: Keras recurrent layer
    """
    assert module_name in ["gru", "lstm", "lstmLN"]

    if module_name == "gru":
        return tf.keras.layers.GRU(
            units=hidden_dim,
            activation="tanh",
            return_sequences=return_sequences,
        )
    elif module_name == "lstm":
        return tf.keras.layers.LSTM(
            units=hidden_dim,
            activation="tanh",
            return_sequences=return_sequences,
        )
    else:
        cell = LayerNormLSTMCell(units=hidden_dim, activation="tanh")
        return tf.keras.layers.RNN(cell, return_sequences=return_sequences)


def random_generator(batch_size, z_dim_s, z_dim_x, T_mb, max_seq_len, rng=None):
    """Vectorized random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim_s: dimension of static random vector
      - z_dim_x: dimension of temporal random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length
      - rng: optional numpy Generator instance

    Returns:
      - Z_S_mb: generated static random vector of shape (batch_size, z_dim_s)
      - Z_T_mb: generated temporal random vector of shape (batch_size, max_seq_len, z_dim_x)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Static random vector
    Z_S_mb = rng.uniform(0.0, 1.0, size=(batch_size, z_dim_s)).astype(np.float32)

    # Vectorized temporal random vector masked by sequence length T_mb
    Z_T_mb = rng.uniform(0.0, 1.0, size=(batch_size, max_seq_len, z_dim_x)).astype(
        np.float32
    )
    t_arr = np.asarray(T_mb, dtype=np.int32)[:, None]
    mask = (np.arange(max_seq_len, dtype=np.int32)[None, :] < t_arr)[:, :, None]
    Z_T_mb *= mask

    return Z_S_mb, Z_T_mb


def batch_generator(data_s, data_x, time, batch_size, rng=None):
    """Mini-batch generator for static and temporal features.

    Args:
      - data_s: static time-series data
      - data_x: temporal time-series data
      - time: time information
      - batch_size: the number of samples in each batch
      - rng: optional numpy Generator instance

    Returns:
      - S_mb: static data in each batch
      - X_mb: temporal data in each batch
      - T_mb: time information in each batch
    """
    if rng is None:
        rng = np.random.default_rng()

    no = len(data_x)
    actual_batch = min(batch_size, no)
    train_idx = rng.choice(no, size=actual_batch, replace=(no < batch_size))

    S_mb = [data_s[i] for i in train_idx]
    X_mb = [data_x[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return S_mb, X_mb, T_mb


def batch_generator1(data, time, batch_size, rng=None):
    """Mini-batch generator for temporal features only.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch
      - rng: optional numpy Generator instance

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    if rng is None:
        rng = np.random.default_rng()

    no = len(data)
    actual_batch = min(batch_size, no)
    train_idx = rng.choice(no, size=actual_batch, replace=(no < batch_size))

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb
