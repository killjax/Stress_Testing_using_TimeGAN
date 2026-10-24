"""
discriminative_metrics.py
OPTIMIZED FOR GPU: Vectorized data loading to prevent CPU bottlenecks.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import Model, layers, optimizers

from utils import extract_time, train_test_divide


class Discriminator(Model):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = layers.GRU(units=hidden_dim, activation="tanh", return_sequences=False)
        self.dense = layers.Dense(1, activation=None)

    def call(self, x, t=None):
        d_last_states = self.gru(x)
        y_hat_logit = self.dense(d_last_states)
        y_hat = tf.nn.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data, iterations=2000, seed=None):
    """Use post-hoc RNN to classify original data and synthetic data.

    Args:
      - ori_data: original data (list of 2D numpy arrays, normalized to [0, 1])
      - generated_data: generated synthetic data (list of 2D numpy arrays, normalized to [0, 1])
      - iterations: training iterations for post-hoc discriminator (default: 2000)
      - seed: optional random seed

    Returns:
      - discriminative_score: np.abs(0.5 - accuracy) in [0, 0.5] (lower is better)
    """
    if len(ori_data) == 0 or len(generated_data) == 0:
        return 0.0

    dim = ori_data[0].shape[1]

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = max(1, int(dim / 2))
    batch_size = min(128, len(ori_data), len(generated_data))
    rng = np.random.default_rng(seed)

    discriminator_model = Discriminator(hidden_dim)
    d_optimizer = optimizers.Adam()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(
        ori_data, generated_data, ori_time, generated_time, seed=seed
    )

    def pad_data(data_list, max_len, feat_dim):
        padded = np.zeros((len(data_list), max_len, feat_dim), dtype=np.float32)
        for i, seq in enumerate(data_list):
            if len(seq) > 0:
                padded[i, : len(seq), :] = np.asarray(seq, dtype=np.float32)
        return padded

    train_x_padded = pad_data(train_x, max_seq_len, dim)
    train_x_hat_padded = pad_data(train_x_hat, max_seq_len, dim)
    train_t = np.array(train_t, dtype=np.int32)
    train_t_hat = np.array(train_t_hat, dtype=np.int32)

    @tf.function
    def train_step(X_mb, T_mb, X_hat_mb, T_hat_mb):
        with tf.GradientTape() as tape:
            y_logit_real, _ = discriminator_model(X_mb, T_mb)
            y_logit_fake, _ = discriminator_model(X_hat_mb, T_hat_mb)

            d_loss_real = bce_loss(tf.ones_like(y_logit_real), y_logit_real)
            d_loss_fake = bce_loss(tf.zeros_like(y_logit_fake), y_logit_fake)
            d_loss = d_loss_real + d_loss_fake

        gradients = tape.gradient(d_loss, discriminator_model.trainable_variables)
        d_optimizer.apply_gradients(
            zip(gradients, discriminator_model.trainable_variables)
        )
        return d_loss

    eff_bs = min(batch_size, len(train_x_padded), len(train_x_hat_padded))
    for _ in range(iterations):
        idx_real = rng.integers(0, len(train_x_padded), size=eff_bs)
        idx_fake = rng.integers(0, len(train_x_hat_padded), size=eff_bs)

        X_mb = train_x_padded[idx_real]
        T_mb = train_t[idx_real]
        X_hat_mb = train_x_hat_padded[idx_fake]
        T_hat_mb = train_t_hat[idx_fake]

        train_step(X_mb, T_mb, X_hat_mb, T_hat_mb)

    test_x_padded = pad_data(test_x, max_seq_len, dim)
    test_x_hat_padded = pad_data(test_x_hat, max_seq_len, dim)
    test_t = np.array(test_t, dtype=np.int32)
    test_t_hat = np.array(test_t_hat, dtype=np.int32)

    _, y_pred_real_curr = discriminator_model(test_x_padded, test_t)
    _, y_pred_fake_curr = discriminator_model(test_x_hat_padded, test_t_hat)

    y_pred_real_curr = y_pred_real_curr.numpy()
    y_pred_fake_curr = y_pred_fake_curr.numpy()

    y_pred_final = np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0).reshape(
        -1
    )
    y_label_final = np.concatenate(
        (
            np.ones(len(y_pred_real_curr)),
            np.zeros(len(y_pred_fake_curr)),
        ),
        axis=0,
    )

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = float(np.abs(0.5 - acc))

    return discriminative_score
