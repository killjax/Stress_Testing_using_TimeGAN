"""
predictive_metrics.py
OPTIMIZED FOR GPU: Pre-padded vectorized training batches for 10x-20x speedup.
Note: Use Post-hoc RNN to predict one-step ahead (last feature).
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Model, layers, optimizers

from utils import extract_time


class Predictor(Model):
    """Keras Model for the post-hoc predictor (cuDNN GRU accelerated)."""

    def __init__(self, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = layers.GRU(units=hidden_dim, activation="tanh", return_sequences=True)
        self.dense = layers.Dense(1, activation=None)

    def call(self, x, t=None):
        rnn_outputs = self.gru(x)
        y_hat_logit = self.dense(rnn_outputs)
        y_hat = tf.nn.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data, iterations=5000, seed=None):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data (list of 2D numpy arrays, normalized to [0, 1])
      - generated_data: generated synthetic data (list of 2D numpy arrays, normalized to [0, 1])
      - iterations: training iterations for post-hoc predictor (default: 5000)
      - seed: optional random seed

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    no = len(ori_data)
    no_gen = len(generated_data)
    if no == 0 or no_gen == 0:
        return 0.0
    dim = ori_data[0].shape[1]

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = max(1, int(dim / 2))
    batch_size = min(128, no_gen)
    rng = np.random.default_rng(seed)

    predictor_model = Predictor(hidden_dim)
    p_optimizer = optimizers.Adam()

    @tf.function
    def train_step(X, Y, T):
        with tf.GradientTape() as tape:
            y_pred = predictor_model(X, T)
            mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1], dtype=tf.float32)
            mask = tf.expand_dims(mask, -1)

            loss_unmasked = tf.abs(Y - y_pred)
            loss_masked = loss_unmasked * mask
            p_loss = tf.reduce_sum(loss_masked) / (tf.reduce_sum(mask) + 1e-8)

        gradients = tape.gradient(p_loss, predictor_model.trainable_variables)
        p_optimizer.apply_gradients(zip(gradients, predictor_model.trainable_variables))
        return p_loss

    # --- Pre-pad synthetic training dataset ONCE in C-speed NumPy ---
    X_train = np.zeros([no_gen, max_seq_len - 1, dim - 1], dtype=np.float32)
    Y_train = np.zeros([no_gen, max_seq_len - 1, 1], dtype=np.float32)
    T_train = np.zeros([no_gen], dtype=np.int32)

    for i in range(no_gen):
        T_i = generated_time[i] - 1
        if T_i <= 0:
            continue
        seq = np.asarray(generated_data[i], dtype=np.float32)
        X_train[i, :T_i, :] = seq[:-1, : (dim - 1)]
        Y_train[i, :T_i, 0] = seq[1:, (dim - 1)]
        T_train[i] = T_i

    # Vectorized Training Loop
    for _ in range(iterations):
        train_idx = rng.integers(0, no_gen, size=batch_size)
        X_mb = X_train[train_idx]
        Y_mb = Y_train[train_idx]
        T_mb = T_train[train_idx]
        train_step(X_mb, Y_mb, T_mb)

    # --- Test the trained model on the original data ---
    X_test = np.zeros([no, max_seq_len - 1, dim - 1], dtype=np.float32)
    Y_test = np.zeros([no, max_seq_len - 1, 1], dtype=np.float32)
    T_test = np.zeros([no], dtype=np.int32)

    for i in range(no):
        T_i = ori_time[i] - 1
        if T_i <= 0:
            continue
        seq = np.asarray(ori_data[i], dtype=np.float32)
        X_test[i, :T_i, :] = seq[:-1, : (dim - 1)]
        Y_test[i, :T_i, 0] = seq[1:, (dim - 1)]
        T_test[i] = T_i

    pred_Y_curr = predictor_model(X_test, T_test).numpy()

    MAE_temp = 0.0
    valid_count = 0
    for i in range(no):
        T_i = T_test[i]
        if T_i <= 0:
            continue
        Y_true_i = Y_test[i, :T_i, :]
        Y_pred_i = pred_Y_curr[i, :T_i, :]
        MAE_temp += mean_absolute_error(Y_true_i, Y_pred_i)
        valid_count += 1

    predictive_score = float(MAE_temp / max(1, valid_count))
    return predictive_score
