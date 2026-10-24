import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Input,
    Lambda,
    Layer,
    RepeatVector,
    RNN,
)
from tensorflow.keras.models import Model

from data_loading import enforce_financial_consistency
from utils import (
    LayerNormLSTMCell,
    batch_generator,
    extract_time,
    random_generator,
    rnn_cell,
    rnn_layer,
)


# Helper function
def MinMaxScaler(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data (can be 2D for static, 3D for temporal)

    Returns:
      - norm_data: normalized data in [0, 1]
      - min_val: minimum values (for renormalization)
      - max_val: range values (max - min, named max_val for backward compatibility)
    """
    if isinstance(data, list):
        data = np.asarray(data, dtype=np.float32)
    else:
        data = np.asarray(data, dtype=np.float32)

    if data.ndim == 3:
        # Temporal data (samples, seq_len, features)
        min_val = np.min(np.min(data, axis=0), axis=0)
        shifted = data - min_val
        range_val = np.max(np.max(shifted, axis=0), axis=0)
        norm_data = shifted / (range_val + 1e-7)
    elif data.ndim == 2:
        # Static data (samples, features)
        min_val = np.min(data, axis=0)
        shifted = data - min_val
        range_val = np.max(shifted, axis=0)
        norm_data = shifted / (range_val + 1e-7)
    else:
        raise ValueError("Data must be 2D (static) or 3D (temporal)")

    return norm_data.astype(np.float32), min_val.astype(np.float32), range_val.astype(np.float32)


# --- Custom Layers & Legacy Lambda Support ---
def _sequence_mask_lambda(t, maxlen):
    import tensorflow

    return tensorflow.sequence_mask(t, maxlen=maxlen)


@tf.keras.utils.register_keras_serializable()
class SequenceMaskLayer(Layer):
    """Serializable Keras layer applying a multiplicative float sequence mask."""

    def __init__(self, maxlen, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = int(maxlen)

    def call(self, inputs):
        x, t = inputs
        mask = tf.sequence_mask(t, maxlen=self.maxlen, dtype=x.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        return x * mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen})
        return config


def build_embedder(max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name):
    """Builds the Embedder Keras Model."""
    X_S_input = Input(shape=(dim_s,), name="S_input")
    X_T_input = Input(shape=(max_seq_len, dim_x), name="X_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    # Static Embedder (e_S)
    H_S = Dense(hidden_dim, activation="sigmoid", name="Embedder_S_Dense")(X_S_input)

    # Temporal Embedder (e_X) conditioned on static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S)
    C_X = Concatenate(axis=-1, name="Embedder_X_Concat")([X_T_input, H_S_repeated])

    x = C_X
    for _ in range(num_layers):
        x = rnn_layer(module_name, hidden_dim, return_sequences=True)(x)
    e_outputs = SequenceMaskLayer(maxlen=max_seq_len, name="Embedder_Mask")(
        [x, T_input]
    )

    H_T = Dense(hidden_dim, activation="sigmoid", name="Embedder_T_Dense")(e_outputs)
    return Model(inputs=[X_S_input, X_T_input, T_input], outputs=[H_S, H_T], name="embedder")


def build_recovery(max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name):
    """Builds the Recovery Keras Model with static conditioning on temporal recovery."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    # Static Recovery (r_S): use softmax for multi-class one-hot static regimes
    static_act = "softmax" if dim_s > 1 else "sigmoid"
    X_S_tilde = Dense(dim_s, activation=static_act, name="Recovery_S_Dense")(H_S_input)

    # Temporal Recovery (r_X) conditioned on static latent code H_S_input
    H_S_repeated = RepeatVector(max_seq_len)(H_S_input)
    C_H = Concatenate(axis=-1, name="Recovery_X_Concat")([H_T_input, H_S_repeated])

    x = C_H
    for _ in range(num_layers):
        x = rnn_layer(module_name, hidden_dim, return_sequences=True)(x)
    r_outputs = SequenceMaskLayer(maxlen=max_seq_len, name="Recovery_Mask")(
        [x, T_input]
    )

    X_T_tilde = Dense(dim_x, activation="sigmoid", name="Recovery_X_Dense")(r_outputs)

    return Model(
        inputs=[H_S_input, H_T_input, T_input],
        outputs=[X_S_tilde, X_T_tilde],
        name="recovery",
    )


def build_generator(max_seq_len, z_dim_s, z_dim_x, hidden_dim, num_layers, module_name):
    """Builds the Generator Keras Model."""
    Z_S_input = Input(shape=(z_dim_s,), name="Z_S_input")
    Z_T_input = Input(shape=(max_seq_len, z_dim_x), name="Z_T_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    # Static Generator (g_S)
    E_S_hat = Dense(hidden_dim, activation="sigmoid", name="Generator_S_Dense")(
        Z_S_input
    )

    # Temporal Generator (g_X) conditioned on static latent code E_S_hat
    E_S_hat_repeated = RepeatVector(max_seq_len)(E_S_hat)
    C_Z = Concatenate(axis=-1, name="Generator_X_Concat")([Z_T_input, E_S_hat_repeated])

    x = C_Z
    for _ in range(num_layers):
        x = rnn_layer(module_name, hidden_dim, return_sequences=True)(x)
    e_outputs = SequenceMaskLayer(maxlen=max_seq_len, name="Generator_Mask")(
        [x, T_input]
    )

    E_T_hat = Dense(hidden_dim, activation="sigmoid", name="Generator_T_Dense")(
        e_outputs
    )

    return Model(
        inputs=[Z_S_input, Z_T_input, T_input],
        outputs=[E_S_hat, E_T_hat],
        name="generator",
    )


def build_supervisor(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Supervisor Keras Model."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input_Sup")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input_Sup")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input_Sup")

    # Condition the supervisor on the static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S_input)
    C_H = Concatenate(axis=-1, name="Supervisor_Concat")([H_T_input, H_S_repeated])

    num_supervisor_layers = max(1, num_layers - 1)
    x = C_H
    for _ in range(num_supervisor_layers):
        x = rnn_layer(module_name, hidden_dim, return_sequences=True)(x)
    e_outputs = SequenceMaskLayer(maxlen=max_seq_len, name="Supervisor_Mask")(
        [x, T_input]
    )

    S = Dense(hidden_dim, activation="sigmoid", name="Supervisor_Dense")(e_outputs)

    return Model(inputs=[H_S_input, H_T_input, T_input], outputs=S, name="supervisor")


def build_discriminator(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Discriminator Keras Model."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input_Disc")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input_Disc")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input_Disc")

    # Static Discriminator (d_S)
    Y_S_hat = Dense(1, activation=None, name="Discriminator_S_Dense")(H_S_input)

    # Temporal Discriminator (d_X) conditioned on static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S_input)
    C_H = Concatenate(axis=-1, name="Discriminator_X_Concat")([H_T_input, H_S_repeated])

    x = C_H
    for _ in range(num_layers):
        rnn_l = rnn_layer(module_name, hidden_dim, return_sequences=True)
        x = Bidirectional(rnn_l, merge_mode="concat")(x)
    d_outputs = SequenceMaskLayer(maxlen=max_seq_len, name="Discriminator_Mask")(
        [x, T_input]
    )

    Y_T_hat = Dense(1, activation=None, name="Discriminator_X_Dense")(d_outputs)

    return Model(
        inputs=[H_S_input, H_T_input, T_input],
        outputs=[Y_S_hat, Y_T_hat],
        name="discriminator",
    )


def timegan(ori_data_s, ori_data_x, parameters, save_dir="saved_models"):
    """TimeGAN function with Static integration (TensorFlow 2.x implementation).

    Args:
      - ori_data_s: original static time-series data
      - ori_data_x: original temporal time-series data
      - parameters: TimeGAN network parameters
      - save_dir: directory to save trained models and scalers

    Returns:
      - generated_data_s: generated static time-series data
      - generated_data_x: generated temporal time-series data
    """
    raw_s = np.asarray(ori_data_s, dtype=np.float32)
    raw_x = np.asarray(ori_data_x, dtype=np.float32)

    # Data parameters
    no, seq_len, dim_x = raw_x.shape
    dim_s = raw_s.shape[1]
    ori_time, max_seq_len = extract_time(raw_x)
    ori_time_arr = np.asarray(ori_time, dtype=np.int32)

    # Normalize data
    norm_s, min_val_s, max_val_s = MinMaxScaler(raw_s)
    norm_x, min_val_x, max_val_x = MinMaxScaler(raw_x)

    # Network parameters
    hidden_dim = int(parameters["hidden_dim"])
    num_layers = int(parameters["num_layer"])
    iterations = int(parameters["iterations"])
    batch_size = int(parameters["batch_size"])
    module_name = parameters["module"]
    learning_rate = float(parameters.get("learning_rate", 1e-3))
    gamma = float(parameters.get("gamma", 1.0))
    gamma_s = float(parameters.get("gamma_s", 100.0))
    gamma_v = float(parameters.get("gamma_v", 100.0))
    seed = parameters.get("seed", None)
    rng = np.random.default_rng(seed)

    z_dim_s = dim_s
    z_dim_x = dim_x

    # Build Models
    embedder = build_embedder(
        max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name
    )
    recovery = build_recovery(
        max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name
    )
    generator = build_generator(
        max_seq_len, z_dim_s, z_dim_x, hidden_dim, num_layers, module_name
    )
    supervisor = build_supervisor(max_seq_len, hidden_dim, num_layers, module_name)
    discriminator = build_discriminator(
        max_seq_len, hidden_dim, num_layers, module_name
    )

    # Loss & Optimizers
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    cce = tf.keras.losses.CategoricalCrossentropy()

    E_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    GS_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _static_recon_loss(s_true, s_pred):
        if dim_s > 1:
            return cce(s_true, s_pred) + mse(s_true, s_pred)
        return mse(s_true, s_pred)

    @tf.function
    def train_step_embedder(S_mb, X_mb, T_mb):
        with tf.GradientTape() as tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [X_S_tilde, X_T_tilde] = recovery([H_S, H_T, T_mb], training=True)

            E_loss_S = _static_recon_loss(S_mb, X_S_tilde)
            E_loss_T = mse(X_mb, X_T_tilde)
            E_loss_T0 = E_loss_S + E_loss_T
            E_loss0 = 10.0 * tf.sqrt(E_loss_T0 + 1e-8)

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(E_loss0, vars_e)
        E_optimizer.apply_gradients(zip(gradients, vars_e))
        return E_loss_T0

    @tf.function
    def train_step_supervisor(S_mb, X_mb, T_mb):
        with tf.GradientTape() as tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=False)
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)
            # Supervised loss is only on temporal dynamics
            G_loss_S = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])

        vars_s = supervisor.trainable_variables
        gradients = tape.gradient(G_loss_S, vars_s)
        GS_optimizer.apply_gradients(zip(gradients, vars_s))
        return G_loss_S

    @tf.function
    def train_step_generator(S_mb, X_mb, T_mb, Z_S_mb, Z_T_mb):
        with tf.GradientTape() as g_tape:
            # Embed real data
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=False)
            # Generate fake latents
            [E_S_hat, E_T_hat] = generator([Z_S_mb, Z_T_mb, T_mb], training=True)
            # Supervise fake latents
            H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb], training=True)
            # Reconstruct fake data
            [X_S_hat, X_T_hat] = recovery([E_S_hat, H_T_hat, T_mb], training=False)

            # Discriminate
            [Y_S_fake, Y_T_fake] = discriminator(
                [E_S_hat, H_T_hat, T_mb], training=False
            )
            [Y_S_fake_e, Y_T_fake_e] = discriminator(
                [E_S_hat, E_T_hat, T_mb], training=False
            )

            # 1. Adversarial Loss
            G_loss_U_S = bce(tf.ones_like(Y_S_fake), Y_S_fake)
            G_loss_U_T = bce(tf.ones_like(Y_T_fake), Y_T_fake)
            G_loss_U = G_loss_U_S + G_loss_U_T

            G_loss_U_e_S = bce(tf.ones_like(Y_S_fake_e), Y_S_fake_e)
            G_loss_U_e_T = bce(tf.ones_like(Y_T_fake_e), Y_T_fake_e)
            G_loss_U_e = G_loss_U_e_S + G_loss_U_e_T

            # 2. Supervised Loss
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)
            G_loss_S = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])

            # 3. Moment Loss (computed across batch axis [0] per time step)
            s_hat_mean, s_hat_var = tf.nn.moments(X_S_hat, axes=[0])
            s_mb_mean, s_mb_var = tf.nn.moments(S_mb, axes=[0])
            G_loss_V_S1 = tf.reduce_mean(
                tf.abs(tf.sqrt(s_hat_var + 1e-6) - tf.sqrt(s_mb_var + 1e-6))
            )
            G_loss_V_S2 = tf.reduce_mean(tf.abs(s_hat_mean - s_mb_mean))
            G_loss_V_S = G_loss_V_S1 + G_loss_V_S2

            x_hat_mean, x_hat_var = tf.nn.moments(X_T_hat, axes=[0])
            x_mb_mean, x_mb_var = tf.nn.moments(X_mb, axes=[0])
            G_loss_V_T1 = tf.reduce_mean(
                tf.abs(tf.sqrt(x_hat_var + 1e-6) - tf.sqrt(x_mb_var + 1e-6))
            )
            G_loss_V_T2 = tf.reduce_mean(tf.abs(x_hat_mean - x_mb_mean))
            G_loss_V_T = G_loss_V_T1 + G_loss_V_T2
            G_loss_V = G_loss_V_S + G_loss_V_T

            # Total G_loss with balanced TimeGAN weights (default 100 instead of 1000)
            G_loss = (
                G_loss_U
                + gamma * G_loss_U_e
                + gamma_s * tf.sqrt(G_loss_S + 1e-8)
                + gamma_v * G_loss_V
            )

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        gradients_g = g_tape.gradient(G_loss, vars_g)
        G_optimizer.apply_gradients(zip(gradients_g, vars_g))
        return G_loss_U, G_loss_S, G_loss_V

    @tf.function
    def train_step_embedder_joint(S_mb, X_mb, T_mb):
        with tf.GradientTape() as e_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [S_tilde, X_tilde] = recovery([H_S, H_T, T_mb], training=True)
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=False)

            G_loss_S_e = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])
            E_loss_S = _static_recon_loss(S_mb, S_tilde)
            E_loss_T = mse(X_mb, X_tilde)
            E_loss_T0 = E_loss_S + E_loss_T
            E_loss0 = 10.0 * tf.sqrt(E_loss_T0 + 1e-8)
            E_loss = E_loss0 + 0.1 * G_loss_S_e

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients_e = e_tape.gradient(E_loss, vars_e)
        E_optimizer.apply_gradients(zip(gradients_e, vars_e))
        return E_loss_T0

    @tf.function
    def train_step_discriminator(S_mb, X_mb, T_mb, Z_S_mb, Z_T_mb):
        with tf.GradientTape() as d_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=False)
            [E_S_hat, E_T_hat] = generator([Z_S_mb, Z_T_mb, T_mb], training=False)
            H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb], training=False)

            [Y_S_fake, Y_T_fake] = discriminator(
                [E_S_hat, H_T_hat, T_mb], training=True
            )
            [Y_S_real, Y_T_real] = discriminator([H_S, H_T, T_mb], training=True)
            [Y_S_fake_e, Y_T_fake_e] = discriminator(
                [E_S_hat, E_T_hat, T_mb], training=True
            )

            D_loss_real = bce(tf.ones_like(Y_S_real), Y_S_real) + bce(
                tf.ones_like(Y_T_real), Y_T_real
            )
            D_loss_fake = bce(tf.zeros_like(Y_S_fake), Y_S_fake) + bce(
                tf.zeros_like(Y_T_fake), Y_T_fake
            )
            D_loss_fake_e = bce(tf.zeros_like(Y_S_fake_e), Y_S_fake_e) + bce(
                tf.zeros_like(Y_T_fake_e), Y_T_fake_e
            )
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if D_loss > 0.15:
            vars_d = discriminator.trainable_variables
            gradients_d = d_tape.gradient(D_loss, vars_d)
            D_optimizer.apply_gradients(zip(gradients_d, vars_d))

        return D_loss

    def _sample_batch():
        actual_bs = min(batch_size, no)
        idx = rng.integers(0, no, size=actual_bs)
        return norm_s[idx], norm_x[idx], ori_time_arr[idx]

    # 1. Embedding network training
    print("Start Embedding Network Training")
    for itt in range(iterations):
        S_mb, X_mb, T_mb = _sample_batch()
        step_e_loss = train_step_embedder(S_mb, X_mb, T_mb)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(step_e_loss.numpy()), 4)}"
            )

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        S_mb, X_mb, T_mb = _sample_batch()
        step_g_loss_s = train_step_supervisor(S_mb, X_mb, T_mb)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(step_g_loss_s.numpy()), 4)}"
            )

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")
    step_d_loss = tf.constant(0.0, dtype=tf.float32)
    step_g_loss_u = tf.constant(0.0, dtype=tf.float32)
    step_g_loss_s = tf.constant(0.0, dtype=tf.float32)
    step_g_loss_v = tf.constant(0.0, dtype=tf.float32)
    step_e_loss_t0 = tf.constant(0.0, dtype=tf.float32)

    for itt in range(iterations):
        # Generator update (2x on freshly sampled batches) and Embedder update (1x)
        for _ in range(2):
            S_mb, X_mb, T_mb = _sample_batch()
            Z_S_mb, Z_T_mb = random_generator(
                len(T_mb), z_dim_s, z_dim_x, T_mb, max_seq_len, rng=rng
            )
            step_g_loss_u, step_g_loss_s, step_g_loss_v = train_step_generator(
                S_mb, X_mb, T_mb, Z_S_mb, Z_T_mb
            )

        step_e_loss_t0 = train_step_embedder_joint(S_mb, X_mb, T_mb)

        # Discriminator update (1x on freshly sampled batch)
        S_mb_d, X_mb_d, T_mb_d = _sample_batch()
        Z_S_mb_d, Z_T_mb_d = random_generator(
            len(T_mb_d), z_dim_s, z_dim_x, T_mb_d, max_seq_len, rng=rng
        )
        step_d_loss = train_step_discriminator(
            S_mb_d, X_mb_d, T_mb_d, Z_S_mb_d, Z_T_mb_d
        )

        if itt % 1000 == 0 or itt == iterations - 1:
            print(
                f"step: {itt}/{iterations}, "
                f"d_loss: {np.round(step_d_loss.numpy(), 4)}, "
                f"g_loss_u: {np.round(step_g_loss_u.numpy(), 4)}, "
                f"g_loss_s: {np.round(np.sqrt(step_g_loss_s.numpy()), 4)}, "
                f"g_loss_v: {np.round(step_g_loss_v.numpy(), 4)}, "
                f"e_loss_t0: {np.round(np.sqrt(step_e_loss_t0.numpy()), 4)}"
            )

    print("Finish Joint Training")

    # --- SAVE TRAINED MODELS AND SCALERS ---
    print("Saving trained models and scalers...")
    os.makedirs(save_dir, exist_ok=True)

    embedder.save(os.path.join(save_dir, "embedder.h5"))
    generator.save(os.path.join(save_dir, "generator.h5"))
    supervisor.save(os.path.join(save_dir, "supervisor.h5"))
    recovery.save(os.path.join(save_dir, "recovery.h5"))
    discriminator.save(os.path.join(save_dir, "discriminator.h5"))

    scaler_path = os.path.join(save_dir, "scalers.npz")
    np.savez(
        scaler_path,
        min_val_s=min_val_s,
        max_val_s=max_val_s,
        min_val_x=min_val_x,
        max_val_x=max_val_x,
    )
    print(f"Models and scalers saved to {save_dir}/")

    return generate_data_from_saved_models(
        ori_data_s, ori_data_x, save_dir=save_dir, seed=seed
    )


def generate_data_from_saved_models(
    ori_data_s,
    ori_data_x,
    save_dir="saved_models",
    target_regime=None,
    n_samples=None,
    enforce_consistency=True,
    seed=None,
):
    """Generates synthetic data using pre-trained models.

    Args:
        - ori_data_s: original static data (for shape/param info)
        - ori_data_x: original temporal data (for shape/param info)
        - save_dir: directory where models and scalers are saved.
        - target_regime: optional regime conditioning ('Normal', 'Crisis', 'Volatile',
          or integer index 0, 1, 2). If None, samples unconditionally.
        - n_samples: optional number of sequences to generate (defaults to len(ori_data_x)).
        - enforce_consistency: whether to enforce financial OHLCV invariants on output.
        - seed: optional random seed.

    Returns:
        - generated_data_s: list of 1D float32 static vectors
        - generated_data_x: list of 2D float32 temporal arrays of shape (seq_len, dim_x)
    """
    print("Loading pre-trained models and scalers...")

    gen_path = os.path.join(save_dir, "generator.h5")
    sup_path = os.path.join(save_dir, "supervisor.h5")
    rec_path = os.path.join(save_dir, "recovery.h5")
    scaler_path = os.path.join(save_dir, "scalers.npz")

    try:
        scalers = np.load(scaler_path)
        min_val_s = scalers["min_val_s"].astype(np.float32)
        max_val_s = scalers["max_val_s"].astype(np.float32)
        min_val_x = scalers["min_val_x"].astype(np.float32)
        max_val_x = scalers["max_val_x"].astype(np.float32)
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}")
        return None, None

    custom_objects = {
        "LayerNormLSTMCell": LayerNormLSTMCell,
        "SequenceMaskLayer": SequenceMaskLayer,
        "_sequence_mask_lambda": _sequence_mask_lambda,
    }

    try:
        generator = tf.keras.models.load_model(
            gen_path, custom_objects=custom_objects, safe_mode=False
        )
        supervisor = tf.keras.models.load_model(
            sup_path, custom_objects=custom_objects, safe_mode=False
        )
        recovery = tf.keras.models.load_model(
            rec_path, custom_objects=custom_objects, safe_mode=False
        )
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading models from {save_dir}. Did you train them first?")
        print(e)
        return None, None

    print("Models loaded successfully.")

    ori_data_s_arr = np.asarray(ori_data_s, dtype=np.float32)
    ori_data_x_arr = np.asarray(ori_data_x, dtype=np.float32)

    orig_no, seq_len, dim_x = ori_data_x_arr.shape
    dim_s = ori_data_s_arr.shape[1]
    ori_time, max_seq_len = extract_time(ori_data_x_arr)

    no = int(n_samples) if n_samples is not None else orig_no
    if no != orig_no:
        rng_t = np.random.default_rng(seed)
        sampled_t_idx = rng_t.integers(0, orig_no, size=no)
        gen_time = [ori_time[i] for i in sampled_t_idx]
    else:
        gen_time = ori_time

    z_dim_s = dim_s
    z_dim_x = dim_x
    rng = np.random.default_rng(seed)

    print("Generating synthetic data...")
    Z_S_mb, Z_T_mb = random_generator(
        no, z_dim_s, z_dim_x, gen_time, max_seq_len, rng=rng
    )

    # Optional regime-targeted sampling for Z_S
    regime_map = {"Normal": 0, "Crisis": 1, "Volatile": 2, 0: 0, 1: 1, 2: 2}
    if target_regime is not None:
        if target_regime not in regime_map:
            raise ValueError(
                f"Unknown target_regime={target_regime}. Choose from {list(regime_map.keys())}."
            )
        target_idx = regime_map[target_regime]
        gen_s_layer = generator.get_layer("Generator_S_Dense")
        rec_s_layer = recovery.get_layer("Recovery_S_Dense")

        collected_z_s = []
        collected_count = 0
        for _ in range(25):
            cand_z_s = rng.uniform(0.0, 1.0, size=(max(no * 4, 256), z_dim_s)).astype(
                np.float32
            )
            cand_e_s = gen_s_layer(cand_z_s)
            cand_x_s = rec_s_layer(cand_e_s).numpy()
            cand_x_s_denorm = cand_x_s * (max_val_s + 1e-7) + min_val_s
            match_mask = np.argmax(cand_x_s_denorm, axis=1) == target_idx
            matched = cand_z_s[match_mask]
            if len(matched) > 0:
                collected_z_s.append(matched)
                collected_count += len(matched)
                if collected_count >= no:
                    break

        if collected_count >= no:
            Z_S_mb = np.concatenate(collected_z_s, axis=0)[:no]
        else:
            # Fallback: bias Z_S towards target regime coordinate
            Z_S_mb = rng.uniform(0.0, 0.2, size=(no, z_dim_s)).astype(np.float32)
            Z_S_mb[:, target_idx] = rng.uniform(0.8, 1.0, size=(no,)).astype(np.float32)

    Z_S_mb_t = tf.convert_to_tensor(Z_S_mb, dtype=tf.float32)
    Z_T_mb_t = tf.convert_to_tensor(Z_T_mb, dtype=tf.float32)
    T_mb_t = tf.convert_to_tensor(gen_time, dtype=tf.int32)

    [E_S_hat, E_T_hat] = generator([Z_S_mb_t, Z_T_mb_t, T_mb_t], training=False)
    H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb_t], training=False)
    [X_S_hat, X_T_hat_curr] = recovery([E_S_hat, H_T_hat, T_mb_t], training=False)

    generated_data_s = X_S_hat.numpy().astype(np.float32)
    generated_data_x_curr = X_T_hat_curr.numpy().astype(np.float32)

    # Renormalize in float32 (avoiding 3D dtype=object array bug)
    generated_data_s = generated_data_s * (max_val_s + 1e-7) + min_val_s
    generated_data_x_curr = generated_data_x_curr * (max_val_x + 1e-7) + min_val_x

    generated_data_x = [
        generated_data_x_curr[i, : gen_time[i], :].astype(np.float32)
        for i in range(no)
    ]

    if enforce_consistency:
        generated_data_x = enforce_financial_consistency(
            generated_data_x, recompute_indicators=False
        )

    print("Data generation complete.")
    return list(generated_data_s), generated_data_x
