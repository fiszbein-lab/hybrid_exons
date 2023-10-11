"""A module for compiling a deep, residual CNN."""
import numpy as np

from Bio.motifs import minimal
from tensorflow.keras import backend, layers, models, utils


def make_res_network(shape: tuple[int, int]):
    """Makes deep, residual network."""
    input_bases, _ = shape

    inputs = layers.Input(shape=shape)
    x = layers.Conv1D(32, 1)(inputs)

    # 100  = ([2*(6-1)*1] * 5)
    # 500  = ([2*(6-1)*1] * 10) + ([2*(6-1)*4] * 10)
    # 1500 = ([2*(6-1)*1] * 10) + ([2*(6-1)*4] * 10) + ([2*(6-1)*10] * 10)

    if input_bases > 99:
        for _ in range(10):
            x = _make_res_block(x, filters=32, kernel_size=6, dilation_rate=1)

    if input_bases > 499:
        for _ in range(10):
            x = _make_res_block(x, filters=32, kernel_size=6, dilation_rate=4)

    if input_bases > 1499:
        for _ in range(10):
            x = _make_res_block(x, filters=32, kernel_size=6, dilation_rate=10)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    res_net = models.Model(inputs, x)
    res_net.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['acc', 'AUC']
    )

    return res_net


def _make_res_block(x, filters, kernel_size, dilation_rate):
    """Makes residual block."""
    shortcut = x

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate, padding='same')(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate, padding='same')(x)

    x = layers.add([x, shortcut])

    return x


def one_hot_encode(seq) -> np.ndarray:
    """Return a one-hot encoded representation of a genomic sequence."""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    int_seq = [base_map[base] for base in seq]

    return utils.to_categorical(int_seq, num_classes=5)[:, :-1]


def viz_conv_net(model_path, x_data_path, save_path):
    """Visualizes the 1st layer of a convolutional neural network.
    """
    model = models.load_model(model_path)
    x_data = np.load(x_data_path)

    conv_layer1 = model.get_layer(model.layers[4].name)
    num_motifs = conv_layer1.filters

    window = conv_layer1.kernel_size[0]
    func = backend.function(
        [model.inputs[0]],
        [backend.max(conv_layer1.output, axis=1),
         backend.argmax(conv_layer1.output, axis=1)]
    )

    max_acts, max_inds = func([x_data])

    motifs = np.zeros((num_motifs, window, 4))
    nsites = np.zeros(num_motifs)

    for m in range(num_motifs):
        for n in range(len(x_data)):
            if max_acts[n, m] > 0.0:
                if len(x_data[n, max_inds[n, m]:max_inds[n, m] + window]) == 6:
                    nsites[m] += 1
                    motifs[m] += x_data[
                                 n, max_inds[n, m]:max_inds[n, m] + window]

    with open(save_path, 'w') as file:
        file.write(
            "MEME version 5.3.3\n\n"
            "ALPHABET= ACGT\n\n"
            "Background letter frequencies (from uniform background):\n"
            "A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for m in range(num_motifs):
            if nsites[m] == 0:
                continue

            file.write(
                f"MOTIF {m}\n"
                f"letter-probability matrix: "
                f"alength= 4 w= {window} "
                f"nsites= {int(nsites[m])} "
                f"E= 0.00\n"
            )

            for n in range(window):
                adenine, cytosine, guanine, thymine = tuple(
                    motifs[m, n, 0:4] / np.sum(motifs[m, n, 0:4]))

                file.write(
                    f"{adenine:.6f} "
                    f"{cytosine:.6f} "
                    f"{guanine:.6f} "
                    f"{thymine:.6f}\n"
                )

            file.write("\n")
