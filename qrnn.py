import tensorflow as tf


class QRNN(tf.keras.layers.Layer):
    """
    Quasi-Recurrent Neural Network

    See [the paper (Bradbury et al., 2016)](https://arxiv.org/abs/1611.01576) for the details

    Arguments:
        hidden_size: Positive integer, dimensionality of the output space.
        kernel_size: Same as kernel_size argument of `tf.keras.layers.Conv1D`.
        zoneout_prob: Float between 0.0 and 1.0. Rate to drop f gate.
        pooling_method: Method to pool gates. Choices: "f", "fo", "ifo".
        activation: Activation function to pass z (candidate vector), such as tf.nn.tanh.
        gate_activation: Activation function to pass f, o (gates), such as tf.nn.tanh.
        kernel_regularizers: Regularizers to apply CNN's kernel.

    Call arguments:
        inputs: A 3D tensor.

    Input shape:
        A 3D tensor with shape:
        `(batch_size, time_steps, input_hidden_size)`

    Ouput shape:
        A 3D tensor with shape:
        `(batch_size, time_steps, hidden_size)`
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        zoneout_prob: float = 0.0,
        pooling_method: str = "fo",
        activation=tf.nn.tanh,
        gate_activation=tf.nn.sigmoid,
        kernel_regularizers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.activation = activation
        self.gate_activation = gate_activation
        self.pooling_method = pooling_method

        self.z_conv = tf.keras.layers.Conv1D(
            filters=hidden_size,
            kernel_size=kernel_size,
            strides=1,
            padding="valid",
            kernel_regularizer=kernel_regularizers,
            activation=activation,
            name="z_conv",
        )
        self.f_conv = tf.keras.layers.Conv1D(
            filters=hidden_size,
            kernel_size=kernel_size,
            strides=1,
            padding="valid",
            kernel_regularizer=kernel_regularizers,
            activation=gate_activation,
            name="f_conv",
        )
        if pooling_method != "f":
            self.o_conv = tf.keras.layers.Conv1D(
                filters=hidden_size,
                kernel_size=kernel_size,
                strides=1,
                padding="valid",
                kernel_regularizer=kernel_regularizers,
                activation=gate_activation,
                name="o_conv",
            )

        self.dropout = tf.keras.layers.Dropout(rate=zoneout_prob)

    def call(self, input_tensor, training=None):
        rank = len(input_tensor.shape)
        assert rank > 2, "rank > 2"

        # padding left to make convolution casual
        x = tf.pad(input_tensor, (rank - 2) * [[0, 0]] + [[self.kernel_size - 1, 0], [0, 0]])
        z = self.z_conv(x)
        f = self.f_conv(x)

        if self.pooling_method != "f":
            o = self.o_conv(x)
        else:
            o = None

        if training:
            # zoneout
            # f = 1 - dropout(1 - f)
            f = tf.subtract(1.0, self.dropout(tf.subtract(1.0, f)))

        pooling_result = qrnn_pooling(self.pooling_method, z, f, o, input_tensor)
        return pooling_result


def qrnn_pooling(pooling_method, z, f, o=None, i=None):
    with tf.name_scope("qrnn_pooling"):
        shapes = tf.shape(z)
        if "i" in pooling_method:
            z = tf.multiply(i, z)
        else:
            z = tf.multiply(tf.subtract(1.0, f), z)

        init_state = tf.zeros([shapes[0], shapes[2]], dtype=z.dtype)
        with tf.name_scope("recurrent"):
            _, outputs, _ = tf.keras.backend.rnn(
                step_function=_step,
                inputs=[z, f],
                initial_states=[init_state],
                constants=None,
                unroll=False,
                time_major=False,
            )

        if "o" in pooling_method:
            outputs = tf.multiply(outputs, o)

        return outputs


def _step(inputs, states):
    z, f = inputs
    h_next = tf.add(tf.multiply(f, states[0]), z)
    return h_next, [h_next]
