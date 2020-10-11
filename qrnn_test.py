import pytest
import tensorflow as tf

from qrnn import QRNN, qrnn_pooling


@pytest.mark.parametrize("hidden_size, kernel_size", [pytest.param(30, 3), pytest.param(30, 5), pytest.param(30, 12)])
@pytest.mark.parametrize(
    "input_shape, pooling_method",
    [
        pytest.param([10, 20, 30], "f"),
        pytest.param([10, 20, 30], "fo"),
        pytest.param([10, 20, 30], "ifo"),
        pytest.param([10, 20, 10], "f"),
        pytest.param([10, 20, 10], "fo"),
    ],
)
def test_qrnn_shape(pooling_method, hidden_size, kernel_size, input_shape):
    qrnn = QRNN(hidden_size=hidden_size, kernel_size=kernel_size, pooling_method=pooling_method)

    output = qrnn(tf.random.uniform(input_shape, dtype=tf.float32))
    assert output.shape == input_shape[:-1] + [hidden_size]


@pytest.mark.parametrize(
    "batch_size, sequence_length, hidden_size",
    [pytest.param(10, 20, 30), pytest.param(50, 20, 40)],
)
def test_f_pooling(batch_size, sequence_length, hidden_size):
    z = tf.random.uniform((batch_size, sequence_length, hidden_size))
    f = tf.random.uniform((batch_size, sequence_length, hidden_size))

    result = qrnn_pooling("f", z, f)

    ta = tf.TensorArray(tf.float32, size=sequence_length)
    h = tf.zeros((batch_size, hidden_size))
    for i in range(sequence_length):
        h = f[:, i] * h + (1.0 - f[:, i]) * z[:, i]
        ta = ta.write(i, h)
    expected_result = tf.transpose(ta.stack(), perm=[1, 0, 2])

    tf.debugging.assert_near(result, expected_result)


@pytest.mark.parametrize(
    "batch_size, sequence_length, hidden_size",
    [pytest.param(10, 20, 30), pytest.param(50, 20, 40)],
)
def test_fo_pooling(batch_size, sequence_length, hidden_size):
    z = tf.random.uniform((batch_size, sequence_length, hidden_size))
    f = tf.random.uniform((batch_size, sequence_length, hidden_size))
    o = tf.random.uniform((batch_size, sequence_length, hidden_size))

    result = qrnn_pooling("fo", z, f, o=o)

    ta = tf.TensorArray(tf.float32, size=sequence_length)
    c = tf.zeros((batch_size, hidden_size))
    for i in range(sequence_length):
        c = f[:, i] * c + (1.0 - f[:, i]) * z[:, i]
        h = o[:, i] * c
        ta = ta.write(i, h)
    expected_result = tf.transpose(ta.stack(), perm=[1, 0, 2])

    tf.debugging.assert_near(result, expected_result)


@pytest.mark.parametrize(
    "batch_size, sequence_length, hidden_size",
    [pytest.param(10, 20, 30), pytest.param(50, 20, 40)],
)
def test_ifo_pooling(batch_size, sequence_length, hidden_size):
    z = tf.random.uniform((batch_size, sequence_length, hidden_size))
    f = tf.random.uniform((batch_size, sequence_length, hidden_size))
    o = tf.random.uniform((batch_size, sequence_length, hidden_size))
    i = tf.random.uniform((batch_size, sequence_length, hidden_size))

    result = qrnn_pooling("ifo", z, f, o=o, i=i)

    ta = tf.TensorArray(tf.float32, size=sequence_length)
    c = tf.zeros((batch_size, hidden_size))
    for index in range(sequence_length):
        c = f[:, index] * c + i[:, index] * z[:, index]
        h = o[:, index] * c
        ta = ta.write(index, h)
    expected_result = tf.transpose(ta.stack(), perm=[1, 0, 2])

    tf.debugging.assert_near(result, expected_result)
