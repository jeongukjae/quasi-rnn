import argparse
import time

import tensorflow as tf

from qrnn import QRNN

parser = argparse.ArgumentParser()
parser.add_argument("--hidden-size", type=int, default=320)
parser.add_argument("--sequence-length", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--loops", type=int, default=100)

args = parser.parse_args()
hidden_size = args.hidden_size
sequence_length = args.sequence_length
batch_size = args.batch_size
loops = args.loops

print("Benchmark inference speed")
print(f" Hidden Size: {hidden_size}")
print(f" Sequence Length: {sequence_length}")
print(f" Batch Size: {batch_size}")

qrnn = QRNN(hidden_size, kernel_size=2)
lstm = tf.keras.layers.LSTM(hidden_size)


@tf.function
def qrnn_call(input_tensor):
    return qrnn(input_tensor)


@tf.function
def lstm_call(input_tensor):
    return lstm(input_tensor)


# draw graph
input = tf.keras.Input((sequence_length, hidden_size))
model = tf.keras.Model(input, qrnn(input))
tf.keras.callbacks.TensorBoard("./logs/qrnn").set_model(model)

input = tf.keras.Input((sequence_length, hidden_size))
model = tf.keras.Model(input, lstm(input))
tf.keras.callbacks.TensorBoard("./logs/lstm").set_model(model)

# Warmup
print("warmup")
random_input = tf.random.uniform((batch_size, sequence_length, hidden_size))
for _ in range(100):
    qrnn_call(random_input)
    lstm_call(random_input)
print("warmup end")

#
# Check QRNN
start_time = time.time()
for _ in range(loops):
    qrnn_call(random_input)
qrnn_elapsed = time.time() - start_time

print(f"QRNN Speed: {qrnn_elapsed:.4f}s")

#
# Check LSTM
start_time = time.time()
for _ in range(loops):
    lstm_call(random_input)
lstm_elapsed = time.time() - start_time

print(f"LSTM Speed: {lstm_elapsed:.4f}s")

#
# Check Diff
print(f"Improvement: {lstm_elapsed / qrnn_elapsed:.2f}x")
