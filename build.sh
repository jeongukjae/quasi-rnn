#!/bin/bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

echo "TF_CFLAGS: ${TF_CFLAGS}"
echo "TF_LFLAGS: ${TF_LFLAGS}"

g++ -std=c++11 -shared ./qrnn/qrnn_recurrent.cc -o ./qrnn/qrnn_recurrent.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
