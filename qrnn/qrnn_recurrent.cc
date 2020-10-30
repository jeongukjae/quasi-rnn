#include <algorithm>
#include <random>
#include <omp.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("QrnnRecurrent")
    .Input("z: float32")
    .Input("f: float32")
    .Output("outputs: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class QrnnRecurrentOp : public OpKernel
{
public:
  explicit QrnnRecurrentOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    const Tensor &z_tensor = context->input(0);
    auto z = z_tensor.flat<float>();

    const Tensor &f_tensor = context->input(1);
    auto f = f_tensor.flat<float>();

    OP_REQUIRES(context, z_tensor.IsSameSize(f_tensor), errors::InvalidArgument("Z should be a same size with F."));
    auto n_dims = z_tensor.dims();
    tensorflow::int64 sequence_length = z_tensor.dim_size(n_dims - 2);
    tensorflow::int64 hidden_size = z_tensor.dim_size(n_dims - 1);
    tensorflow::int64 batch_dim = 0;
    if (n_dims >= 3)
      for (int i = 0; i < n_dims - 2; i++)
        batch_dim += z_tensor.dim_size(i);

    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<float>();

    // first timestep output[:,0,:] = z[:,0,:]
#pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_dim; batch++)
    {
      for (int hdn = 0; hdn < hidden_size; hdn++)
      {
        auto index = hidden_size * sequence_length * batch + hdn;
        output(index) = z(index);
      }
    }

    for (int seq_len = 1; seq_len < sequence_length; seq_len++)
    {
#pragma omp parallel for collapse(2)
      for (int batch = 0; batch < batch_dim; batch++)
      {
        for (int hdn = 0; hdn < hidden_size; hdn++)
        {
          auto batch_index = hidden_size * sequence_length * batch;
          auto prev_index = batch_index + (seq_len - 1) * hidden_size + hdn;
          auto curr_index = prev_index + hidden_size;
          output(curr_index) = output(prev_index) * f(curr_index) + z(curr_index);
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("QrnnRecurrent").Device(DEVICE_CPU), QrnnRecurrentOp);
