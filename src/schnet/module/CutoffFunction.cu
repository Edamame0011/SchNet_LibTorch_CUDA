#include <schnet/module/CutoffFunction.h>

namespace {
    __global__ void cutoff_kernel(
        const float* __restrict__ distance, 
        float* __restrict__ output, 
        float cutoff, 
        int num_edges
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_edges) {
            float d = distance[idx];
            if (d > cutoff) output[idx] = 0.0f;
            else output[idx] = 0.5f * cosf(d * 3.14159265f / cutoff) + 1.0f;
        }
    }

    __global__ void cutoff_backward_kernel(
        const float* __restrict__ grad_C, 
        const float* __restrict__ distance, 
        float* __restrict__ grad_dist, 
        float cutoff, 
        int num_edges
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_edges) {
            float d = distance[idx];
            if (d > cutoff) grad_dist[idx] = 0.0f;
            else {
                float grad_val = -(3.14159265f / (2.0f * cutoff)) * sinf(d * 3.14159265f / cutoff);
                grad_dist[idx] = grad_C[idx] * grad_val;
            }
        }
    }
}

using namespace schnet::module;

torch::Tensor CutoffFunction::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor dist,
    float cutoff
) {
    ctx->save_for_backward({dist});
    ctx->saved_data["cutoff"] = cutoff;

    auto C = torch::zeros_like(dist);
    int num_edges = dist.size(0);
    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;

    cutoff_kernel<<<blocks, threads>>>(
        dist.data_ptr<float>(), 
        C.data_ptr<float>(), 
        cutoff, 
        num_edges
    );

    return C;
}

torch::autograd::variable_list CutoffFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto grad_C = grad_outputs[0].contiguous();
    
    auto saved = ctx->get_saved_variables();
    auto dist = saved[0];
    float cutoff = ctx->saved_data["cutoff"].toDouble();

    auto grad_dist = torch::zeros_like(dist);
    int num_edges = dist.size(0);
    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;

    cutoff_backward_kernel<<<blocks, threads>>>(
        grad_C.data_ptr<float>(), 
        dist.data_ptr<float>(), 
        grad_dist.data_ptr<float>(), 
        cutoff, 
        num_edges
    );

    return {grad_dist, torch::Tensor()};
}