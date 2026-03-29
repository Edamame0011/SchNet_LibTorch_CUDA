#include <schnet/module/CalcForceFunction.h>

namespace {
    __global__ void calc_force_kernel(
        const float3* __restrict__ diff_E, 
        const int32_t* __restrict__ dst_node_ptr, 
        float3* __restrict__ force, 
        int num_nodes
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_nodes) {
            int start_edge = dst_node_ptr[idx];
            int end_edge = dst_node_ptr[idx + 1];

            float sum_x = 0.0f;
            float sum_y = 0.0f;
            float sum_z = 0.0f;
            for (int e = start_edge; e < end_edge; e ++) {
                float3 val = diff_E[e];
                sum_x += val.x;
                sum_y += val.y;
                sum_z += val.z;
            }
            force[idx] = make_float3(sum_x, sum_y, sum_z);
        }
    }

    __global__ void calc_force_grad_kernel(
        const float3* __restrict__ grad_force, 
        const int32_t* __restrict__ dst_node_ptr, 
        float3* __restrict__ grad_diff_E, 
        int num_nodes
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_nodes) {
            int start_edge = dst_node_ptr[idx];
            int end_edge = dst_node_ptr[idx + 1];

            float3 g = grad_force[idx];
            for (int e = start_edge; e < end_edge; e++) {
                grad_diff_E[e] = g;  
            }
        }
    }
}

using namespace schnet::module;

torch::Tensor CalcForceFunction::forward(
    torch::autograd::AutogradContext *ctx, 
    const torch::Tensor& diff_E, 
    const torch::Tensor& dst_node_ptr, 
    int num_nodes
) {
    ctx->save_for_backward({dst_node_ptr});
    ctx->saved_data["num_nodes"] = num_nodes;
    ctx->saved_data["num_edges"] = diff_E.size(0);

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    auto force = torch::zeros({num_nodes, 3}, diff_E.options());

    calc_force_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float3*>(diff_E.data_ptr<float>()), 
        dst_node_ptr.data_ptr<int32_t>(), 
        reinterpret_cast<float3*>(force.data_ptr<float>()), 
        num_nodes
    );

    return force;
}

torch::autograd::tensor_list CalcForceFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto grad_force = grad_outputs[0].contiguous(); 
        
    auto saved = ctx->get_saved_variables();
    auto dst_node_ptr = saved[0];

    int num_nodes = ctx->saved_data["num_nodes"].toInt();
    int num_edges = ctx->saved_data["num_edges"].toInt();
    
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    auto grad_diff_E = torch::empty({num_edges, 3}, grad_force.options());

    calc_force_grad_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float3*>(grad_force.data_ptr<float>()),
        dst_node_ptr.data_ptr<int32_t>(),
        reinterpret_cast<float3*>(grad_diff_E.data_ptr<float>()),
        num_nodes
    );

    return {
        grad_diff_E, 
        torch::Tensor(), torch::Tensor()
    };
}