#include <schnet/module/AggMessagesFunction.h>

namespace {
    __global__ void agg_messages_kernel(
        const float* __restrict__ W, 
        const float* __restrict__ V, 
        const int32_t* __restrict__ dst_node_ptr, 
        const int32_t* __restrict__ src_list, 
        float* agg_messages, 
        int num_nodes, 
        int num_filters
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_nodes * num_filters;

        if (idx < total_elements) {
            int dst = idx / num_filters;
            int d = idx % num_filters;

            int start_edge = dst_node_ptr[dst];
            int end_edge = dst_node_ptr[dst + 1];

            float sum = 0.0f;
            for (int e = start_edge; e < end_edge; e ++) {
                int src = src_list[e];
                sum += W[e * num_filters + d] * V[src * num_filters + d];
            }
            agg_messages[idx] = sum;
        }
    }

__global__ void agg_messages_backward_V_kernel(
        const float* __restrict__ grad_agg, 
        const float* __restrict__ W,
        const int32_t* __restrict__ dst_node_ptr, 
        const int32_t* __restrict__ src_list, 
        float* __restrict__ grad_V,
        int num_nodes,
        int num_filters
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_nodes * num_filters;
        if (idx < total_elements) {
            int dst = idx / num_filters;
            int d = idx % num_filters;

            int start_edge = dst_node_ptr[dst];
            int end_edge = dst_node_ptr[dst + 1];
            
            for (int e = start_edge; e < end_edge; e++) {
                int src = src_list[e];
                float grad_val = grad_agg[dst * num_filters + d] * W[e * num_filters + d];

                atomicAdd(&grad_V[src * num_filters + d], grad_val);
            }
        }
    }

    __global__ void agg_messages_backward_W_kernel(
        const float* __restrict__ grad_agg,
        const float* __restrict__ V,
        const int32_t* __restrict__ dst_node_ptr, 
        const int32_t* __restrict__ src_list, 
        float* __restrict__ grad_W, 
        int num_edges, 
        int num_filters
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_edges * num_filters;
        if (idx < total_elements) {
            int dst = idx / num_filters;
            int d = idx % num_filters;
            int start_edge = dst_node_ptr[dst];
            int end_edge = dst_node_ptr[dst + 1];
            for (int e = start_edge; e < end_edge; e++) {
                int src = src_list[e];
                grad_W[e * num_filters + d] = grad_agg[dst * num_filters + d] * V[src * num_filters + d];
            }
        }
    }
}

using namespace schnet::module;

torch::Tensor AggMessagesFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor& W,
    const torch::Tensor& V,
    const torch::Tensor& dst_node_ptr, 
    const torch::Tensor& src_list, 
    int num_nodes
) {
    ctx->save_for_backward({W, V, dst_node_ptr, src_list});
    ctx->saved_data["num_nodes"] = num_nodes;

    int num_filters = W.size(1);
    auto agg_messages = torch::zeros({num_nodes, num_filters}, W.options());

    int total_elements = num_nodes * num_filters;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    agg_messages_kernel<<<blocks, threads>>>(
        W.data_ptr<float>(), 
        V.data_ptr<float>(), 
        dst_node_ptr.data_ptr<int32_t>(), 
        src_list.data_ptr<int32_t>(), 
        agg_messages.data_ptr<float>(), 
        num_nodes, 
        num_filters
    );

    return agg_messages;
}

torch::autograd::tensor_list AggMessagesFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto grad_agg = grad_outputs[0].contiguous(); 
        
    auto saved = ctx->get_saved_variables();
    auto W = saved[0];
    auto V = saved[1];
    auto dst_node_ptr = saved[2];
    auto src_list = saved[3];
    int num_nodes = ctx->saved_data["num_nodes"].toInt();
    
    int num_edges = W.size(0);
    int num_filters = W.size(1);
    int threads = 256;

    auto grad_W = torch::zeros_like(W);
    auto grad_V = torch::zeros_like(V);

    // grad_V の計算
    int total_V_elements = num_nodes * num_filters;
    int blocks_V = (total_V_elements + threads - 1) / threads;
    agg_messages_backward_V_kernel<<<blocks_V, threads>>>(
        grad_agg.data_ptr<float>(),
        W.data_ptr<float>(),
        dst_node_ptr.data_ptr<int32_t>(),
        src_list.data_ptr<int32_t>(),
        grad_V.data_ptr<float>(),
        num_nodes,
        num_filters
    );

    // grad_W の計算
    int total_W_elements = num_edges * num_filters;
    int blocks_W = (total_W_elements + threads - 1) / threads;
    agg_messages_backward_W_kernel<<<blocks_W, threads>>>(
        grad_agg.data_ptr<float>(),
        V.data_ptr<float>(),
        dst_node_ptr.data_ptr<int32_t>(),
        src_list.data_ptr<int32_t>(),
        grad_W.data_ptr<float>(),
        num_edges,
        num_filters
    );

    return {
        grad_W, grad_V, 
        torch::Tensor(), torch::Tensor(), torch::Tensor()
    };
}