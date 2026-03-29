#pragma once

#include <torch/extension.h>

namespace schnet::module {
    class AggMessagesFunction : public torch::autograd::Function<AggMessagesFunction> {
        public:
            static torch::Tensor forward(
                torch::autograd::AutogradContext *ctx, 
                const torch::Tensor& W, 
                const torch::Tensor& V, 
                const torch::Tensor& dst_node_ptr, 
                const torch::Tensor& src_list, 
                int num_nodes
            );
            static torch::autograd::tensor_list backward(
                torch::autograd::AutogradContext *ctx,
                torch::autograd::tensor_list grad_outputs
            );
    };
}