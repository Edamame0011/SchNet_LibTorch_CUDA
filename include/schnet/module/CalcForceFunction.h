#pragma once

#include <torch/extension.h>

namespace schnet::module {
    class CalcForceFunction : public torch::autograd::Function<CalcForceFunction> {
        public:
            static torch::Tensor forward(
                torch::autograd::AutogradContext *ctx,
                const torch::Tensor& diff_E, 
                const torch::Tensor& dst_node_ptr,
                int num_nodes
            );
            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext *ctx,
                torch::autograd::variable_list grad_outputs
            );
    };
}