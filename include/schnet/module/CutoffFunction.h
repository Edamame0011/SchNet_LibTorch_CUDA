#pragma once

#include <torch/extension.h>

namespace schnet::module {
    class CutoffFunction : public torch::autograd::Function<CutoffFunction> {
        public:
            static torch::Tensor forward(
                torch::autograd::AutogradContext *ctx,
                torch::Tensor dist,
                float cutoff
            );
            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext *ctx,
                torch::autograd::variable_list grad_outputs
            );
    };
}