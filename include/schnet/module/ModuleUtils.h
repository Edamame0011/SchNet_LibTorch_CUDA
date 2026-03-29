#pragma once

#include <torch/extension.h>

namespace {
    torch::Tensor gaussian_rbf(const torch::Tensor& inputs, const torch::Tensor& offsets, const torch::Tensor& widths) {
        auto coeff = -0.5 / torch::pow(widths, 2);
        auto diff = inputs.unsqueeze(-1) - offsets;
                            
        return torch::exp(coeff * torch::pow(diff, 2));
    }
}

namespace schnet::module {
    struct ShiftedSoftplusImpl : torch::nn::Module {
        torch::Tensor log2;

        ShiftedSoftplusImpl() {
            log2 = register_buffer("log2", torch::log(torch::tensor(2.0)));
        }
        torch::Tensor forward(const torch::Tensor& x) {
            return torch::nn::functional::softplus(x) - log2;
        }
    };
    TORCH_MODULE(ShiftedSoftplus);

    struct GaussianRBFImpl : public torch::nn::Module {
        torch::Tensor offsets, widths;

        GaussianRBFImpl(int64_t n_rbf, float cutoff, float start = 0.0) {
            offsets = register_buffer("offsets", torch::linspace(start, cutoff, n_rbf));
            widths = register_buffer("widths", torch::full({n_rbf}, cutoff / (float)n_rbf));
        }
        torch::Tensor forward(const torch::Tensor& distances) {
            return gaussian_rbf(distances, offsets, widths);
        }
    };
    TORCH_MODULE(GaussianRBF);
}