#pragma once

#include <torch/torch.h>
#include <schnet/core/graph.h>
#include <schnet/module/ModuleUtils.h>

using Graph = schnet::core::Graph;

namespace schnet::module {
    class InteractionLayerImpl : public torch::nn::Module {
        public: 
            InteractionLayerImpl(
                int _hidden_dim, 
                int _num_gaussians, 
                int _num_filters, 
                float _cutoff
            );
            torch::Tensor forward(
                const torch::Tensor& x, 
                const torch::Tensor& distance, 
                Graph& graph, 
                const torch::Tensor& edge_attr
            );
        private:
            float cutoff;
            int num_filters;
            torch::nn::Linear lin1{nullptr}, lin2{nullptr};
            torch::nn::Sequential mlp{nullptr};
            schnet::module::ShiftedSoftplus act{nullptr};
    };
    TORCH_MODULE(InteractionLayer);
}