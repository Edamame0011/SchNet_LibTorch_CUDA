#pragma once

#include <torch/extension.h>
#include <schnet/core/graph.h>
#include <schnet/module/InteractionLayer.h>
#include <schnet/module/ModuleUtils.h>
#include <tuple>

using Graph = schnet::core::Graph;

namespace schnet::module {
    class SchNetModelImpl : public torch::nn::Module {
        public:
            SchNetModelImpl(
                int hidden_dim, 
                int num_gaussians, 
                int num_filters, 
                int num_interactions,  
                float cutoff, 
                int type_num = 100
            );
            std::tuple<torch::Tensor, torch::Tensor> forward(Graph& graph, torch::optional<torch::Tensor> batch = torch::nullopt);

        private:
            torch::nn::Embedding embedding{nullptr};
            schnet::module::GaussianRBF rbf{nullptr}; 
            torch::nn::ModuleList interactions{nullptr};
            torch::nn::Sequential output{nullptr};   
    };
    TORCH_MODULE(SchNetModel);
}