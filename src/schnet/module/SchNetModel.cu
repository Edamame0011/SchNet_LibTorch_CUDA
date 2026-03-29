#include <schnet/module/SchNetModel.h>
#include <schnet/module/CalcForceFunction.h>

using namespace schnet::module;

SchNetModelImpl::SchNetModelImpl(
    int _hidden_dim, 
    int _num_gaussians, 
    int _num_filters, 
    int _num_interactions,  
    float _cutoff, 
    int _type_num
) {
    this->embedding = register_module("embedding", torch::nn::Embedding(_type_num, _hidden_dim));
    this->rbf = register_module("rbf", GaussianRBF(_num_gaussians, _cutoff));
    this->interactions = register_module("interactions", torch::nn::ModuleList());
    for (int i = 0; i < _num_interactions; i ++) {
        interactions->push_back(InteractionLayer(_hidden_dim, _num_gaussians, _num_filters, _cutoff));
    }
    this->output = register_module("output", torch::nn::Sequential(
        torch::nn::Linear(_hidden_dim, _hidden_dim / 2), 
        schnet::module::ShiftedSoftplus(), 
        torch::nn::Linear(_hidden_dim / 2, 1)
    ));
}

std::tuple<torch::Tensor, torch::Tensor> SchNetModelImpl::forward(Graph& graph, torch::optional<torch::Tensor> batch = torch::nullopt) {
    graph.edge_weight.set_requires_grad(true);

    auto h = embedding(graph.x);
    auto distances = torch::norm(graph.edge_weight, 2, {1});
    auto rbf_expansion = rbf(distances);

    for (auto& interaction : *interactions) {
        h = interaction->as<InteractionLayer>()->forward(h, distances, graph, rbf_expansion);
    }

    auto energy = output->forward(h);
    auto diff_E = torch::autograd::grad({energy.sum()}, {graph.edge_weight}, {}, true)[0];
    auto forces = CalcForceFunction::apply(
        diff_E, 
        graph.dst_node_ptr, 
        graph.src_list, 
        graph.src_node_ptr, 
        graph.dst_list, 
        graph.num_nodes
    );

    torch::Tensor total_energy;
    if (batch.has_value()) {
        auto b = batch.value();
        total_energy = torch::zeros({b.max().item<int64_t>() + 1}, energy.options());
        total_energy.index_add_(0, b, energy.squeeze());
    } else {
        total_energy = energy.sum();
    }

    return std::make_tuple(total_energy, forces);
}