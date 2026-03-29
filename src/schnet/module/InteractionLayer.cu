#include <schnet/module/InteractionLayer.h>
#include <schnet/module/AggMessagesFunction.h>
#include <schnet/module/CutoffFunction.h>

using namespace schnet::module;

InteractionLayerImpl::InteractionLayerImpl(
    int _hidden_dim, 
    int _num_gaussians, 
    int _num_filters, 
    float _cutoff) {
    this->cutoff = _cutoff;
    this->num_filters = _num_filters;
    this->mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(_num_gaussians, _num_filters), 
        schnet::module::ShiftedSoftplus(), 
        torch::nn::Linear(_num_filters, _num_filters)
    ));
    this->lin1 = register_module("lin1", torch::nn::Linear(
        torch::nn::LinearOptions(_hidden_dim, _num_filters).bias(false)
    ));
    this->lin2 = register_module("lin2", torch::nn::Linear(_num_filters, _hidden_dim));
    this->act = register_module("act", schnet::module::ShiftedSoftplus());
}

torch::Tensor InteractionLayerImpl::forward(
    const torch::Tensor& x, 
    const torch::Tensor& distance, 
    Graph& graph, 
    const torch::Tensor& edge_attr
) {
    auto C = CutoffFunction::apply(distance, cutoff);

    auto W = (mlp->forward(edge_attr) * C.unsqueeze(-1)).contiguous();
    auto V = lin1(x).contiguous();

    auto agg_messages = AggMessagesFunction::apply(
        W, 
        V, 
        graph.dst_node_ptr, 
        graph.src_list, 
        graph.num_nodes
    );
    auto h = act(lin2(agg_messages));
    return h + x;
}