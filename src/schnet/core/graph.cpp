#include <schnet/core/graph.h>

using Graph = schnet::core::Graph;

schnet::core::Graph::Graph(
    int _num_nodes, 
    int _num_edges, 
    torch::Device device
) : num_nodes(_num_nodes), 
    num_edges(_num_edges), 
    x(torch::zeros({_num_nodes}, torch::TensorOptions().device(device).dtype(torch::kInt64))), 
    edge_weight(torch::zeros({_num_edges}, torch::TensorOptions().device(device).dtype(torch::kFloat32))), 
    src_list(torch::zeros({_num_edges}, torch::TensorOptions().device(device).dtype(torch::kInt32))), 
    dst_list(torch::zeros({_num_edges}, torch::TensorOptions().device(device).dtype(torch::kInt32))), 
    src_node_ptr(torch::zeros({_num_nodes + 1}, torch::TensorOptions().device(device).dtype(torch::kInt32))), 
    dst_node_ptr(torch::zeros({_num_nodes + 1}, torch::TensorOptions().device(device).dtype(torch::kInt32))) {}

std::unique_ptr<Graph> schnet::core::generate_graph_from_device_ptr(
        int num_nodes, 
        int num_edges, 
        int64_t* x, 
        float* edge_weight, 
        int32_t* src_list, 
        int32_t* dst_list, 
        int32_t* src_node_ptr, 
        int32_t* dst_node_ptr
) {
    auto graph = std::make_unique<Graph>();

    auto opts_i64 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto opts_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    graph->x = torch::from_blob(x, {num_nodes}, opts_i64);
    graph->edge_weight = torch::from_blob(edge_weight, {num_edges, 3}, opts_f32);
    graph->src_list = torch::from_blob(src_list, {num_edges}, opts_i32);
    graph->dst_list = torch::from_blob(dst_list, {num_edges}, opts_i32);
    graph->src_node_ptr = torch::from_blob(src_node_ptr, {num_nodes + 1}, opts_i32);
    graph->dst_node_ptr = torch::from_blob(dst_node_ptr, {num_nodes + 1}, opts_i32);

    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;

    return graph;
}
