#pragma once

#include <torch/extension.h>

namespace schnet::core {
    struct Graph {
        int num_nodes;
        int num_edges;
        
        torch::Tensor x;            // [num_nodes]
        torch::Tensor edge_weight;  // [num_edges]
        torch::Tensor src_list, dst_list;           // [num_edges]
        torch::Tensor src_node_ptr, dst_node_ptr;   // [num_nodes + 1]

        Graph(int num_nodes, int num_edges, torch::Device device = torch::kCUDA);
    };

    std::unique_ptr<Graph> generate_graph_from_device_ptr(
        int num_nodes, 
        int num_edges, 
        int64_t* x, 
        float* edge_weight, 
        int32_t* src_list, 
        int32_t* dst_list, 
        int32_t* src_node_ptr, 
        int32_t* dst_node_ptr
    );
}