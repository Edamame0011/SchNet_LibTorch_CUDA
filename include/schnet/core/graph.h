#pragma once

#include <torch/extension.h>

namespace schnet::core {
    struct Graph {
        int num_nodes;
        int num_edges;
        
        torch::Tensor x;            // [num_nodes]
        torch::Tensor edge_weight;  // [num_edges]
        torch::Tensor src_list;       // [num_edges]
        torch::Tensor dst_node_ptr;   // [num_nodes + 1]
    };
}