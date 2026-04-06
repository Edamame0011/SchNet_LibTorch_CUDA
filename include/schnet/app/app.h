#pragma once

#include <schnet/module/SchNetModel.h>

using SchNetModel = schnet::module::SchNetModel;
using Graph = schnet::core::Graph;

namespace schnet::app {
    void train(
        SchNetModel& model, 
        Graph* train_graph, 
        torch::Tensor* train_energy, 
        torch::Tensor* train_forces, 
        Graph* test_graph, 
        torch::Tensor* test_energy, 
        torch::Tensor* test_forces, 
        int num_train_datas, 
        int num_test_datas, 
        int num_epochs, 
        float learning_rate, 
        float energy_weight, 
        float force_weight, 
        torch::Device device = torch::kCUDA
    );

    int read_xyz_as_graph(
        const std::string& path, 
        std::vector<Graph>& graph_list, 
        std::vector<torch::Tensor>& energy_list, 
        std::vector<torch::Tensor>& forces_list, 
        float cutoff
    );
    Graph build_graph(const torch::Tensor& positions, const torch::Tensor& atomic_numbers, float cutoff);

    void save_graph(Graph& graph, const std::string& path);
    void load_graph(Graph& graph, const std::string& path);
}