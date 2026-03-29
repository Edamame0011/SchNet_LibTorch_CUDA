#include <schnet/app/app.h>
#include <schnet/module/SchNetModel.h>
#include <iostream>

void schnet::app::train(
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
    torch::Device device
) {
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    torch::optim::StepLR scheduler(optimizer, 100, 0.5);
    float loss_total = 0.0f;
    float loss_e_total = 0.0f;
    float loss_f_total = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch ++) {
        // 訓練ループ
        loss_total = 0.0f;
        loss_e_total = 0.0f;
        loss_f_total = 0.0f;
        model->train();
        for (int i = 0; i < num_train_datas; i ++) {
            optimizer.zero_grad();

            auto [pred_energy, pred_forces] = model->forward(train_graph[i]);
            auto loss_e = torch::mse_loss(pred_energy, train_energy[i]);
            auto loss_f = torch::mse_loss(pred_forces, train_forces[i]);

            auto l = loss_e * energy_weight + loss_f * force_weight;
            l.backward();
            optimizer.step();

            loss_total += l.item<float>();
            loss_e_total += loss_e.item<float>();
            loss_f_total += loss_f.item<float>();
        }
        loss_total /= num_train_datas;
        loss_e_total /= num_train_datas;
        loss_f_total /= num_train_datas;
        scheduler.step();
        std::cout << "epoch: train" << epoch << " loss_total: " << loss_total << " loss_e: " << loss_e_total << " loss_f: " << loss_f_total << std::endl; 
        // 評価ループ
        loss_total = 0.0f;
        loss_e_total = 0.0f;
        loss_f_total = 0.0f;
        model->eval();
        for (int i = 0; i < num_test_datas; i ++) {
            auto [pred_energy, pred_forces] = model->forward(test_graph[i]);
            auto loss_e = torch::mse_loss(pred_energy, test_energy[i]);
            auto loss_f = torch::mse_loss(pred_forces, test_forces[i]);

            auto l = loss_e * energy_weight + loss_f * force_weight;

            loss_total += l.item<float>();
            loss_e_total += loss_e.item<float>();
            loss_f_total += loss_f.item<float>();
        }
        loss_total /= num_test_datas;
        loss_e_total /= num_test_datas;
        loss_f_total /= num_test_datas;
        std::cout << "epoch: test" << epoch << " loss_total: " << loss_total << " loss_e: " << loss_e_total << " loss_f: " << loss_f_total << std::endl; 
    }
}

void schnet::app::save_graph(Graph& graph, const std::string& path) {
    torch::serialize::OutputArchive archive;
        
    // int型は0次元のTensorに変換して保存
    archive.write("num_nodes", torch::tensor(graph.num_nodes));
    archive.write("num_edges", torch::tensor(graph.num_edges));
    
    // Tensorの保存
    archive.write("x", graph.x);
    archive.write("edge_weight", graph.edge_weight);
    archive.write("src_list", graph.src_list);
    archive.write("dst_list", graph.dst_list);
    archive.write("src_node_ptr", graph.src_node_ptr);
    archive.write("dst_node_ptr", graph.dst_node_ptr);
    
    archive.save_to(path);
}

void schnet::app::load_graph(Graph& graph, const std::string& path) {
torch::serialize::InputArchive archive;
    archive.load_from(path);

    // int型の復元
    torch::Tensor tmp_num_nodes, tmp_num_edges;
    archive.read("num_nodes", tmp_num_nodes);
    archive.read("num_edges", tmp_num_edges);
    graph.num_nodes = tmp_num_nodes.item<int>();
    graph.num_edges = tmp_num_edges.item<int>();

    // Tensorの復元
    archive.read("x", graph.x);
    archive.read("edge_weight", graph.edge_weight);
    archive.read("src_list", graph.src_list);
    archive.read("dst_list", graph.dst_list);
    archive.read("src_node_ptr", graph.src_node_ptr);
    archive.read("dst_node_ptr", graph.dst_node_ptr);
}

std::unique_ptr<Graph> schnet::app::read_xyz_as_graph(const std::string& path, float cutoff) {
    
}