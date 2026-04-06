#include <schnet/app/app.h>
#include <string>

int main() {
    constexpr int hidden_dim = 1;
    constexpr int num_gaussians = 1;
    constexpr int num_filters = 1;
    constexpr int num_interactions = 1;
    constexpr float cutoff = 5.0f;
    constexpr int type_num = 100;
    constexpr int num_epochs = 100;
    constexpr float learning_rate = 0.5f;
    constexpr float energy_weight = 0.99f;
    constexpr float force_weight = 100.0f;

    const std::string save_path = "";

    const std::string train_path = "";
    const std::string test_path = "";

    std::vector<Graph> train_graph_list;
    std::vector<torch::Tensor> train_energy; 
    std::vector<torch::Tensor> train_forces;
    std::vector<Graph> test_graph_list;
    std::vector<torch::Tensor> test_energy; 
    std::vector<torch::Tensor> test_forces;

    int num_train_datas = schnet::app::read_xyz_as_graph(
        train_path, 
        train_graph_list, 
        train_energy, 
        train_forces, 
        cutoff
    );
    int num_test_datas = schnet::app::read_xyz_as_graph(
        test_path, 
        test_graph_list, 
        test_energy, 
        test_forces, 
        cutoff
    );

    schnet::module::SchNetModel model(
        hidden_dim, 
        num_gaussians, 
        num_filters, 
        num_interactions, 
        cutoff, 
        type_num
    );

    schnet::app::train(
        model, 
        train_graph_list.data(), 
        train_energy.data(), 
        train_forces.data(), 
        test_graph_list.data(), 
        test_energy.data(), 
        test_forces.data(), 
        num_train_datas, 
        num_test_datas, 
        num_epochs, 
        learning_rate, 
        energy_weight, 
        force_weight
    );

    torch::save(model, save_path);
}