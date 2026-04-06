#include <schnet/app/app.h>
#include <schnet/module/SchNetModel.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace {
    std::map<std::string, int> atom_number_map = {
        {"H",  1},
        {"He", 2},
        {"Li", 3},
        {"Be", 4},
        {"B",  5},
        {"C",  6},
        {"N",  7},
        {"O",  8},
        {"F",  9},
        {"Ne", 10},
        {"Na", 11},
        {"Mg", 12},
        {"Al", 13},
        {"Si", 14},
        {"P",  15},
        {"S",  16},
        {"Cl", 17},
        {"Ar", 18},
        {"K",  19},
        {"Ca", 20}
    };

    //原子種類と原子質量を関連づけるmap
    std::map<std::string, double> atom_mass_map = {
        {"H",   1.0080},
        {"He",  4.0026},
        {"Li",  6.94},
        {"Be",  9.0122},
        {"B",   10.81},
        {"C",   12.011},
        {"N",   14.007},
        {"O",   15.999},
        {"F",   18.998},
        {"Ne",  20.180},
        {"Na",  22.990},
        {"Mg",  24.305},
        {"Al",  26.982},
        {"Si",  28.0855},
        {"P",   30.974},
        {"S",   32.06},
        {"Cl",  35.45},
        {"Ar",  39.95},
        {"K",   39.098},
        {"Ca",  40.078}
    };

    //文字列からenergyを見つける
    std::string find_energy(const std::string& input) {
        //開始位置のキーワード
        std::string start_tag = "energy=";
        //開始位置
        size_t start_position = input.find(start_tag);
        if(start_position != std::string::npos){
            start_position = start_position + start_tag.length();
            //開始位置から次の空白を探す
            size_t end_position = input.find(' ', start_position);
            if(end_position != std::string::npos){
                //抜き出す部分の長さ
                std::size_t length = end_position - start_position;

                //文字列の抜き出し
                std::string result = input.substr(start_position, length);

                return result;
            }
            else{ 
                throw std::runtime_error("終了の半角スペースが見つかりません。"); 
            }
        }
        else{ 
            throw std::runtime_error("ファイルにenergyデータが含まれていません。"); 
        }
    }
}

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

Graph schnet::app::build_graph(const torch::Tensor& positions, const torch::Tensor& atomic_numbers, float cutoff) {
    // 1. ノード数（原子数）の取得
    int num_nodes = positions.size(0);

    // 2. 全原子間のペアワイズ距離行列の計算 [num_nodes, num_nodes]
    torch::Tensor dist_mat = torch::cdist(positions, positions);

    // 3. カットオフ距離以下のペアを抽出するためのマスクを作成
    torch::Tensor mask = dist_mat.le(cutoff);
    // 自己ループ（同一原子間の距離0）はエッジとして含めないためFalseにする
    mask.fill_diagonal_(false);

    // 4. エッジの重み（距離）を抽出 [num_edges]
    // masked_select はメモリ上をリニアに走査するため、nonzeroと同じ順序で値を取得します
    torch::Tensor edge_weight = dist_mat.masked_select(mask);
    int num_edges = edge_weight.size(0);

    // 5. エッジの接続関係（インデックス）を取得 [num_edges, 2]
    // nonzero は (row, col) のインデックスペアを行優先で返します。
    // row を「終点 (dst)」、col を「始点 (src)」とみなすことで、
    // 自動的に dst についてソートされたリストが得られます。
    torch::Tensor edges = torch::nonzero(mask);
    
    // PyTorchのバージョンや抽出結果によっては次元が潰れるのを防ぐため、空の場合を考慮
    torch::Tensor dst, src_list;
    if (num_edges > 0) {
        dst = edges.select(1, 0).contiguous();
        src_list = edges.select(1, 1).contiguous();
    } else {
        dst = torch::empty({0}, torch::kLong);
        src_list = torch::empty({0}, torch::kLong);
    }

    // 6. dst_node_ptr の作成 (CSRフォーマットのポインタ配列)
    // bincount で各ノードを終点とするエッジの数をカウント
    torch::Tensor counts = torch::bincount(dst, /*weights=*/{}, /*minlength=*/num_nodes);
    
    // countの累積和 (cumsum) を取り、先頭に0を結合して [num_nodes + 1] の配列を作成
    torch::Tensor cumsum = counts.cumsum(0);
    torch::Tensor zero = torch::zeros({1}, counts.options());
    torch::Tensor dst_node_ptr = torch::cat({zero, cumsum});

    // 7. Graph構造体の組み立てと返却
    Graph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.x = atomic_numbers.clone(); // 特徴量として原子番号を保持
    g.edge_weight = edge_weight;
    g.src_list = src_list;
    g.dst_node_ptr = dst_node_ptr;

    return g;
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
    archive.read("dst_node_ptr", graph.dst_node_ptr);
}

int schnet::app::read_xyz_as_graph(
    const std::string& path, 
    std::vector<Graph>& graph_list, 
    std::vector<torch::Tensor>& energy_list, 
    std::vector<torch::Tensor>& forces_list, 
    float cutoff
) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("構造ファイルを開けません。");

    std::string line;
    size_t num_atoms = 0;
    size_t num_structures = 0;
    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 3>> forces;
    std::vector<int64_t> atomic_numbers;
    float current_energy = 0.0f;

    auto opt_f_cpu = torch::TensorOptions().dtype(torch::kFloat);
    auto opt_i_cpu = torch::TensorOptions().dtype(torch::kInt64);

    auto save_current_structure = [&]() {
        if (positions.empty()) return; // 空なら何もしない

        // 1. CPU上でTensorを作成 (Shapeを正しく指定)
        torch::Tensor t_pos_cpu = torch::from_blob(positions.data(), {static_cast<int64_t>(num_atoms), 3}, opt_f_cpu);
        torch::Tensor t_force_cpu = torch::from_blob(forces.data(), {static_cast<int64_t>(num_atoms), 3}, opt_f_cpu);
        torch::Tensor t_num_cpu = torch::from_blob(atomic_numbers.data(), {static_cast<int64_t>(num_atoms)}, opt_i_cpu);

        // 2. GPUへ転送 (ここで暗黙的にディープコピーされるため、この後vectorをclearしても安全)
        torch::Tensor t_pos = t_pos_cpu.to(torch::kCUDA);
        torch::Tensor t_force = t_force_cpu.to(torch::kCUDA);
        torch::Tensor t_num = t_num_cpu.to(torch::kCUDA);
        
        // エネルギーはスカラー値から直接Tensorを作成
        torch::Tensor t_energy = torch::tensor({current_energy}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

        // 3. グラフの作成
        Graph g = build_graph(t_pos, t_num, cutoff);
        
        graph_list.push_back(g);
        forces_list.push_back(t_force);
        energy_list.push_back(t_energy);

        num_structures++;
        
        // バッファのクリア
        positions.clear();
        forces.clear();
        atomic_numbers.clear();
    };

    int j = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // 数値かどうか判定
        if(std::all_of(line.begin(), line.end(), [](unsigned char c){ return std::isdigit(c) || std::isspace(c); })){
            // 新しい構造の開始前に、前の構造を保存
            save_current_structure();

            num_atoms = std::stoi(line);
            positions.reserve(num_atoms);
            forces.reserve(num_atoms);
            atomic_numbers.reserve(num_atoms);
            j = 0;
        } else {
            if (j == 1) {
                current_energy = std::stof(find_energy(line));
            } else if (j > 1) {
                // 原子座標データのパース
                std::istringstream iss(line);
                std::array<float, 3> pos_arr;
                std::array<float, 3> force_arr;
                std::string atom_type;

                iss >> atom_type >> pos_arr[0] >> pos_arr[1] >> pos_arr[2] >> force_arr[0] >> force_arr[1] >> force_arr[2];
                
                atomic_numbers.push_back(atom_number_map.at(atom_type));

                positions.push_back(pos_arr);
                forces.push_back(force_arr);
            }
            j++;
        }
    }

    // ファイル末尾に到達後、最後の構造を保存
    save_current_structure();

    return num_structures;
}