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

Graph schnet::app::read_xyz_as_graph(const std::string& path, float cutoff) {
    
}