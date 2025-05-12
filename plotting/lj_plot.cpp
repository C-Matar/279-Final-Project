#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <iomanip>
#include "MLP.h"

struct GroupedConformer {
    std::vector<PairSample> pairs;
    double energy;
};

std::vector<GroupedConformer> load_grouped_csv(const std::string& filename) {
    std::vector<GroupedConformer> conformers;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    std::getline(file, line);

    std::map<std::string, GroupedConformer> conf_map;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string mol_id;
        std::string formula;
        PairSample sample;
        char comma;

        std::getline(ss, mol_id, ',');
        ss >> sample.z1 >> comma >> sample.z2 >> comma >> sample.distance >> comma >> sample.energy >> comma >> formula;

        conf_map[mol_id].pairs.push_back(sample);
        conf_map[mol_id].energy = sample.energy;
    }

    for (auto& [_, mol] : conf_map)
        conformers.push_back(std::move(mol));

    std::cout << "Loaded " << conformers.size() << " conformers.\n";
    return conformers;
}

double normalize_value(double x, double mean, double stddev) {
    return stddev > 1e-6 ? (x - mean) / stddev : x;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " z1 z2 output.csv trained_model.txt training_data.csv\n";
        return 1;
    }

    int z1 = std::stoi(argv[1]);
    int z2 = std::stoi(argv[2]);
    std::string out_file = argv[3];
    std::string model_weights = argv[4];
    std::string training_csv = argv[5];

    auto conformers = load_grouped_csv(training_csv);

    std::set<int> atom_numbers;
    for (const auto& mol : conformers) {
        for (const auto& p : mol.pairs) {
            atom_numbers.insert(p.z1);
            atom_numbers.insert(p.z2);
        }
    }

    std::vector<int> z_list(atom_numbers.begin(), atom_numbers.end());
    std::map<int, int> z_to_idx;
    for (size_t i = 0; i < z_list.size(); ++i)
        z_to_idx[z_list[i]] = static_cast<int>(i);

    int num_atom_types = z_list.size();
    int input_dim = 2 * num_atom_types + 1;
    std::vector<int> hidden_layers = {256, 128, 64};
    MLP nn(input_dim, hidden_layers, 1, 0.0001);
    nn.load_weights(model_weights);
    std::cout << "Loaded weights from " << model_weights << "\n";

    const double mean_d = 2.34664;
    const double std_d  = 0.934951;

    double d_min = 0.5;
    double d_max = 5.0;
    double step = 0.01;
    std::vector<double> distances;
    for (double d = d_min; d <= d_max; d += step)
        distances.push_back(d);

    std::ofstream out(out_file);
    out << "distance,predicted_energy\n";

    int i1 = z_to_idx[z1];
    int i2 = z_to_idx[z2];
    std::cout << "Encoding input for atom types " << z1 << " and " << z2 << i1 << i2 << "\n";
    

    for (double d : distances) {
        double norm_d = normalize_value(d, mean_d, std_d);
        Vector input = nn.encode_input(i1, i2, norm_d, num_atom_types);
        double energy = nn.predict_pair_energy(input);
        out << d << "," << energy << "\n";
    }

    std::cout << "Saved Lennard-Jones–like plot data to " << out_file << "\n";
    return 0;
}
