#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <set>
#include <iomanip>
#include "MLP.h"

struct GroupedConformer {
    std::vector<PairSample> pairs;
    double energy;
};

std::vector<GroupedConformer> load_grouped_csv(const std::string& filename) {
    std::vector<GroupedConformer> conformers;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::getline(file, line); // skip header

    std::map<std::string, GroupedConformer> conf_map;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string mol_id;
        PairSample sample;
        char comma;

        std::getline(ss, mol_id, ',');
        ss >> sample.z1 >> comma >> sample.z2 >> comma >> sample.distance >> comma >> sample.energy;

        conf_map[mol_id].pairs.push_back(sample);
        conf_map[mol_id].energy = sample.energy;
    }

    for (auto& [_, mol] : conf_map)
        conformers.push_back(std::move(mol));

    std::cout << "Loaded " << conformers.size() << " conformers.\n";
    return conformers;
}

std::pair<double, double> calculate_mean_stddev(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) sum += v;
    double mean = sum / values.size();
    double sq_sum = 0.0;
    for (double v : values) sq_sum += (v - mean) * (v - mean);
    double stddev = std::sqrt(sq_sum / values.size());
    return {mean, stddev};
}

double normalize_value(double x, double mean, double stddev) {
    return stddev > 1e-6 ? (x - mean) / stddev : x;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " ani_pairs_with_ids.csv\n";
        return 1;
    }

    std::string csv_file = argv[1];
    auto conformers = load_grouped_csv(csv_file);

    // Build atom-type list and index map
    std::set<int> atom_numbers;
    for (const auto& mol : conformers)
        for (const auto& p : mol.pairs)
            atom_numbers.insert(p.z1), atom_numbers.insert(p.z2);

    std::vector<int> z_list(atom_numbers.begin(), atom_numbers.end());
    std::map<int, int> z_to_idx;
    for (size_t i = 0; i < z_list.size(); ++i)
        z_to_idx[z_list[i]] = static_cast<int>(i);

    int num_atom_types = z_list.size();
    int input_dim = 2 * num_atom_types + 1;
    std::vector<int> hidden_layers = {256, 128, 64};

    MLP nn(input_dim, hidden_layers, 1, 0.0001);
    nn.load_weights("trained_model.txt");
    std::cout << "Loaded weights from trained_model.txt\n";

    // Normalize distance
    std::vector<double> distances;
    for (const auto& mol : conformers)
        for (const auto& p : mol.pairs)
            distances.push_back(p.distance);
    auto [mean_d, std_d] = calculate_mean_stddev(distances);

    // Predicts and saves
    std::ofstream out("predictions.csv");
    out << "molecule_id,true_energy,predicted_energy,error\n";

    int mol_idx = 0;
    for (const auto& mol : conformers) {
        std::vector<Vector> inputs;
        for (const auto& p : mol.pairs) {
            int i1 = z_to_idx[p.z1];
            int i2 = z_to_idx[p.z2];
            double norm_d = normalize_value(p.distance, mean_d, std_d);
            inputs.push_back(nn.encode_input(i1, i2, norm_d, num_atom_types));
        }

        double pred_energy = 0.0;
        for (const auto& input : inputs)
            pred_energy += nn.predict_pair_energy(input);

        double err = std::abs(pred_energy - mol.energy);
        out << "mol_" << std::setfill('0') << std::setw(5) << mol_idx++
            << "," << mol.energy << "," << pred_energy << "," << err << "\n";
    }

    std::cout << "Saved predictions to predictions.csv\n";
    return 0;
}
