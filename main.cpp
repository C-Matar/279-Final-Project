#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "MLP.h"

std::vector<Molecule> load_molecules_from_file(const std::string& filename,
                                             const std::map<std::string, int>& atom_type_to_index) {
    std::vector<Molecule> molecules;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        throw std::invalid_argument("Error: Could not open file.");
    }

    std::string line;
    int line_number = 0;
    int molecule_start_line = 0;

    while (std::getline(infile, line)) {
        line_number++;
        molecule_start_line = line_number;
        std::stringstream ss_header(line);
        int num_atoms;
        double total_energy;

        if (!(ss_header >> num_atoms >> total_energy)) {
            std::cerr << "Error parsing header line " << line_number << ": \"" << line << "\". Skipping entry." << std::endl;
            continue;
        }

        Molecule current_molecule;
        current_molecule.total_energy = total_energy;
        current_molecule.atoms.reserve(num_atoms);

        bool read_error = false;
        for (int i = 0; i < num_atoms; ++i) {
            if (!std::getline(infile, line)) {
                std::cerr << "Error: Unexpected end of file while reading atom " << (i + 1)
                          << "/" << num_atoms << " for molecule starting on line "
                          << molecule_start_line << std::endl;
                read_error = true;
                break;
            }
            line_number++;
            int atom_line_number = line_number;

            std::stringstream ss_atom(line);
            std::string atom_type_str;
            Atom current_atom;

            // Read atom line (Type X Y Z)
            if (!(ss_atom >> atom_type_str >> current_atom.x >> current_atom.y >> current_atom.z)) {
                std::cerr << "Error parsing atom line " << atom_line_number << ": \"" << line << "\"" << std::endl;
                read_error = true;
                break;
            }

            if (atom_type_to_index.find(atom_type_str) == atom_type_to_index.end()) {
                 std::cerr << "Warning: Unknown atom type '" << atom_type_str
                           << "' found on line " << atom_line_number << ". Skipping molecule starting on line " << molecule_start_line << "." << std::endl;
                  read_error = true;
                  break;
            }

            current_atom.type = atom_type_str;
            current_molecule.atoms.push_back(current_atom);
        }

        if (!read_error && current_molecule.atoms.size() == static_cast<size_t>(num_atoms)) {
             molecules.push_back(current_molecule);
        } else if (!read_error && current_molecule.atoms.size() != static_cast<size_t>(num_atoms)) {
             std::cerr << "Internal error: Atom count mismatch (" << current_molecule.atoms.size()
                       << " read vs " << num_atoms << " expected) for molecule starting on line "
                       << molecule_start_line << std::endl;
        }
    }
    infile.close();
    std::cout << "Successfully loaded " << molecules.size() << " molecules from " << filename << std::endl;
    return molecules;
}

std::pair<double, double> calculate_mean_stddev(const std::vector<double>& data) {
    if (data.empty()) {
        return {0.0, 1.0};
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double variance = (sq_sum / data.size()) - (mean * mean);
    if (variance <= std::numeric_limits<double>::epsilon()) {
        return {mean, 1.0};
    }
    double stddev = std::sqrt(variance);
    return {mean, stddev};
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_molecule_data.txt>" << std::endl;
        return 1;
    }
    std::string data_filename = argv[1];
    std::vector<std::string> atom_types_list = {"H", "C", "N", "O"};
    double TRAIN_SPLIT_RATIO = 0.9;

    std::map<std::string, int> atom_type_to_index;
    for (size_t i = 0; i < atom_types_list.size(); ++i) {
        atom_type_to_index[atom_types_list[i]] = static_cast<int>(i);
    }

    int num_atom_types = static_cast<int>(atom_types_list.size());
    int input_layer_size = 2 * num_atom_types + 1; // hard-coded, 2 one-hot atom types + 1 normalized distance
    std::vector<int> hidden_layer_sizes = {128, 64};
    int output_layer_size = 1; // hard-coded, output 1 scalar energy value
    double learning_rate = 0.0001;
    int epochs = 1000;

    std::mt19937 rng(42);

    std::vector<Molecule> all_molecules = load_molecules_from_file(data_filename, atom_type_to_index);

    std::shuffle(all_molecules.begin(), all_molecules.end(), rng);

    size_t train_size = static_cast<size_t>(all_molecules.size() * TRAIN_SPLIT_RATIO);

    std::vector<Molecule> training_dataset(all_molecules.begin(), all_molecules.begin() + train_size);
    std::vector<Molecule> testing_dataset(all_molecules.begin() + train_size, all_molecules.end());

    std::cout << "Data split: " << training_dataset.size() << " training molecules, "
              << testing_dataset.size() << " testing molecules." << std::endl;

    std::vector<double> all_distances_train;
    for (const auto& mol : training_dataset) {
        if (mol.atoms.size() < 2) continue;
        for (size_t i = 0; i < mol.atoms.size(); ++i) {
            for (size_t j = i + 1; j < mol.atoms.size(); ++j) {
                all_distances_train.push_back(calculate_distance(mol.atoms[i], mol.atoms[j]));
            }
        }
    }
    auto dist_stats = calculate_mean_stddev(all_distances_train);
    double distance_mean = dist_stats.first;
    double distance_stddev = dist_stats.second;

    std::cout << "Distance Stats (Training Data): Mean=" << distance_mean << ", StdDev=" << distance_stddev << std::endl;


    MLP nn(input_layer_size, hidden_layer_sizes, output_layer_size, learning_rate);
    std::cout << "MLP Created. Architecture (Layer Sizes): [";
    for (int size : nn.get_layer_sizes()) {
        std::cout << size << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\n--- Starting Training ---" << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_epoch_loss = 0.0;
        int molecule_count = 0;

        // Shuffle training data each epoch
        std::shuffle(training_dataset.begin(), training_dataset.end(), rng);

        for (const auto& molecule : training_dataset) {
            double molecule_loss = nn.train_molecule(molecule, atom_type_to_index, num_atom_types,
                                                     distance_mean, distance_stddev);
            total_epoch_loss += molecule_loss;
            molecule_count++;
        }

        double avg_loss = (molecule_count == 0) ? 0.0 : total_epoch_loss / molecule_count;
        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            std::cout << "Epoch [" << std::setw(3) << (epoch + 1) << "/" << epochs
                      << "], Average Training Loss: " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
        }
    }
    std::cout << "--- Training Finished ---\n" << std::endl;


    std::cout << "--- Evaluating on Test Set ---" << std::endl;
    double total_test_abs_error = 0.0;
    int test_molecule_count = 0;

    for (const auto& test_molecule : testing_dataset) {

        double predicted_energy = nn.predict_molecule_energy(test_molecule, atom_type_to_index, num_atom_types, 
                                                            distance_mean, distance_stddev);

        double true_energy = test_molecule.total_energy;
        double error = predicted_energy - true_energy;
        total_test_abs_error += std::abs(error);
        test_molecule_count++;

        if ((test_molecule_count + 1) % 10 == 0 || test_molecule_count == 1) {
            std::cout << "Test Mol #" << test_molecule_count << ": True E = " << std::fixed << std::setprecision(4) << true_energy
                        << ", Pred E = " << predicted_energy << ", Abs Error = " << std::abs(error) << std::endl;
        }
    }

    double mean_absolute_error = total_test_abs_error / test_molecule_count;
    std::cout << "\nTest Set Mean Absolute Error (MAE): " << std::fixed << std::setprecision(6) << mean_absolute_error << std::endl;

    return 0;
}