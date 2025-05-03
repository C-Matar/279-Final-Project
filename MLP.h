#pragma once

#include <vector>
#include <string>
#include <cmath>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct PairSample {
    int z1;
    int z2;
    double distance;
    double energy;
};

class MLP {
public:
    MLP(int input_dim, const std::vector<int>& hidden_layers, int output_dim, double lr);

    Vector encode_input(int i1, int i2, double norm_d, int num_atom_types);
    double predict_pair_energy(const Vector& input);
    void forward_pair(const Vector& input);
    void backward_pair(const Vector& input, double grad, std::vector<Matrix>& dw, std::vector<Vector>& db);
    void apply_gradients(const std::vector<Matrix>& dw, const std::vector<Vector>& db);

    const std::vector<int>& get_layer_sizes() const;

private:
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    std::vector<Vector> activations;
    std::vector<Vector> zs;
    std::vector<int> layer_sizes;
    double learning_rate;

    double relu(double x);
    double relu_deriv(double x);
};
