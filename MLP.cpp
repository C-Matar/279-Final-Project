#include "MLP.h"
#include <random>
#include <iostream>
#include <cassert>

MLP::MLP(int input_dim, const std::vector<int>& hidden_layers, int output_dim, double lr) {
    learning_rate = lr;
    layer_sizes.push_back(input_dim);
    for (int h : hidden_layers) layer_sizes.push_back(h);
    layer_sizes.push_back(output_dim);

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        Matrix W(layer_sizes[i + 1], Vector(layer_sizes[i]));
        Vector b(layer_sizes[i + 1]);
        for (auto& row : W)
            for (auto& w : row)
                w = dist(gen) * std::sqrt(2.0 / layer_sizes[i]);
        weights.push_back(W);
        biases.push_back(b);
    }
}

Vector MLP::encode_input(int i1, int i2, double norm_d, int num_atom_types) {
    Vector input(2 * num_atom_types + 1, 0.0);
    input[i1] = 1.0;
    input[num_atom_types + i2] = 1.0;
    input[2 * num_atom_types] = norm_d;
    return input;
}

double MLP::predict_pair_energy(const Vector& input) {
    Vector a = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        Vector z(layer_sizes[i + 1], 0.0);
        for (size_t j = 0; j < z.size(); ++j) {
            for (size_t k = 0; k < a.size(); ++k)
                z[j] += weights[i][j][k] * a[k];
            z[j] += biases[i][j];
        }
        a.resize(z.size());
        for (size_t j = 0; j < z.size(); ++j)
            a[j] = (i == weights.size() - 1) ? z[j] : relu(z[j]);
    }
    return a[0];
}

void MLP::forward_pair(const Vector& input) {
    activations.clear();
    zs.clear();
    Vector a = input;
    activations.push_back(a);
    for (size_t i = 0; i < weights.size(); ++i) {
        Vector z(layer_sizes[i + 1], 0.0);
        for (size_t j = 0; j < z.size(); ++j) {
            for (size_t k = 0; k < a.size(); ++k)
                z[j] += weights[i][j][k] * a[k];
            z[j] += biases[i][j];
        }
        zs.push_back(z);
        a.resize(z.size());
        for (size_t j = 0; j < z.size(); ++j)
            a[j] = (i == weights.size() - 1) ? z[j] : relu(z[j]);
        activations.push_back(a);
    }
}

void MLP::backward_pair(const Vector& input, double grad, std::vector<Matrix>& dw, std::vector<Vector>& db) {
    Vector delta = {grad}; // dL/dy
    for (int l = static_cast<int>(weights.size()) - 1; l >= 0; --l) {
        for (size_t i = 0; i < dw[l].size(); ++i) {
            for (size_t j = 0; j < dw[l][i].size(); ++j) {
                dw[l][i][j] += delta[i] * activations[l][j];
            }
            db[l][i] += delta[i];
        }

        if (l == 0) break;

        Vector new_delta(layer_sizes[l], 0.0);
        for (size_t j = 0; j < new_delta.size(); ++j) {
            for (size_t i = 0; i < delta.size(); ++i)
                new_delta[j] += weights[l][i][j] * delta[i];
            new_delta[j] *= relu_deriv(zs[l - 1][j]);
        }
        delta = new_delta;
    }
}

void MLP::apply_gradients(const std::vector<Matrix>& dw, const std::vector<Vector>& db) {
    for (size_t l = 0; l < weights.size(); ++l) {
        for (size_t i = 0; i < weights[l].size(); ++i) {
            for (size_t j = 0; j < weights[l][i].size(); ++j)
                weights[l][i][j] -= learning_rate * dw[l][i][j];
            biases[l][i] -= learning_rate * db[l][i];
        }
    }
}

const std::vector<int>& MLP::get_layer_sizes() const {
    return layer_sizes;
}

double MLP::relu(double x) {
    return std::max(0.0, x);
}

double MLP::relu_deriv(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}
