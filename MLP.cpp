#include "MLP.h"

#include <iostream>

// --- Activation functions ---
double relu(double x) {
    return std::max(0.0, x);
}
double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}
double linear(double x) {
    return x;
}
double linear_derivative(double x) {
    return 1.0;
}

// --- Helper functions ---
double calculate_distance(const Atom& a1, const Atom& a2) {
    double dx = a1.x - a2.x;
    double dy = a1.y - a2.y;
    double dz = a1.z - a2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}
double normalize_value(double value, double mean, double stddev) {
    // Handle cases where stddev is zero or very close to it
    if (std::abs(stddev) < std::numeric_limits<double>::epsilon()) {
        return value - mean;
    }
    return (value - mean) / stddev;
}

// --- Vector/Matrix operations ---
double dot_product(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for dot product.");
    }
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

Vector matrix_vector_multiply(const Matrix& M, const Vector& V) {
    if (M.empty()) return {};
    size_t num_rows = M.size();
    size_t num_cols = M[0].size();
    if (num_cols != V.size()) {
        throw std::invalid_argument("Matrix columns must match vector size for multiplication.");
    }
    Vector result(num_rows, 0.0);
    for (size_t i = 0; i < num_rows; ++i) {
        result[i] = dot_product(M[i], V);
    }
    return result;
}

Vector vector_add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition.");
    }
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vector vector_subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Vector elementwise_multiply(const Vector& a, const Vector& b) {
     if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for element-wise multiplication.");
    }
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

Matrix transpose(const Matrix& M) {
    if (M.empty()) return {};
    size_t rows = M.size();
    size_t cols = M[0].size();
    Matrix T(cols, Vector(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (j >= M[i].size()) {
                 throw std::out_of_range("Inconsistent matrix dimensions during transpose.");
            }
            T[j][i] = M[i][j];
        }
    }
    return T;
}

Matrix outer_product(const Vector& a, const Vector& b) {
    size_t rows = a.size();
    size_t cols = b.size();
    Matrix result(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

// --- MLP Class ---
MLP::MLP(int input_size, const std::vector<int>& hidden_sizes, int output_size, double lr)
    : input_size_(input_size),
      hidden_sizes_(hidden_sizes),
      output_size_(output_size),
      learning_rate(lr),
      rng_(std::random_device{}()) // Seed RNG
{
    if (input_size <= 0 || lr <= 0.0) {
         throw std::invalid_argument("MLP input size and learning rate must be positive.");
    }
    if (output_size != 1) {
         throw std::invalid_argument("MLP output size must be 1 for pair energy prediction.");
    }
    for(int size : hidden_sizes) {
        if (size <= 0) throw std::invalid_argument("Hidden layer sizes must be positive.");
    }

    layer_sizes_.push_back(input_size_);
    layer_sizes_.insert(layer_sizes_.end(), hidden_sizes_.begin(), hidden_sizes_.end());
    layer_sizes_.push_back(output_size_);

    initialize_weights();
}

void MLP::initialize_weights() {
    weights_.resize(layer_sizes_.size() - 1);
    biases_.resize(layer_sizes_.size() - 1);

    // Using Xavier initialization
    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dist(-limit, limit);

        weights_[i].resize(fan_out, Vector(fan_in));
        biases_[i].resize(fan_out);

        for (int r = 0; r < fan_out; ++r) {
            for (int c = 0; c < fan_in; ++c) {
                weights_[i][r][c] = dist(rng_);
            }
            // Biases initialized to zero
            biases_[i][r] = 0.0;
        }
    }
}

Vector MLP::encode_input(int type1_idx, int type2_idx, double normalized_distance, int num_atom_types) const {
    int expected_input_size = 2 * num_atom_types + 1;
    if (input_size_ != expected_input_size) {
         throw std::logic_error("MLP input size (" + std::to_string(input_size_)
                                + ") does not match expected encoding size ("
                                + std::to_string(expected_input_size) + ").");
    }
     if (type1_idx < 0 || type1_idx >= num_atom_types || type2_idx < 0 || type2_idx >= num_atom_types) {
        throw std::out_of_range("Atom type index out of range during encode_input.");
    }

    Vector input(input_size_, 0.0);

    // One-hot encode atoms
    input[type1_idx] = 1.0;
    input[num_atom_types + type2_idx] = 1.0;

    // Normalized distance
    input[2 * num_atom_types] = normalized_distance;

    return input;
}

Vector MLP::forward_pair(const Vector& pair_input) {
    if (pair_input.size() != input_size_) {
        throw std::invalid_argument("Input vector size mismatch for pair forward pass.");
    }

    // Clear previous pair's intermediate values
    current_pair_layer_outputs_.clear();
    current_pair_layer_inputs_.clear();

    Vector current_activation = pair_input;
    current_pair_layer_outputs_.push_back(current_activation); // Store initial input as layer 0 output

    // Iterate through layers
    for (size_t i = 0; i < weights_.size(); ++i) {
        // Linear transformation: z = W * a_prev + b
        Vector z = matrix_vector_multiply(weights_[i], current_activation);
        z = vector_add(z, biases_[i]);
        current_pair_layer_inputs_.push_back(z); // Store pre-activation (z) for layer i+1

        // Apply activation function: linear for the last layer, relu for others
        bool is_output_layer = (i == weights_.size() - 1);
        Vector next_activation(z.size());
        for (size_t j = 0; j < z.size(); ++j) {
            next_activation[j] = is_output_layer ? linear(z[j]) : relu(z[j]);
        }

        current_activation = next_activation;
        current_pair_layer_outputs_.push_back(current_activation); // Store activation (a) for layer i+1
    }

    return current_activation;
}


double MLP::predict_pair_energy(const Vector& pair_input) {
    if (pair_input.size() != input_size_) {
        throw std::invalid_argument("Input vector size mismatch for pair prediction.");
    }

    Vector current_activation = pair_input;
    for (size_t i = 0; i < weights_.size(); ++i) {
        Vector z = matrix_vector_multiply(weights_[i], current_activation);
        z = vector_add(z, biases_[i]);
        bool is_output_layer = (i == weights_.size() - 1);
        Vector next_activation(z.size());
        for (size_t j = 0; j < z.size(); ++j) {
            next_activation[j] = is_output_layer ? linear(z[j]) : relu(z[j]);
        }
        current_activation = next_activation;
    }

    return current_activation[0];
}


void MLP::backward_pair(const Vector& pair_input, double molecule_error_raw, // dLoss/dE_total
                       std::vector<Matrix>& grad_weights_accum,
                       std::vector<Vector>& grad_biases_accum)
{
    if (current_pair_layer_outputs_.empty() || current_pair_layer_inputs_.empty()) {
        throw std::logic_error("forward_pair must precede backward_pair for the same input.");
    }
    if (grad_weights_accum.size() != weights_.size() || grad_biases_accum.size() != biases_.size()){
         throw std::logic_error("Gradient accumulator dimensions mismatch.");
    }

    int num_layers = layer_sizes_.size();
    int last_layer_idx = num_layers - 1; // Index of the output layer
    int last_weight_idx = weights_.size() - 1; // Index for weights connecting to output layer

    // --- Calculate output layer delta (delta^L) ---
    // delta^L = dLoss/dE_total * dE_total/dE_pair * dE_pair/da^L * da^L/dz^L
    // dLoss/dE_total = molecule_error_raw (passed in); dE_total/dE_pair = 1; dE_pair/da^L = 1
    // da^L/dz^L = activation_derivative(z^L)
    Vector last_z = current_pair_layer_inputs_.back(); // z values for the output layer
    Vector output_activation_deriv(output_size_);
    for(size_t i=0; i < output_size_; ++i) {
        output_activation_deriv[i] = linear_derivative(last_z[i]);
    }

    // Calculate delta for the output layer (L)
    Vector delta(output_size_); // Vector of size 1
    delta[0] = molecule_error_raw * output_activation_deriv[0];


    // Calculate gradients for output layer
    // grad_W^L = delta^L * (a^{L-1})^T  (outer product); grad_b^L = delta^L
    Vector prev_layer_activation = current_pair_layer_outputs_[last_layer_idx - 1]; // a^{L-1}
    Matrix grad_W_output = outer_product(delta, prev_layer_activation);
    Vector grad_b_output = delta;

    // Accumulate gradients for the output layer
    for(size_t r = 0; r < grad_W_output.size(); ++r) {
        for(size_t c = 0; c < grad_W_output[r].size(); ++c) {
            grad_weights_accum[last_weight_idx][r][c] += grad_W_output[r][c];
        }
    }
    for(size_t r = 0; r < grad_b_output.size(); ++r) {
        grad_biases_accum[last_weight_idx][r] += grad_b_output[r];
    }

    // Back prop through hidden layers
    // Iterate from second-to-last layer (l = L-1) down to the first hidden layer (l = 1)
    for (int l = last_layer_idx - 1; l > 0; --l) {
        int weight_idx = l; // Index for W^{l+1} (weights connecting layer l to l+1)
        int grad_accum_idx = l - 1; // Index for grad_W^l and grad_b^l

        // delta^l = ((W^{l+1})^T * delta^{l+1}) * activation_derivative(z^l)
        Matrix W_next_T = transpose(weights_[weight_idx]);
        Vector propagated_error = matrix_vector_multiply(W_next_T, delta); // Sum over next layer's delta contribution

        Vector current_z = current_pair_layer_inputs_[grad_accum_idx]; 
        Vector activation_deriv(current_z.size());
        for(size_t j = 0; j < current_z.size(); ++j) {
            activation_deriv[j] = relu_derivative(current_z[j]);
        }

        delta = elementwise_multiply(propagated_error, activation_deriv);

        // Calculate and accumulate gradients for hidden Layer l
        // grad_W^l = delta^l * (a^{l-1})^T
        // grad_b^l = delta^l
        Vector prev_activation = current_pair_layer_outputs_[l - 1];
        Matrix grad_W_hidden = outer_product(delta, prev_activation);
        Vector grad_b_hidden = delta;

        // Accumulate
        if (grad_accum_idx < 0 || grad_accum_idx >= grad_weights_accum.size()){
             throw std::out_of_range("Gradient accumulator index out of range during backprop.");
        }
        for(size_t r = 0; r < grad_W_hidden.size(); ++r) {
             if (r >= grad_weights_accum[grad_accum_idx].size()){ throw std::out_of_range("Row index out of range for gradient accumulator."); }
            for(size_t c = 0; c < grad_W_hidden[r].size(); ++c) {
                 if (c >= grad_weights_accum[grad_accum_idx][r].size()){ throw std::out_of_range("Col index out of range for gradient accumulator."); }
                grad_weights_accum[grad_accum_idx][r][c] += grad_W_hidden[r][c];
            }
        }
         if (grad_accum_idx >= grad_biases_accum.size()){ throw std::out_of_range("Bias gradient accumulator index out of range."); }
        for(size_t r = 0; r < grad_b_hidden.size(); ++r) {
             if (r >= grad_biases_accum[grad_accum_idx].size()){ throw std::out_of_range("Bias gradient index out of range for accumulator."); }
            grad_biases_accum[grad_accum_idx][r] += grad_b_hidden[r];
        }
    }
}

void MLP::apply_gradients(const std::vector<Matrix>& grad_weights, const std::vector<Vector>& grad_biases) {

    if (grad_weights.size() != weights_.size() || grad_biases.size() != biases_.size()) {
        throw std::logic_error("Gradient dimensions mismatch during parameter update.");
    }

    for (size_t i = 0; i < weights_.size(); ++i) {
        // Update weights: W = W - lr * grad_W
        if (weights_[i].size() != grad_weights[i].size()) throw std::logic_error("Weight gradient row mismatch.");
        for (size_t r = 0; r < weights_[i].size(); ++r) {
            if (weights_[i][r].size() != grad_weights[i][r].size()) throw std::logic_error("Weight gradient column mismatch.");
            for (size_t c = 0; c < weights_[i][r].size(); ++c) {
                weights_[i][r][c] -= learning_rate * grad_weights[i][r][c];
            }
        }
        // Update biases: b = b - lr * grad_b
        if (biases_[i].size() != grad_biases[i].size()) throw std::logic_error("Bias gradient size mismatch.");
        for (size_t r = 0; r < biases_[i].size(); ++r) {
            biases_[i][r] -= learning_rate * grad_biases[i][r];
        }
    }
}

// --- Molecule Level ---

double MLP::train_molecule(const Molecule& molecule, const std::map<std::string, int>& atom_type_to_index,
                          int num_atom_types, double distance_mean, double distance_stddev)
{
    // --- Initialize accumulated gradients ---
    std::vector<Matrix> molecule_grad_weights(weights_.size());
    std::vector<Vector> molecule_grad_biases(biases_.size());
    for(size_t i = 0; i < weights_.size(); ++i) {
        molecule_grad_weights[i] = Matrix(weights_[i].size(), Vector(weights_[i][0].size(), 0.0));
        molecule_grad_biases[i] = Vector(biases_[i].size(), 0.0);
    }

    // --- Forward Pass: Predict total energy and store pair inputs ---
    double predicted_total_energy = 0.0;
    // Store inputs needed for backprop phase (vector of encoded pair inputs)
    std::vector<Vector> pair_inputs_for_backprop;
    pair_inputs_for_backprop.reserve(molecule.atoms.size() * (molecule.atoms.size() -1) / 2);

    for (size_t i = 0; i < molecule.atoms.size(); ++i) {
        for (size_t j = i + 1; j < molecule.atoms.size(); ++j) { // Unique pairs
            const Atom& atom1 = molecule.atoms[i];
            const Atom& atom2 = molecule.atoms[j];

            auto it1 = atom_type_to_index.find(atom1.type);
            auto it2 = atom_type_to_index.find(atom2.type);
            if (it1 == atom_type_to_index.end() || it2 == atom_type_to_index.end()) {
                  std::cerr << "Warning: Skipping pair with unknown atom type in train_molecule ("
                           << atom1.type << ", " << atom2.type << ")" << std::endl;
                  continue;
            }
            int type1_idx = it1->second;
            int type2_idx = it2->second;

            // Calculate and normalize distance
            double distance = calculate_distance(atom1, atom2);
            double normalized_distance = normalize_value(distance, distance_mean, distance_stddev);

            // Encode input for this pair
            Vector pair_input = encode_input(type1_idx, type2_idx, normalized_distance, num_atom_types);
            pair_inputs_for_backprop.push_back(pair_input); // Store for backprop

            // Predict energy for this pair using the method that stores intermediates
            Vector pair_energy_vec = forward_pair(pair_input);
            predicted_total_energy += pair_energy_vec[0];
        }
    }

    // --- Calculate Loss ---
    double true_total_energy = molecule.total_energy;
    double loss = 0.5 * std::pow(predicted_total_energy - true_total_energy, 2); // NOTE: MSE Loss

    // Calculate the derivative of the Loss w.r.t the total predicted energy dLoss/d(predicted_total_energy)
    double molecule_error_raw = predicted_total_energy - true_total_energy;

    // --- Backward Pass: Accumulate gradients from each pair ---
    for (const auto& pair_input : pair_inputs_for_backprop) {

         // Re-run forward_pair to load its intermediate values (z, a)
         forward_pair(pair_input);

         backward_pair(pair_input, molecule_error_raw, molecule_grad_weights, molecule_grad_biases);
     }

    // --- Update Parameters ---
    apply_gradients(molecule_grad_weights, molecule_grad_biases);

    return loss;
}


double MLP::predict_molecule_energy(const Molecule& molecule, const std::map<std::string, int>& atom_type_to_index,
                                   int num_atom_types, double distance_mean, double distance_stddev)
{
    double predicted_total_energy = 0.0;
    for (size_t i = 0; i < molecule.atoms.size(); ++i) {
        for (size_t j = i + 1; j < molecule.atoms.size(); ++j) {
            const Atom& atom1 = molecule.atoms[i];
            const Atom& atom2 = molecule.atoms[j];

            auto it1 = atom_type_to_index.find(atom1.type);
            auto it2 = atom_type_to_index.find(atom2.type);
            if (it1 == atom_type_to_index.end() || it2 == atom_type_to_index.end()) {
                 std::cerr << "Warning: Skipping pair with unknown atom type in predict_molecule_energy ("
                           << atom1.type << ", " << atom2.type << ")" << std::endl;
                 continue;
            }
            int type1_idx = it1->second;
            int type2_idx = it2->second;

            double distance = calculate_distance(atom1, atom2);
            double normalized_distance = normalize_value(distance, distance_mean, distance_stddev);

            Vector pair_input = encode_input(type1_idx, type2_idx, normalized_distance, num_atom_types);

            predicted_total_energy += predict_pair_energy(pair_input);
        }
    }
    return predicted_total_energy;
}

std::vector<int> MLP::get_layer_sizes() const {
    return layer_sizes_;
}