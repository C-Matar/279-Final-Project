#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <map>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <tuple>

// Convenient Definitions
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

double relu(double x);
double relu_derivative(double x);
double linear(double x);
double linear_derivative(double x);

struct Atom {
    std::string type;
    double x, y, z;
};

struct Molecule {
    std::vector<Atom> atoms;
    double total_energy; // Ground truth total energy for this conformer
};

// --- Helper Functions ---
double calculate_distance(const Atom& a1, const Atom& a2);
double normalize_value(double value, double mean, double stddev);

// Vector/Matrix operations
double dot_product(const Vector& a, const Vector& b);
Vector matrix_vector_multiply(const Matrix& M, const Vector& V);
Vector vector_add(const Vector& a, const Vector& b);
Vector vector_subtract(const Vector& a, const Vector& b);
Vector elementwise_multiply(const Vector& a, const Vector& b);
Matrix transpose(const Matrix& M);
Matrix outer_product(const Vector& a, const Vector& b);


// --- MLP Class for Pair Interaction Prediction ---
class MLP {
public:
    // Constructor
    // input_size: Size of the encoded pair input (2 * num_atom_types + 1)
    // hidden_sizes: Vector containing the number of neurons in each hidden layer
    // output_size: 1 (predicting a single scalar energy value per pair)
    MLP(int input_size, const std::vector<int>& hidden_sizes, int output_size, double learning_rate);

    // Encodes atom type indices and normalized distance into the input vector format
    Vector encode_input(int type1_idx, int type2_idx, double normalized_distance, int num_atom_types) const;

    // Trains the network on a single molecule
    // Calculates total predicted energy, computes loss against true total energy, performs backpropagation accumulating gradients from all pairs, and updates weights.
    // Returns the loss value for this molecule.
    double train_molecule(const Molecule& molecule,
                          const std::map<std::string, int>& atom_type_to_index, // Map atom type string to index
                          int num_atom_types,
                          double distance_mean, // Mean distance for normalization
                          double distance_stddev); // Std dev distance for normalization

    // Predicts the total energy for a given molecule by summing pair predictions.
    double predict_molecule_energy(const Molecule& molecule,
                                   const std::map<std::string, int>& atom_type_to_index,
                                   int num_atom_types,
                                   double distance_mean,
                                   double distance_stddev);

    // Get the sizes of all layers (input, hidden(s), output)
    std::vector<int> get_layer_sizes() const;

    double learning_rate;

private:
    int input_size_;
    std::vector<int> hidden_sizes_;
    int output_size_; // Always 1
    std::vector<int> layer_sizes_;

    // --- Parameters (Learned) ---
    std::vector<Matrix> weights_; // weights_[i] connects layer i to i+1
    std::vector<Vector> biases_;  // biases_[i] is for layer i+1

    // Intermediate values stored during a pair's forward pass
    std::vector<Vector> current_pair_layer_outputs_; // Outputs after activation (a)
    std::vector<Vector> current_pair_layer_inputs_;  // Outputs before activation (z)

    // Random number generator for weight initialization
    std::mt19937 rng_;

    // Performs forward pass for one atom pair and stores intermediate z and a values
    // Returns the predicted pair energy as a scalar.
    Vector forward_pair(const Vector& pair_input);

    // Performs forward pass for one atom pair without storing intermediates. Used for inference. Returns the scalar pair energy.
    double predict_pair_energy(const Vector& pair_input);

    // Performs backpropagation for ONE pair's contribution to the total molecule error.
    void backward_pair(const Vector& pair_input, double molecule_error_signal, // dLoss/dE_total
                       std::vector<Matrix>& grad_weights_accum, // Accumulator for weight gradients (passed by ref)
                       std::vector<Vector>& grad_biases_accum // Accumulator for bias gradients (passed by ref)
                      );

    // Updates the network's weights and biases using the accumulated gradients.
    void apply_gradients(const std::vector<Matrix>& grad_weights,
                         const std::vector<Vector>& grad_biases);

    // Initializes weights and biases
    void initialize_weights();
};


#endif