# Machine Learned Conformer Energy Prediction via Approximating Pairwise Potential

## Local Build and Run
**To compile:**
```
make
```
**To train the model:**
```
./mlp_train data/ani_s01-4_sampled_100_train.csv
```
or optionally:
```
make train
```
**To run inference using trained model:**
```
./mlp_predict data/ani_s01-4_sampled_100_test.csv
```
or optionally:
```
make test
```
**Clean:**
```
make clean
```

## Docker Build and Run

Build Image:
```
docker build -t mlp-energy-predictor .
```
Train in Docker:
```
docker run --rm -it \
  -v "$PWD/data":/app/data \
  -v "$PWD/trained_model":/app/trained_model \
  mlp-energy-predictor \
  make train
```
Test in Docker:
```
docker run --rm -it \
  -v "$PWD/data":/app/data \
  -v "$PWD/trained_model":/app/trained_model \
  mlp-energy-predictor \
  make test
```

## File Overview:

**`data/`** : Contains training and testing data CSVs needed to replicate our model training and testing

**`Figs/`** : Holds figures and plots.

**`trained_model/`**: Holds trained model weights.

**`plotting/`**: Contains predictions on the test set using the `trained_model/trained_model_1000_256.txt`, and other plotting functionalities.

**`convert.py`**: Reads an ANI-1 HDF5 file, extracts pairwise informations and write them to a CSV file.

**`main_predict.cpp`** : Loads saved weights from the trained model, runs inference on the test set, and outputs `predictions_test.csv`.

**`main.cpp`** : Loads training data CSV, trains MLP, saves weights, and outputs `predictions_train.csv`.

**`MLP.cpp`** : MLP class implementation.

**`MLP.h`** : PairSample Struct and MLP class layout

### Needed to Run:
- g++
- Make
- python3
- python3-pip
- libhdf5-dev