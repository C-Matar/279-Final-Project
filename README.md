# Machine Learned Conformer Energy Prediction via Approximating Pairwise Potential

## Running
To compile:
```
make
```
To train the model:
```
./mlp_train data/ani_s01-4_sampled_100_train.csv
```
To run inference using trained model (Note: weights loading filename is hardcoded and needs changing):
```
./mlp_predict data/ani_s01-4_sampled_100_test.csv
```

## Update -----------------------------------------

### File Overview:

**`data`** : Contains generated CSVs needed to replicate our model training and testing

**`Figs`** : Holds figures and plots

**`trained_model/`**: Storage for `mlp_train` weights

**`convert.py`**: Convert ANI-1 HDF5 to a pairwise CSV file

**`main_predict.cpp`** : Builds MLP architecture, loads saved weights from the trained model, runs forward passes to predict energy, outputs predictions.csv

**`main.cpp`** : Loads CSV, calculates normalization, trains MLP, saveds weights, outputs predictions.csv for training

**`MLP.cpp`** : MLP class implemented

**`MLP.h`** : PairSample Struct and MLP class layout



### Needed to Run:
- g++
- Make
- python3
- python3-pip
- libhdf5-dev

### Can also use Dockerfile to run

### Local Build and Run

Build C++ Binary Files (mlp_train and mlp_predict):
```
make
```
Train using Train CSV File under Data Folder:
```
make train
```
Test on Test CSV under Data Folder:
```
make test
```
Clean:
```
make clean
```

### Docker Build and Run

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
