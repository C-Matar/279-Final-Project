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