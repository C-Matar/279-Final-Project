CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall

TRAIN_BIN := mlp_train
PREDICT_BIN := mlp_predict

MLP_SRC := MLP.cpp
MLP_HDR := MLP.h

TRAIN_SRC := main.cpp $(MLP_SRC)
PREDICT_SRC := main_predict.cpp $(MLP_SRC)

WEIGHTS := trained_model/trained_model_1000_256.txt

all: $(TRAIN_BIN) $(PREDICT_BIN)

$(TRAIN_BIN): $(TRAIN_SRC) $(MLP_HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(TRAIN_SRC)

$(PREDICT_BIN): $(PREDICT_SRC) $(MLP_HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(PREDICT_SRC)

train: $(TRAIN_BIN)
	./$(TRAIN_BIN) data/ani_s01-4_sampled_100_train.csv

predict: $(PREDICT_BIN)
	./$(PREDICT_BIN) data/ani_s01-4_sampled_100_test.csv $(WEIGHTS)

clean:
	rm -f *.o $(TRAIN_BIN) $(PREDICT_BIN) predictions.csv trained_model.txt predictions_test.csv predictions_train.csv mlp_energy_predictor 

.PHONY: all train predict clean

docker-train:
	@docker run --rm -it \
		-v "$$PWD/data":/app/data \
		-v "$$PWD/trained_model":/app/trained_model \
		-v "$$PWD":/app/output \
		-w /app \
		mlp-energy-predictor \
		bash -c "./mlp_train data/ani_s01-4_sampled_100_train.csv && cp predictions_train.csv /app/output/"

docker-test:
	@docker run --rm -it \
		-v "$$PWD/data":/app/data \
		-v "$$PWD/trained_model":/app/trained_model \
		-v "$$PWD":/app/output \
		-w /app \
		mlp-energy-predictor \
		bash -c "./mlp_predict data/ani_s01-4_sampled_100_test.csv trained_model/trained_model_1000_256.txt && cp predictions.csv /app/output/"
