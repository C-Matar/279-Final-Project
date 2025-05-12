CXX       := g++
CXXFLAGS  := -std=c++17 -O2 -Wall

TRAIN_TARGET   := mlp_train
TRAIN_SRCS     := main.cpp MLP.cpp
TRAIN_OBJS     := $(TRAIN_SRCS:.cpp=.o)

PREDICT_TARGET := mlp_predict
PREDICT_SRCS   := main_predict.cpp MLP.cpp
PREDICT_OBJS   := $(PREDICT_SRCS:.cpp=.o)

# CSV files 
TRAIN_CSV      := data/ani_s01-4_sampled_100_train.csv
TEST_CSV       := data/ani_s01-4_sampled_100_test.csv

.PHONY: all train test clean

all: $(TRAIN_TARGET) $(PREDICT_TARGET)

# build binaries
$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(PREDICT_TARGET): $(PREDICT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# compile .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# run training
train: $(TRAIN_TARGET)
	@echo "→ Running training on $(TRAIN_CSV)"
	./$(TRAIN_TARGET) $(TRAIN_CSV)

# run test
test: $(PREDICT_TARGET)
	@echo "→ Running prediction on $(TEST_CSV)"
	./$(PREDICT_TARGET) $(TEST_CSV)

clean:
	rm -f $(TRAIN_OBJS) $(PREDICT_OBJS) $(TRAIN_TARGET) $(PREDICT_TARGET)
