# Simple GNU Makefile for the RDMFT-HEG project.
#
# Usage:
#   make            -- build the main driver and the test binary
#   make run        -- run the default rs/functional sweep
#   make plot       -- regenerate figures/correlation_energy.png
#   make test       -- run the HF exchange unit test
#   make clean      -- remove build artifacts

CXX      ?= g++
CXXSTD   ?= -std=c++17
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic $(CXXSTD)
INCLUDES := -Iinclude

BIN_DIR  := build
SRCS     := src/main.cpp
TARGET   := $(BIN_DIR)/rdmft_heg
TEST_BIN := $(BIN_DIR)/test_hf_exchange

HEADERS := $(wildcard include/*.hpp)

.PHONY: all run plot test clean

all: $(TARGET) $(TEST_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): src/main.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/main.cpp -o $@

$(TEST_BIN): tests/test_hf_exchange.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_hf_exchange.cpp -o $@

run: $(TARGET)
	mkdir -p data
	./$(TARGET) --N 401 --kmax 6 \
		--rs 0.2,0.3,0.5,1,2,3,4,5,6,8,10 \
		--funcs HF,Mueller,GU,CGA,Power@0.55,Power@0.58,Beta@0.45,Beta@0.55,Beta@0.65 \
		--out data/results.tsv

plot:
	mkdir -p figures
	python3 scripts/plot_results.py

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BIN_DIR)
