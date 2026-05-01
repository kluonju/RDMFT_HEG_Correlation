# Simple GNU Makefile for the RDMFT-HEG project.
#
# Usage:
#   make            -- build the main driver and the test binary
#   make run        -- run any functionals whose data/<name>.tsv is missing
#                      (cheap incremental sweep; no recomputation of others)
#   make rerun      -- like run, but force-recompute every default functional
#   make geo        -- (re)compute only the GEO functional into data/GEO.tsv
#   make optgm      -- (re)compute optGM (angles from data/log optimization)
#   make plot       -- regenerate figures/correlation_energy.png from
#                      every *.tsv in data/
#   make test       -- run the HF exchange unit test
#   make clean      -- remove build artifacts
#   make clean-data -- remove every per-functional TSV under data/
#
#   make USE_OPENMP=0   -- build without OpenMP (single-threaded matvecs)

CXX      ?= g++
CXXSTD   ?= -std=c++17
# OpenMP parallelizes O(N^2) exchange matvecs (set USE_OPENMP=0 to disable).
USE_OPENMP ?= 1
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic $(CXXSTD)
ifeq ($(USE_OPENMP),1)
CXXFLAGS += -fopenmp
endif
INCLUDES := -Iinclude

BIN_DIR  := build
SRCS     := src/main.cpp
TARGET   := $(BIN_DIR)/rdmft_heg
TEST_BIN := $(BIN_DIR)/test_hf_exchange

HEADERS := $(wildcard include/*.hpp)

DATA_DIR := data
RS_LIST  := 0.2,0.3,0.5,1,2,3,4,5,6,8,10
# OptGM uses ';' inside the key; --funcs must be quoted for the shell.
FUNCS    := HF,Mueller,GU,CGA,GEO,OptGM@-0.1;0.8;0.6,Power@0.55,Power@0.58,Beta@0.45,Beta@0.55,Beta@0.65

.PHONY: all run rerun geo optgm plot test clean clean-data

all: $(TARGET) $(TEST_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): src/main.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/main.cpp -o $@

$(TEST_BIN): tests/test_hf_exchange.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_hf_exchange.cpp -o $@

# Incremental sweep: skip functionals whose data/<name>.tsv already exists.
# Adding a new functional therefore only runs that functional, leaving the
# previously-computed ones untouched.
run: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 6 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--out-dir $(DATA_DIR)

# Full rebuild: --force overwrites every functional's TSV.
rerun: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 6 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--out-dir $(DATA_DIR) \
		--force

# Convenience target: only refresh GEO (e.g. after tweaking its kernel).
geo: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 6 \
		--rs $(RS_LIST) \
		--funcs GEO \
		--out-dir $(DATA_DIR) \
		--force

# Angles match ``data/log`` recommended CLI (rounded); change if you re-fit.
optgm: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 6 \
		--rs $(RS_LIST) \
		--funcs "OptGM@-0.1;0.8;0.6" \
		--out-dir $(DATA_DIR) \
		--force

plot:
	mkdir -p figures
	python3 scripts/plot_results.py --in $(DATA_DIR) \
		--out figures/correlation_energy.png

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BIN_DIR)

clean-data:
	rm -f $(DATA_DIR)/*.tsv
