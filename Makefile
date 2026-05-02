# Simple GNU Makefile for the RDMFT-HEG project.
#
# Usage:
#   make            -- build the main driver and the test binary
#   make run / rerun -- only functionals in FUNCS (Makefile variable; not the whole codebase list)
#   make geo        -- (re)compute only the GEO functional into data/GEO.tsv
#   make optgm      -- (re)compute optGM (angles from data/log optimization)
#   make plot       -- correlation_energy.png, nk.png, and nk_optgm.png (optGM n(k) vs r_s)
#   make nk-data     -- export n(k) TSVs under data/nk (same funcs as correlation figure)
#   make plot-nk     -- figures/nk.png (only r_s that exist under data/nk; --rs auto)
#   make plot-nk-optgm -- figures/nk_optgm.png (optGM n(k) overlay only)
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
# Default sweep: only these functionals (edit FUNCS). OptGM uses ';' — quote for the shell.
# OptGM: (a;b;c) on unit sphere; weights w1,w2,w3 = a^2,b^2,c^2 = 0.00675,0.64213,0.35112
FUNCS := Mueller,CGA,CHF,BBC3,OptGM@-0.0821547206643049;0.8013311419635793;0.5925545538598386,Power@0.55,Power@0.58

NK_DIR := data/nk
NK_FUNCS := $(FUNCS)

.PHONY: all run rerun geo optgm plot nk-data plot-nk plot-nk-optgm test clean clean-data

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
	./$(TARGET) --N 401 --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--out-dir $(DATA_DIR)

# Full rebuild: --force overwrites every functional's TSV.
rerun: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--out-dir $(DATA_DIR) \
		--force

# Convenience target: only refresh GEO (e.g. after tweaking its kernel).
geo: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 3 \
		--rs $(RS_LIST) \
		--funcs GEO \
		--out-dir $(DATA_DIR) \
		--force

# Angles match ``data/log`` recommended CLI (rounded); change if you re-fit.
optgm: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N 401 --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "OptGM@-0.0821547206643049;0.8013311419635793;0.5925545538598386" \
		--out-dir $(DATA_DIR) \
		--force

plot:
	mkdir -p figures
	python3 scripts/plot_results.py --in $(DATA_DIR) \
		--out figures/correlation_energy.png
	python3 scripts/plot_nk.py --dir $(NK_DIR) --rs auto \
		--out figures/nk.png
	python3 scripts/plot_nk_optgm.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_optgm.png

nk-data: $(TARGET)
	mkdir -p $(NK_DIR)
	./$(TARGET) --N 401 --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(NK_FUNCS)" \
		--out-dir $(DATA_DIR) --nk-out $(NK_DIR)

plot-nk:
	mkdir -p figures
	python3 scripts/plot_nk.py --dir $(NK_DIR) --rs auto \
		--out figures/nk.png

plot-nk-optgm:
	mkdir -p figures
	python3 scripts/plot_nk_optgm.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_optgm.png

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BIN_DIR)

clean-data:
	rm -f $(DATA_DIR)/*.tsv
