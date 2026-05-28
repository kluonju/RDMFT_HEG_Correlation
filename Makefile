# Simple GNU Makefile for the RDMFT-HEG project.
#
# Usage:
#   make            -- build the main driver and the test binary
#   make run        -- only functionals in FUNCS; incremental (skip existing data/*.tsv).
#   make rerun      -- same flags as run plus --force (overwrite every FUNCS TSV).  Both use the
#                      driver's uniform k mesh on [0, k_max] (Grid::uniform_trapezoid in main.cpp),
#                      default N_GRID=401 and --init-uniform 0.5 (override: make rerun N_GRID=801).
#   make geo        -- (re)compute only the GEO functional into data/GEO.tsv
#   make optgeo     -- (re)compute optGeo (angles from data/log optimization)
#   make plot       -- all figures: correlation_energy, nk, nk_optgeo, nk_hybopt
#   make nk-data     -- export n(k) TSVs under data/nk (same funcs as correlation figure)
#   make plot-nk     -- figures/nk.png (only r_s that exist under data/nk; --rs auto)
#   make plot-nk-optgeo -- figures/nk_optgeo.png (optGeo n(k) overlay only)
#   make plot-nk-hybopt  -- figures/nk_hybopt.png (hybopt n(k) overlay; skips if no nk TSVs)
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
SRCS     := src/main.cpp src/MomentumDistributionGZ.cpp
TARGET   := $(BIN_DIR)/rdmft_heg
TEST_BIN := $(BIN_DIR)/test_hf_exchange
TEST_GZ_BIN := $(BIN_DIR)/test_gz_momentum

HEADERS := $(wildcard include/*.hpp)

DATA_DIR := data
RS_LIST  := 0.2,0.3,0.5,1,2,3,4,5,6,8,10
# Default k-grid size and uniform initial occupation (passed to rdmft_heg).
N_GRID       ?= 401
INIT_UNIFORM ?= 0.5
# Default sweep: only these functionals (edit FUNCS). OptGeo uses ';' — quote for the shell.
# OptGeo: (a;b;c) on unit sphere; weights w1,w2,w3 = a^2,b^2,c^2 = 0.00675,0.64213,0.35112
# HybOpt: HF/Power mix fit vs PW92 on r_s in [0.2, 6] at N=401 (see data/optimize_optGM_rs6.log).
FUNCS := Mueller,CGA,CHF,OptGeo@-0.0821547206643049;0.8013311419635793;0.5925545538598386,Power@0.55,Power@0.58,HybOpt@0.938328;0.541076

NK_DIR := data/nk
NK_FUNCS := $(FUNCS)

.PHONY: all run rerun geo optgeo hybopt plot nk-data plot-nk plot-nk-optgeo plot-nk-hybopt test clean clean-data

all: $(TARGET) $(TEST_BIN) $(TEST_GZ_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRCS) $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

$(TEST_BIN): tests/test_hf_exchange.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_hf_exchange.cpp -o $@

$(TEST_GZ_BIN): tests/test_gz_momentum.cpp src/MomentumDistributionGZ.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_gz_momentum.cpp src/MomentumDistributionGZ.cpp -o $@

# Incremental sweep: skip functionals whose data/<name>.tsv already exists.
# Adding a new functional therefore only runs that functional, leaving the
# previously-computed ones untouched.
run: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR)

# Full rebuild: --force overwrites every functional's TSV.
rerun: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(FUNCS)" \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR) \
		--force

# Convenience target: only refresh GEO (e.g. after tweaking its kernel).
geo: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs GEO \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR) \
		--force

# Angles match ``data/log`` recommended CLI (rounded); change if you re-fit.
optgeo: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "OptGeo@-0.0821547206643049;0.8013311419635793;0.5925545538598386" \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR) \
		--force

# Refit HybOpt (lambda; alpha) only; matches FUNCS HybOpt@… (edit there + here if you re-fit).
hybopt: $(TARGET)
	mkdir -p $(DATA_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs 'HybOpt@0.938328;0.541076' \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR) \
		--force

plot:
	mkdir -p figures
	python3 scripts/plot_results.py --in $(DATA_DIR) \
		--out figures/correlation_energy.png
	python3 scripts/plot_nk.py --dir $(NK_DIR) --rs auto \
		--out figures/nk.png
	python3 scripts/plot_nk_optgeo.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_optgeo.png
	python3 scripts/plot_nk_hybopt.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_hybopt.png

nk-data: $(TARGET)
	mkdir -p $(NK_DIR)
	./$(TARGET) --N $(N_GRID) --kmax 3 \
		--rs $(RS_LIST) \
		--funcs "$(NK_FUNCS)" \
		--init-uniform $(INIT_UNIFORM) \
		--out-dir $(DATA_DIR) --nk-out $(NK_DIR)

plot-nk:
	mkdir -p figures
	python3 scripts/plot_nk.py --dir $(NK_DIR) --rs auto \
		--out figures/nk.png

plot-nk-optgeo:
	mkdir -p figures
	python3 scripts/plot_nk_optgeo.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_optgeo.png

plot-nk-hybopt:
	mkdir -p figures
	python3 scripts/plot_nk_hybopt.py --dir $(NK_DIR) --rs auto \
		--out figures/nk_hybopt.png

test: $(TEST_BIN) $(TEST_GZ_BIN)
	./$(TEST_BIN)
	./$(TEST_GZ_BIN)

clean:
	rm -rf $(BIN_DIR)

clean-data:
	rm -f $(DATA_DIR)/*.tsv
