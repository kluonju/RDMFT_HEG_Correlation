// RDMFT for the homogeneous electron gas.
//
// Driver: sweeps a list of rs values, runs each requested functional, and
// writes one TSV file per functional with the correlation energy per
// electron (E_total - E_HF) in Hartree.  The PW92 parameterization is used
// as the QMC reference.
//
// Per-functional files (e.g. data/HF.tsv, data/GEO.tsv) make it cheap to add
// a new functional without re-running every other one: existing files are
// left untouched unless --force is given.  Only functionals passed in --funcs
// are run; there is no built-in default list.

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <vector>

#include "Energy.hpp"
#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"
#include "QMC.hpp"
#include "Solver.hpp"

using namespace rdmft;

namespace {

struct Args {
    std::vector<double> rs_list = {
        0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0
    };
    std::vector<std::string> funcs;  // must be set via --funcs (no silent default list)
    // Standard sweep (Makefile / nk figures): 401 odd nodes, k_max = 3 k_F.
    // grid_type: uniform | log | graded | logfermi  (see Grid.hpp).
    int    N_grid = 401;
    double k_max_factor = 3.0;   // k_max = factor * k_F  (default 3 => k/k_F <= 3)
    std::string grid_type = "uniform";
    std::string out_dir = "data";
    std::string nk_out_dir;  // if non-empty, write n(k) TSVs under this directory
    bool   force = false;
    bool   verbose = false;
    bool   show_help = false;
    // If set, solver seeds n(k) uniformly (clamped to [0,1]); single-start only.
    std::optional<double> init_uniform_n;
};

void print_help() {
    std::cout <<
        "Usage: rdmft_heg [options]\n"
        "  --rs <list>          comma-separated rs values\n"
        "  --funcs <list>       **required** comma-separated functionals\n"
        "                       (HF, Mueller, GU, CGA, CHF, BBC1, BBC3, GEO,\n"
        "                        OptGeo@<w>;<k>, HybOpt@lambda;alpha, Power@<alpha>,\n"
        "                        Beta@<beta>, BOW or BOW@alpha, SymBow or SymBow@alpha,\n"
        "                        NN@<path/to/model.json>)\n"
        "                       OptGeo: HF + w*(2*sigmoid(k*x)-1), x=(n_i-1/2)^2(n_j-1/2)^2; w,k >= 0.\n"
        "                       HybOpt: (1-lambda) HF + lambda Power(alpha); two floats (one ';').\n"
        "                       Shell: quote keys that contain ';' (e.g. --funcs 'HybOpt@0.4;0.56').\n"
        "                       OptGM@... is accepted as a legacy alias for HybOpt@....\n"
        "  --N <int>            #grid points (odd, default 401)\n"
        "  --kmax <float>       k_max = factor * k_F(r_s) at each r_s (default 3)\n"
        "  --grid <type>        k-mesh: uniform (default), log (dense near k=0),\n"
        "                       graded (dense near k_F), logfermi (log + Fermi band)\n"
        "  --out-dir <dir>      directory for per-functional TSVs (default data)\n"
        "  --nk-out <dir>       write n(k) to <dir>/<func>_rs<rs>.tsv for each solve.\n"
        "                       If the energy TSV already exists, only n(k) is written\n"
        "                       unless --force is given (then both are refreshed).\n"
        "  --force              recompute and overwrite existing TSVs\n"
        "                       (default: skip functionals whose TSV exists)\n"
        "  --init-uniform <n>   initial occupation n(k)=<n> on the grid (clamped to [0,1]);\n"
        "                       uses one multistart trial instead of step/smeared seeds\n"
        "  --verbose            verbose solver logs\n"
        "  --help               show this help and exit\n";
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::vector<std::string> split_csv(const std::string& csv, const char* option) {
    if (csv.empty()) {
        throw std::invalid_argument(std::string(option) + " requires a non-empty list");
    }
    std::vector<std::string> items;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t comma = csv.find(',', begin);
        const std::size_t end = (comma == std::string::npos) ? csv.size() : comma;
        std::string item = trim(csv.substr(begin, end - begin));
        if (item.empty()) {
            throw std::invalid_argument(
                std::string(option) + " contains an empty list item");
        }
        items.push_back(std::move(item));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return items;
}

double parse_finite_double(const std::string& text, const char* option) {
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a number");
    }
    if (parsed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(option) + " expects a finite number: " + text);
    }
    return value;
}

int parse_int(const std::string& text, const char* option) {
    std::size_t parsed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects an integer");
    }
    if (parsed != text.size()
        || value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(option) + " integer is out of range");
    }
    return static_cast<int>(value);
}

std::vector<double> parse_doubles(const std::string& csv, const char* option) {
    std::vector<double> values;
    for (const std::string& item : split_csv(csv, option)) {
        values.push_back(parse_finite_double(item, option));
    }
    return values;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc
                || std::string(argv[i + 1]).rfind("--", 0) == 0) {
                throw std::invalid_argument("missing value for " + s);
            }
            return std::string(argv[++i]);
        };
        if      (s == "--help")      a.show_help = true;
        else if (s == "--rs")        a.rs_list = parse_doubles(next(), "--rs");
        else if (s == "--funcs")     a.funcs   = split_csv(next(), "--funcs");
        else if (s == "--N")         a.N_grid  = parse_int(next(), "--N");
        else if (s == "--kmax")      a.k_max_factor =
                                          parse_finite_double(next(), "--kmax");
        else if (s == "--grid")      a.grid_type = next();
        else if (s == "--out-dir")   a.out_dir = next();
        else if (s == "--nk-out")    a.nk_out_dir = next();
        else if (s == "--force")     a.force   = true;
        else if (s == "--init-uniform") {
            a.init_uniform_n = parse_finite_double(next(), "--init-uniform");
        }
        else if (s == "--verbose")   a.verbose = true;
        else throw std::invalid_argument("unknown argument: " + s);
    }
    return a;
}

void validate_args(const Args& args) {
    if (args.show_help) return;
    if (args.N_grid < 3 || (args.N_grid % 2) == 0) {
        throw std::invalid_argument("--N must be an odd integer of at least 3");
    }
    if (!std::isfinite(args.k_max_factor) || args.k_max_factor <= 1.0) {
        throw std::invalid_argument("--kmax must be finite and greater than 1");
    }
    if (args.grid_type != "uniform" && args.grid_type != "log"
        && args.grid_type != "graded" && args.grid_type != "logfermi") {
        throw std::invalid_argument(
            "--grid must be 'uniform', 'log', 'graded', or 'logfermi'");
    }
    if (args.rs_list.empty()) {
        throw std::invalid_argument("--rs requires at least one value");
    }
    for (double rs : args.rs_list) {
        if (!std::isfinite(rs) || rs <= 0.0) {
            throw std::invalid_argument("--rs values must be finite and positive");
        }
    }
    if (args.funcs.empty()) {
        throw std::invalid_argument(
            "no functionals given; pass --funcs <comma-separated list>");
    }
    if (trim(args.out_dir).empty()) {
        throw std::invalid_argument("--out-dir must not be empty");
    }
    if (!args.nk_out_dir.empty() && trim(args.nk_out_dir).empty()) {
        throw std::invalid_argument("--nk-out must not be blank");
    }
    if (args.init_uniform_n
        && (*args.init_uniform_n < 0.0 || *args.init_uniform_n > 1.0)) {
        throw std::invalid_argument("--init-uniform must be in [0,1]");
    }
}

// Convert a functional CLI key (e.g. "Power@0.55", "Beta@0.450") into a
// filesystem-friendly stem.  We replace '@' with '_' and avoid dots; the
// plot script keys back off the `functional` column inside each TSV, so the
// filename only needs to be stable and unique.
std::string filename_for(const std::string& key) {
    std::string s = key;
    for (char& c : s) {
        if (c == '@')       c = '_';
        else if (c == ';')  c = '_';
        else if (c == '/') c = '_';
        else if (c == ' ') c = '_';
    }
    return s + ".tsv";
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

class AtomicOutput {
public:
    explicit AtomicOutput(std::filesystem::path final_path)
        : final_path_(std::move(final_path)) {
        static std::size_t counter = 0;
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch().count();
        temp_path_ = final_path_;
        temp_path_ += ".tmp." + std::to_string(stamp)
                    + "." + std::to_string(counter++);
        out_.open(temp_path_, std::ios::out | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error(
                "cannot open temporary output " + temp_path_.string());
        }
    }

    AtomicOutput(const AtomicOutput&) = delete;
    AtomicOutput& operator=(const AtomicOutput&) = delete;

    ~AtomicOutput() {
        if (!committed_) {
            out_.close();
            std::error_code ignored;
            std::filesystem::remove(temp_path_, ignored);
        }
    }

    std::ofstream& stream() { return out_; }

    void commit() {
        out_.flush();
        if (!out_) {
            throw std::runtime_error(
                "failed writing temporary output " + temp_path_.string());
        }
        out_.close();
        if (out_.fail()) {
            throw std::runtime_error(
                "failed closing temporary output " + temp_path_.string());
        }
        std::error_code error;
        std::filesystem::rename(temp_path_, final_path_, error);
        if (error) {
            throw std::runtime_error(
                "cannot publish " + final_path_.string() + ": " + error.message());
        }
        committed_ = true;
    }

private:
    std::filesystem::path final_path_;
    std::filesystem::path temp_path_;
    std::ofstream out_;
    bool committed_ = false;
};

// filename_for(key) without trailing ".tsv" (for nk export basename).
std::string nk_stem_for(const std::string& key) {
    std::string s = filename_for(key);
    if (s.size() >= 4 && s.compare(s.size() - 4, 4, ".tsv") == 0) {
        s.resize(s.size() - 4);
    }
    return s;
}

}  // namespace

int run(const Args& args) {
    validate_args(args);
    if (args.show_help) {
        print_help();
        return 0;
    }

    std::filesystem::create_directories(args.out_dir);
    if (!args.nk_out_dir.empty()) {
        std::filesystem::create_directories(args.nk_out_dir);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "rs       functional        Ec(Ha)         Ec_QMC(Ha)\n";
    std::cout << "-----------------------------------------------------\n";

    int n_written = 0;
    int n_skipped = 0;
    bool had_errors = false;

    for (const std::string& key : args.funcs) {
        try {
            auto F = make_functional(key);
            if (!F) {
                throw std::invalid_argument("unknown functional: " + key);
            }

            const std::filesystem::path out_path =
                std::filesystem::path(args.out_dir) / filename_for(key);
            const bool write_energy =
                args.force || !file_exists(out_path.string());
            const bool export_nk = !args.nk_out_dir.empty();

            if (!write_energy && !export_nk) {
                std::cout << "[skip] " << out_path.string()
                          << " already exists; pass --force to recompute.\n";
                ++n_skipped;
                continue;
            }

            std::unique_ptr<AtomicOutput> energy_output;
            if (write_energy) {
                energy_output = std::make_unique<AtomicOutput>(out_path);
                auto& out = energy_output->stream();
                out << "# rs\tfunctional\tE_per_N(Ha)\tEc_per_N(Ha)\tEc_QMC(Ha)"
                       "\tT/N\tExc/N\tmu\trho_err\tconverged\titers\n";
                out << std::scientific << std::setprecision(10);
            } else {
                std::cout << "[nk-only] " << out_path.string()
                          << " exists; writing n(k) to " << args.nk_out_dir << "\n";
            }

            // Rebuild the Grid for each rs; k_max = factor * k_F.
            for (double rs : args.rs_list) {
                const double kf = HEG::kF(rs);
                const double k_max = args.k_max_factor * kf;
                const std::size_t N = static_cast<std::size_t>(args.N_grid);
                Grid g;
                if (args.grid_type == "log") {
                    g = Grid::log_trapezoid(kf, k_max, N);
                } else if (args.grid_type == "graded") {
                    g = Grid::graded_fermi_trapezoid(kf, k_max, N);
                } else if (args.grid_type == "logfermi") {
                    g = Grid::log_fermi_trapezoid(kf, k_max, N);
                } else {
                    g = Grid::uniform_trapezoid(k_max, N);
                }
                ExchangeKernel W = ExchangeKernel::build(g);

                const double E_HF_per_N = HEG::HF_per_electron(rs);
                const double Ec_QMC = PW92::ec_per_electron(rs);

                SolveOptions opt;
                opt.verbose = args.verbose;
                opt.tol_n = 1.0e-8;
                opt.max_iter = 1200;
                opt.mix = 0.40;
                opt.init_uniform_n = args.init_uniform_n;

                SolveResult r = solve_rdmft(rs, *F, g, W, opt);
                bool valid_result =
                    r.n.size() == g.n() && std::isfinite(r.mu)
                    && std::isfinite(r.rho) && r.rho > 0.0
                    && std::isfinite(r.T_per_V)
                    && std::isfinite(r.Exc_per_V)
                    && std::isfinite(r.E_per_N);
                for (double occupation : r.n) {
                    valid_result = valid_result && std::isfinite(occupation)
                        && occupation >= 0.0 && occupation <= 1.0;
                }
                if (!valid_result) {
                    throw std::runtime_error(
                        "solver produced no finite result for " + F->name()
                        + " at r_s=" + std::to_string(rs)
                        + (r.failure_reason.empty()
                               ? std::string()
                               : ": " + r.failure_reason));
                }

                if (export_nk) {
                    std::ostringstream nk_filename;
                    nk_filename << nk_stem_for(key) << "_rs"
                                << std::fixed << std::setprecision(4) << rs
                                << ".tsv";
                    const std::filesystem::path nk_path =
                        std::filesystem::path(args.nk_out_dir) / nk_filename.str();
                    AtomicOutput nk_output(nk_path);
                    auto& nk = nk_output.stream();
                    // Always export n(k), even when the SCF did not fully
                    // converge.  Downstream tooling (optimize_nn_gz.py) needs
                    // a finite RMSE signal on bad weight vectors, so the
                    // ``# converged:`` flag is reported instead of deleting
                    // the file.  The energy TSV still flags (r.converged?1:0).
                    const double rho_target = HEG::density(rs);
                    const double rho_err = std::abs(r.rho - rho_target);
                    nk << std::scientific << std::setprecision(16);
                    nk << "# rs=" << rs << " kF=" << kf << " k_max=" << k_max
                       << "\n";
                    nk << "# functional: " << F->name() << "\n";
                    nk << "# converged: " << (r.converged ? 1 : 0) << "\n";
                    if (!r.failure_reason.empty()) {
                        nk << "# failure_reason: " << r.failure_reason << "\n";
                    }
                    nk << "# rho_target=" << rho_target
                       << " rho_actual=" << r.rho
                       << " rho_err=" << rho_err << "\n";
                    nk << "# iters=" << r.iters << " mu=" << r.mu << "\n";
                    nk << "# k\tn\n";
                    for (std::size_t i = 0; i < g.n(); ++i) {
                        nk << g.k[i] << '\t' << r.n[i] << "\n";
                    }
                    nk_output.commit();
                    if (!r.converged) {
                        std::cerr << "Warning: SCF did not converge for "
                                  << F->name() << " r_s=" << rs
                                  << " (rho_err=" << rho_err
                                  << ", reason=" << r.failure_reason
                                  << "); n(k) exported anyway.\n";
                    }
                }

                const double Ec = r.E_per_N - E_HF_per_N;
                const double rho_err = std::abs(r.rho - HEG::density(rs));

                if (write_energy) {
                    auto& out = energy_output->stream();
                    out << rs << '\t' << F->name() << '\t' << r.E_per_N << '\t'
                        << Ec << '\t' << Ec_QMC << '\t'
                        << r.T_per_V / r.rho << '\t'
                        << r.Exc_per_V / r.rho << '\t'
                        << r.mu << '\t'
                        << rho_err << '\t'
                        << (r.converged ? 1 : 0) << '\t'
                        << r.iters << "\n";
                }

                std::cout << std::setw(7) << rs << "  "
                          << std::setw(16) << std::left << F->name() << std::right
                          << "  " << std::setw(13) << Ec
                          << "   " << std::setw(13) << Ec_QMC
                          << (r.converged ? "" : "  [not converged]")
                          << "\n";
            }

            if (write_energy) {
                energy_output->commit();
                std::cout << "Wrote " << out_path.string() << "\n";
                ++n_written;
            } else if (export_nk) {
                std::cout << "Wrote n(k) files for " << F->name()
                          << " under " << args.nk_out_dir << "\n";
                ++n_written;
            }
        } catch (const std::exception& error) {
            std::cerr << "rdmft_heg: failed to process " << key
                      << ": " << error.what() << "\n";
            had_errors = true;
        }
    }

    std::cout << "\n" << n_written << " functional(s) processed (energy TSV and/or n(k) export), "
              << n_skipped << " skipped (already up to date).\n";
    if (n_skipped > 0) {
        std::cout << "Use --force to overwrite existing per-functional TSVs.\n";
    }
    return had_errors ? 1 : 0;
}

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "rdmft_heg: " << error.what() << "\n";
        return 2;
    }
}
