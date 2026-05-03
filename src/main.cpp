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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
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
    // Standard sweep (Makefile / nk figures): 401 odd nodes, k_max = 3 k_F;
    // ``Grid::uniform_trapezoid`` on [0, k_max] (composite trapezoid weights).
    int    N_grid = 401;
    double k_max_factor = 3.0;   // k_max = factor * k_F  (default 3 => k/k_F <= 3)
    std::string out_dir = "data";
    std::string nk_out_dir;  // if non-empty, write n(k) TSVs under this directory
    bool   force = false;
    bool   verbose = false;
};

void print_help() {
    std::cout <<
        "Usage: rdmft_heg [options]\n"
        "  --rs <list>          comma-separated rs values\n"
        "  --funcs <list>       **required** comma-separated functionals\n"
        "                       (HF, Mueller, GU, CGA, CHF, BBC1, BBC3, GEO,\n"
        "                        OptGeo@<a>;<b>;<c>, OptGM@lambda;alpha, Power@<alpha>, Beta@<beta>)\n"
        "                       OptGeo: GEO-style three channels; three angles (two ';').\n"
        "                       OptGM: (1-lambda) HF + lambda Power(alpha); two floats (one ';').\n"
        "                       Shell: quote keys that contain ';' (e.g. --funcs 'OptGM@0.4;0.56').\n"
        "  --N <int>            #grid points (odd, default 401)\n"
        "  --kmax <float>       k_max = factor * k_F(r_s) at each r_s (default 3)\n"
        "  --out-dir <dir>      directory for per-functional TSVs (default data)\n"
        "  --nk-out <dir>       write n(k) to <dir>/<func>_rs<rs>.tsv for each solve.\n"
        "                       If the energy TSV already exists, only n(k) is written\n"
        "                       unless --force is given (then both are refreshed).\n"
        "  --force              recompute and overwrite existing TSVs\n"
        "                       (default: skip functionals whose TSV exists)\n"
        "  --verbose            verbose solver logs\n"
        "  --help               show this help and exit\n";
}

std::vector<double> parse_doubles(const std::string& csv) {
    std::vector<double> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    return out;
}
std::vector<std::string> parse_strings(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (s == "--help") { print_help(); std::exit(0); }
        else if (s == "--rs")        a.rs_list = parse_doubles(next());
        else if (s == "--funcs")     a.funcs   = parse_strings(next());
        else if (s == "--N")         a.N_grid  = std::stoi(next());
        else if (s == "--kmax")      a.k_max_factor = std::stod(next());
        else if (s == "--out-dir")   a.out_dir = next();
        else if (s == "--nk-out")    a.nk_out_dir = next();
        else if (s == "--force")     a.force   = true;
        else if (s == "--verbose")   a.verbose = true;
        else { std::cerr << "Unknown arg: " << s << "\n"; print_help(); std::exit(1); }
    }
    return a;
}

std::unique_ptr<Functional> make(const std::string& key) {
    if (key.rfind("Power@", 0) == 0) {
        double alpha = std::stod(key.substr(6));
        return std::make_unique<PowerFunctional>(alpha);
    }
    if (key.rfind("Beta@", 0) == 0) {
        double beta = std::stod(key.substr(5));
        return std::make_unique<BetaFunctional>(beta);
    }
    if (key.rfind("OptGeo@", 0) == 0 || key.rfind("OptGM@", 0) == 0) {
        return make_functional(key);
    }
    return make_functional(key);
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

// filename_for(key) without trailing ".tsv" (for nk export basename).
std::string nk_stem_for(const std::string& key) {
    std::string s = filename_for(key);
    if (s.size() >= 4 && s.compare(s.size() - 4, 4, ".tsv") == 0) {
        s.resize(s.size() - 4);
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.N_grid < 3) {
        std::cerr << "rdmft_heg: --N must be at least 3 for the k grid\n";
        return 1;
    }
    if ((args.N_grid % 2) == 0) {
        std::cerr << "rdmft_heg: warning: --N must be odd; using " << (args.N_grid - 1)
                  << " instead of " << args.N_grid << "\n";
        args.N_grid -= 1;
    }
    if (args.N_grid < 3) {
        std::cerr << "rdmft_heg: after odd correction, --N is too small\n";
        return 1;
    }
    if (args.k_max_factor <= 1.0) {
        std::cerr << "rdmft_heg: --kmax must be > 1 (need k_max > k_F)\n";
        return 1;
    }
    if (args.funcs.empty()) {
        std::cerr << "rdmft_heg: no functionals given; pass --funcs <comma-separated list>\n"
                     "Example: --funcs HF,Mueller,CGA\n";
        return 1;
    }

    std::filesystem::create_directories(args.out_dir);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "rs       functional        Ec(Ha)         Ec_QMC(Ha)\n";
    std::cout << "-----------------------------------------------------\n";

    int n_written = 0;
    int n_skipped = 0;

    for (const std::string& key : args.funcs) {
        auto F = make(key);
        if (!F) { std::cerr << "Unknown functional " << key << "\n"; continue; }

        const std::string out_path =
            args.out_dir + "/" + filename_for(key);
        const bool write_energy = args.force || !file_exists(out_path);
        const bool export_nk    = !args.nk_out_dir.empty();

        if (!write_energy && !export_nk) {
            std::cout << "[skip] " << out_path
                      << " already exists; pass --force to recompute.\n";
            ++n_skipped;
            continue;
        }

        std::ofstream out;
        if (write_energy) {
            out.open(out_path);
            if (!out) {
                std::cerr << "Cannot open " << out_path << "\n";
                continue;
            }
            out << "# rs\tfunctional\tE_per_N(Ha)\tEc_per_N(Ha)\tEc_QMC(Ha)"
                   "\tT/N\tExc/N\tmu\trho_err\tconverged\titers\n";
            out << std::scientific << std::setprecision(10);
        } else {
            std::cout << "[nk-only] " << out_path
                      << " exists; writing n(k) to " << args.nk_out_dir << "\n";
        }

        if (export_nk) {
            std::filesystem::create_directories(args.nk_out_dir);
        }

        // Rebuild the Grid for each rs: uniform k in [0, k_max], k_max = factor * k_F.
        for (double rs : args.rs_list) {
            const double kf    = HEG::kF(rs);
            const double k_max = args.k_max_factor * kf;
            Grid g = Grid::uniform_trapezoid(
                k_max, static_cast<std::size_t>(args.N_grid));
            ExchangeKernel W  = ExchangeKernel::build(g);

            const double E_HF_per_N = HEG::HF_per_electron(rs);
            const double Ec_QMC     = PW92::ec_per_electron(rs);

            SolveOptions opt;
            opt.verbose  = args.verbose;
            opt.tol_n    = 1.0e-8;
            opt.max_iter = 1200;
            opt.mix      = 0.40;

            SolveResult r = solve_rdmft(rs, *F, g, W, opt);

            if (export_nk) {
                char nk_name[512];
                std::snprintf(nk_name, sizeof(nk_name), "%s/%s_rs%.4f.tsv",
                              args.nk_out_dir.c_str(),
                              nk_stem_for(key).c_str(),
                              rs);
                const std::filesystem::path nk_path(nk_name);
                if (!r.converged) {
                    std::error_code ec;
                    std::filesystem::remove(nk_path, ec);
                    std::cerr << "Warning: skip n(k) export (not converged): "
                              << nk_path.string() << "\n";
                } else {
                    std::ofstream nk(nk_name);
                    if (nk) {
                        nk << std::scientific << std::setprecision(16);
                        nk << "# rs=" << rs << " kF=" << kf << " k_max=" << k_max
                           << "\n";
                        nk << "# functional: " << F->name() << "\n";
                        nk << "# converged: 1\n";
                        nk << "# k\tn\n";
                        for (std::size_t i = 0; i < g.n(); ++i) {
                            nk << g.k[i] << '\t' << r.n[i] << "\n";
                        }
                    } else {
                        std::cerr << "Warning: cannot open " << nk_name
                                  << " for nk export\n";
                    }
                }
            }

            const double Ec      = r.E_per_N - E_HF_per_N;
            const double rho_err = std::abs(r.rho - HEG::density(rs));

            if (write_energy) {
                out << rs << '\t' << F->name() << '\t' << r.E_per_N << '\t'
                    << Ec << '\t' << Ec_QMC << '\t'
                    << r.T_per_V / r.rho << '\t'
                    << r.Exc_per_V / r.rho << '\t'
                    << r.mu << '\t'
                    << rho_err << '\t'
                    << (r.converged ? 1 : 0) << '\t'
                    << r.iters << "\n";
                out.flush();
            }

            std::cout << std::setw(7) << rs << "  "
                      << std::setw(16) << std::left << F->name() << std::right
                      << "  " << std::setw(13) << Ec
                      << "   " << std::setw(13) << Ec_QMC
                      << (r.converged ? "" : "  [not converged]")
                      << "\n";
        }

        if (write_energy) {
            std::cout << "Wrote " << out_path << "\n";
            ++n_written;
        } else if (export_nk) {
            std::cout << "Wrote n(k) files for " << F->name()
                      << " under " << args.nk_out_dir << "\n";
            ++n_written;
        }
    }

    std::cout << "\n" << n_written << " functional(s) processed (energy TSV and/or n(k) export), "
              << n_skipped << " skipped (already up to date).\n";
    if (n_skipped > 0) {
        std::cout << "Use --force to overwrite existing per-functional TSVs.\n";
    }
    return 0;
}
