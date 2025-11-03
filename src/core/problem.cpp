// src/core/problem.cpp 
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#include "problem.hpp"
#include "planner.hpp"      // make_plan(...)
#include "dispatcher.hpp"   // dispatch_and_run(...)

namespace dphpc {

// helpers reused from your earlier parsing
std::string lc(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}
bool parse_bool(const char* s) {
    std::string v = lc(s ? std::string(s) : "");
    return (v == "1" || v == "true" || v == "on" || v == "yes");
}
bool parse_trans(const char* s) {
    if (!s) return false;
    char c = std::toupper(static_cast<unsigned char>(s[0]));
    if (c == 'T') return true;
    if (c == 'N') return false;
    return parse_bool(s);
}
dphpc::DataType parse_dtype(const char* s) {
    std::string v = lc(s ? std::string(s) : "");
    using dphpc::DataType;
    if (v == "f16"  || v == "fp16"  || v == "half")  return DataType::f16;
    if (v == "bf16" || v == "bfloat16")              return DataType::bf16;
    if (v == "tf32")                                 return DataType::tf32;
    if (v == "f32"  || v == "fp32"  || v == "float") return DataType::f32;
    if (v == "f64"  || v == "fp64"  || v == "double")return DataType::f64;
    if (v == "i8"   || v == "int8")                  return DataType::i8;
    if (v == "i32"  || v == "int32")                 return DataType::i32;
    return DataType::f32;
}

void print_usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " [--verify] M N K transA transB transC typeA typeB typeC typeD wmma cutlass cute\n"
      "  --verify              : run correctness verification instead of benchmark\n"
      "  M N K                 : integers (defaults 5120 5120 4096)\n"
      "  transX                : N/T or 0/1 (defaults N N N)\n"
      "  type{A,B,C,D}         : f16|bf16|tf32|f32|f64|i8|i32 (default f32)\n"
      "  backend flags         : true/false or 1/0 for wmma/cutlass/cute (default 1 1 1)\n";
}

} // namespace

int main(int argc, char** argv)
{
    dphpc::Problem prob; // defaults set in problem.hpp

    // Detect flags first (we treat only --help/--verify as flags for now)
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { dphpc::print_usage(argv[0]); return 0; }
        if (a == "--verify") verify = true;
    }

    // Collect positional args (skip recognized flags)
    std::vector<const char*> pos;
    pos.reserve(argc);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verify" || a == "-h" || a == "--help") continue;
        pos.push_back(argv[i]);
    }

    // Parse positionals in the same order as before
    size_t argi = 0;
    if (argi < pos.size()) prob.M = std::stoll(pos[argi++]); else prob.M = 5120;
    if (argi < pos.size()) prob.N = std::stoll(pos[argi++]); else prob.N = 5120;
    if (argi < pos.size()) prob.K = std::stoll(pos[argi++]); else prob.K = 4096;

    if (argi < pos.size()) prob.transA = dphpc::parse_trans(pos[argi++]);
    if (argi < pos.size()) prob.transB = dphpc::parse_trans(pos[argi++]);
    if (argi < pos.size()) prob.transC = dphpc::parse_trans(pos[argi++]);

    if (argi < pos.size()) prob.typeA = dphpc::parse_dtype(pos[argi++]);
    if (argi < pos.size()) prob.typeB = dphpc::parse_dtype(pos[argi++]);
    if (argi < pos.size()) prob.typeC = dphpc::parse_dtype(pos[argi++]);
    if (argi < pos.size()) prob.typeD = dphpc::parse_dtype(pos[argi++]);

    if (argi < pos.size()) prob.wmma_available    = dphpc::parse_bool(pos[argi++]);
    if (argi < pos.size()) prob.cutlass_available = dphpc::parse_bool(pos[argi++]);
    if (argi < pos.size()) prob.cute_available    = dphpc::parse_bool(pos[argi++]);

    // Query and attach the current device info to the problem
    try {
        prob.device = dphpc::device_query::query(/*device_id*/ -1); // -1 => current device
    } catch (const std::exception& e) {
        std::cerr << "Device query failed: " << e.what() << "\n";
        return 1;
    

    // Show the parsed problem
    std::cout << prob.repr() << "\n";

    try {
        // Get a backend-specific plan
        auto plan = dphpc::make_plan(prob);

        // Hand off to dispatcher to pick backend entry and run bench/verify
        return dphpc::dispatch_and_run(prob, *plan, verify, /*stream=*/0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
}
