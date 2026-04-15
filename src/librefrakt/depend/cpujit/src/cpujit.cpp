#include "cpujit.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>

struct cpujit_module {
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::string asm_text;
};

struct sleef_func {
    const char* scalar_name;
    const char* intrinsic_name;
    const char* sleef_base;
    int nargs;
};

static constexpr sleef_func g_sleef_funcs[] = {
    {"sinf",      "llvm.sin.f32",      "sinf",      1},
    {"cosf",      "llvm.cos.f32",      "cosf",      1},
    {"tanf",      "llvm.tan.f32",      "tanf",      1},
    {"asinf",     "llvm.asin.f32",     "asinf",     1},
    {"acosf",     "llvm.acos.f32",     "acosf",     1},
    {"atanf",     "llvm.atan.f32",     "atanf",     1},
    {"sinhf",     "llvm.sinh.f32",     "sinhf",     1},
    {"coshf",     "llvm.cosh.f32",     "coshf",     1},
    {"tanhf",     "llvm.tanh.f32",     "tanhf",     1},
    {"asinhf",    nullptr,             "asinhf",    1},
    {"acoshf",    nullptr,             "acoshf",    1},
    {"atanhf",    nullptr,             "atanhf",    1},
    {"expf",      "llvm.exp.f32",      "expf",      1},
    {"exp2f",     "llvm.exp2.f32",     "exp2f",     1},
    {"exp10f",    "llvm.exp10.f32",    "exp10f",    1},
    {"expm1f",    nullptr,             "expm1f",    1},
    {"logf",      "llvm.log.f32",      "logf",      1},
    {"log2f",     "llvm.log2.f32",     "log2f",     1},
    {"log10f",    "llvm.log10.f32",    "log10f",    1},
    {"log1pf",    nullptr,             "log1pf",    1},
    {"cbrtf",     nullptr,             "cbrtf",     1},
    {"sqrtf",     "llvm.sqrt.f32",     "sqrtf",     1},
    {"erff",      nullptr,             "erff",      1},
    {"sinpif",    nullptr,             "sinpif",    1},
    {"cospif",    nullptr,             "cospif",    1},
    {"atan2f",    "llvm.atan2.f32",    "atan2f",    2},
    {"powf",      "llvm.pow.f32",      "powf",      2},
    {"hypotf",    nullptr,             "hypotf",    2},
    {"fmodf",     nullptr,             "fmodf",     2},
    {"copysignf", "llvm.copysign.f32", "copysignf", 2},
};

struct sleef_registry {
    std::vector<std::string> owned_strings;
    std::vector<llvm::VecDesc> mappings;
    std::string preamble;

    sleef_registry() {
        struct width_info {
            int width;
            const char* isa;
        };

        static constexpr width_info float_widths[] = {
            {4,  "avx2128"},
            {8,  "avx2"},
            {16, "avx512f"},
        };

        constexpr auto num_funcs = std::size(g_sleef_funcs);
        constexpr auto num_widths = std::size(float_widths);
        owned_strings.reserve(num_funcs * num_widths * 2);
        mappings.reserve(num_funcs * num_widths * 2);

        for (const auto& func : g_sleef_funcs) {
            std::string vabi_suffix(func.nargs, 'v');

            for (const auto& w : float_widths) {
                auto sleef_name = "Sleef_" + std::string(func.sleef_base)
                    + std::to_string(w.width) + "_u10" + w.isa;
                auto vabi = "_ZGV_LLVM_N" + std::to_string(w.width) + vabi_suffix;

                owned_strings.push_back(sleef_name);
                owned_strings.push_back(vabi);

                auto& sn = owned_strings[owned_strings.size() - 2];
                auto& vb = owned_strings[owned_strings.size() - 1];

                mappings.push_back({func.scalar_name, sn,
                    llvm::ElementCount::getFixed(w.width), false, vb, std::nullopt});

                if (func.intrinsic_name) {
                    mappings.push_back({func.intrinsic_name, sn,
                        llvm::ElementCount::getFixed(w.width), false, vb, std::nullopt});
                }

                std::cout << "Mapping " << func.scalar_name << " to " << sn << " with ABI " << vb << std::endl;
            }
        }

        std::string result;
        for (const auto& func : g_sleef_funcs) {
            result += "declare float @";
            result += func.scalar_name;
            result += "(";
            for (int i = 0; i < func.nargs; ++i) {
                if (i > 0) result += ", ";
                result += "float";
            }
            result += ") nounwind willreturn memory(none)\n";
        }
        preamble = std::move(result);
    }
};

static const sleef_registry& get_sleef_registry() {
    static sleef_registry instance;
    return instance;
}

static std::once_flag g_init_flag;

static void ensure_initialized() {
    std::call_once(g_init_flag, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    });
}

static char* make_error(const std::string& msg) {
    auto* buf = new char[msg.size() + 1];
    std::memcpy(buf, msg.c_str(), msg.size() + 1);
    return buf;
}

static std::unique_ptr<llvm::TargetMachine> create_host_target_machine() {
    auto triple = llvm::sys::getDefaultTargetTriple();
    std::string err;
    auto* target = llvm::TargetRegistry::lookupTarget(triple, err);
    if (!target) return nullptr;

    auto cpu = std::string(llvm::sys::getHostCPUName());
    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> host_features = llvm::sys::getHostCPUFeatures();
    for (auto& [key, val] : host_features)
        features.AddFeature(key, val);

    return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        llvm::Triple(triple), cpu, features.getString(), llvm::TargetOptions{}, std::nullopt));
}

static void optimize(llvm::Module& mod, llvm::TargetMachine& tm) {
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    mod.setDataLayout(tm.createDataLayout());

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::TargetLibraryInfoImpl tlii(triple);

    auto& registry = get_sleef_registry();
    tlii.addVectorizableFunctions(registry.mappings);

    fam.registerPass([&] { return llvm::TargetLibraryAnalysis(tlii); });

    llvm::PassBuilder pb(&tm);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    auto mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(mod, mam);
}

static std::string emit_asm(llvm::Module& mod, llvm::TargetMachine& tm) {
    auto clone = llvm::CloneModule(mod);
    clone->setDataLayout(tm.createDataLayout());

    llvm::SmallVector<char, 0> asm_buf;
    llvm::raw_svector_ostream os(asm_buf);
    llvm::legacy::PassManager pm;
    if (tm.addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::AssemblyFile))
        return {};

    pm.run(*clone);
    return std::string(asm_buf.begin(), asm_buf.end());
}

extern "C" {

CPUJIT_API cpujit_module_t cpujit_compile(const char* llvm_ir, const char** error_out) {
    if (error_out) *error_out = nullptr;

    ensure_initialized();

    // Create the JIT instance
    auto jit_or_err = llvm::orc::LLJITBuilder().create();
    if (!jit_or_err) {
        if (error_out) {
            std::string msg;
            llvm::raw_string_ostream os(msg);
            os << jit_or_err.takeError();
            *error_out = make_error(msg);
        } else {
            llvm::consumeError(jit_or_err.takeError());
        }
        return nullptr;
    }
    auto jit = std::move(*jit_or_err);

    // Register SLEEF for runtime symbol resolution
    auto prefix = jit->getDataLayout().getGlobalPrefix();
    if (auto sleef_gen = llvm::orc::DynamicLibrarySearchGenerator::Load("sleef.dll", prefix)) {
        jit->getMainJITDylib().addGenerator(std::move(*sleef_gen));
    } else {
        std::cerr << "Failed to register SLEEF for runtime symbol resolution" << std::endl;
        std::cerr << llvm::toString(sleef_gen.takeError()) << std::endl;
    }

    // Parse the IR
    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diag;
    auto mem = llvm::MemoryBuffer::getMemBuffer(llvm_ir);
    auto mod = llvm::parseIR(*mem, diag, *ctx);
    if (!mod) {
        if (error_out) {
            std::string msg;
            llvm::raw_string_ostream os(msg);
            diag.print("cpujit", os);
            *error_out = make_error(msg);
        }
        return nullptr;
    }

    auto tm = create_host_target_machine();
    if (tm) optimize(*mod, *tm);

    auto asm_text = tm ? emit_asm(*mod, *tm) : std::string{};

    // Add the module to the JIT
    auto tsm = llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx));
    if (auto err = jit->addIRModule(std::move(tsm))) {
        if (error_out) {
            std::string msg;
            llvm::raw_string_ostream os(msg);
            os << err;
            *error_out = make_error(msg);
        } else {
            llvm::consumeError(std::move(err));
        }
        return nullptr;
    }

    auto* result = new cpujit_module{};
    result->jit = std::move(jit);
    result->asm_text = std::move(asm_text);
    return result;
}

CPUJIT_API void* cpujit_lookup(cpujit_module_t mod, const char* symbol_name) {
    if (!mod || !mod->jit) return nullptr;

    auto sym_or_err = mod->jit->lookup(symbol_name);
    if (!sym_or_err) {
        llvm::consumeError(sym_or_err.takeError());
        return nullptr;
    }
    return sym_or_err->toPtr<void*>();
}

CPUJIT_API const char* cpujit_preamble(void) {
    return get_sleef_registry().preamble.c_str();
}

CPUJIT_API const char* cpujit_asm(cpujit_module_t mod) {
    if (!mod) return "";
    return mod->asm_text.c_str();
}

CPUJIT_API void cpujit_destroy(cpujit_module_t mod) {
    delete mod;
}

CPUJIT_API void cpujit_error_free(const char* error) {
    delete[] error;
}

} // extern "C"
