#define EZRTC_IMPLEMENTATION_UNIT

#include <stack>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <random>
#include <span>
#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/gpuinfo.h>
#include <librefrakt/util/filesystem.h>

#include <inja/inja.hpp>

#include <flang/grammar.h>
#include <flang/matcher.h>

using json = nlohmann::json;

std::string expand_tabs(const std::string& src, const std::string& tab_str = "    ") {
    std::string result = src;

    result.insert(0, tab_str);

    for (std::size_t i = 0; i < result.length(); i++) {
        if (result[i] == '\n' /* && result.find('\n', i + 1) != std::string::npos*/) {
            result.insert(i + 1, tab_str);
            i+=tab_str.size();
        }
    }

    return result;
}

std::string create_source(std::string_view name, const flang::ast_node* node);

auto make_table() {
    auto table = std::vector<std::pair<flang::matcher, std::add_pointer_t<std::string(std::string_view, const flang::ast_node*)>>>{};

    using namespace flang::matchers;
    using namespace flang::grammar;

    // fma
    table.emplace_back(
        of_type<op::plus, op::minus, op::plus_assign, op::minus_assign>
        and with_child(of_type<op::times>),
        [](std::string_view name, const flang::ast_node* n) {
            auto multiply_child = (n->first()->is_type<op::times>()) ? 0 : 1;
            auto mul_lhs = create_source(name, n->nth(multiply_child)->nth(0));
            auto mul_rhs = create_source(name, n->nth(multiply_child)->nth(1));
            auto add_lhs = create_source(name, n->nth(1 - multiply_child));

            if (n->is_type<op::plus>()) {
                return std::format("fl::fma<FloatT>({}, {}, {})", mul_lhs, mul_rhs, add_lhs);
            }
            else if (n->is_type<op::minus>()) {
                if (multiply_child == 0)
                    return std::format("fl::fma<FloatT>({}, {}, -({}))", mul_lhs, mul_rhs, add_lhs);
                else
                    return std::format("fl::fma<FloatT>({}, -({}), {})", mul_lhs, mul_rhs, add_lhs);
            }
            else if (n->is_type<op::plus_assign>()) {
                return std::format("{} = fl::fma<FloatT>({}, {}, {})", add_lhs, mul_lhs, mul_rhs, add_lhs);
            }
            else {
                return std::format("{} = fl::fma<FloatT>({}, -({}), {})", add_lhs, mul_lhs, mul_rhs, add_lhs);
            }
        }
    );

    // affine/param/common access
    table.emplace_back(
        of_type<flang::grammar::member_access>
        and with_child(
            of_type<flang::grammar::variable>
            and with_content<"aff", "param", "math", "common">
        ),
        [](std::string_view name, const flang::ast_node* n) {
            auto group = n->nth(0)->content();
            auto member = n->nth(1)->content();
            if (group == "aff") {
                return std::format("aff.{}", member);
            }
            else if (group == "param") {
                return std::format("p_{}_{}", name, member);
            }
            else if (group == "math") {
                return std::format("fl::math::{}<FloatT>", member);
            }
            else {
                return std::format("common_{}", member);
            }
        }
    );

    // root/scope
    table.emplace_back(
        of_type<scoped> or [](const flang::ast_node* n) { return n->type() == "root"; },
        [](std::string_view name, const flang::ast_node* n) {
            auto child_statements = std::string{};
            for (auto i = 0; i < n->size(); ++i) {
                child_statements += std::format("{}", create_source(name, n->nth(i)));
                if (i + 1 != n->size()) {
                    child_statements += "\n";
                }
            }
            if (n->type() == "root") {
                if (n->has_descendent(of_type<break_statement>)) {
                    return std::format("do {{\n{}\n}} while(0);", expand_tabs(child_statements));
                }
                if (n->has_descendent(of_type<declaration_statement>)) {
                    return std::format("{{\n{}\n}}", expand_tabs(child_statements));
                }
                return child_statements;
            }
            return std::format("{{\n{}\n}}", expand_tabs(child_statements));
        });

    // assignment
    table.emplace_back(of_type<assignment_statement>,
        [](std::string_view name, const flang::ast_node* n) {
            return std::format("{};", create_source(name, n->first()));
        });

    // declaration
    table.emplace_back(of_type<declaration_statement>,
        [](std::string_view name, const flang::ast_node* n) {
            auto lhs = std::format("u_{}__", n->nth(0)->content());
            auto rhs = create_source(name, n->nth(1));

            const auto variable_assigned = [&vname = n->nth(0)->content()](const flang::ast_node* n) {
                constexpr static auto is_assignment_target =
                    of_type<variable, member_access> and of_rank<0> and with_parent(
                        of_type<op::assignment, op::plus_assign, op::minus_assign, op::times_assign, op::divided_assign>
                    );

                return is_assignment_target(n) and vname == ((n->is_type<variable>())? n->content() : n->nth(0)->content());
            };

            if (n->parent()->has_descendent(variable_assigned)) {
                return std::format("auto {} = {};", lhs, rhs);
			}

            return std::format("const auto {} = {};", lhs, rhs);
        });

    table.emplace_back(of_type<if_statement>,
        [](std::string_view name, const flang::ast_node* n) {
            const auto* condition = n->nth(0);
            const auto* branch = n->nth(1);

            auto condition_src = create_source(name, condition);
            auto branch_src = create_source(name, branch);

            if (!branch->is_type<scoped>()) {
                branch_src = "\n    " + branch_src;
            }

            auto str = std::format("if {} {}", condition_src, branch_src);
            if (n->size() == 3) {
                auto else_branch = n->nth(2);
                auto else_src = create_source(name, else_branch);
                if (!else_branch->is_type<scoped>() && !else_branch->is_type<if_statement>()) {
                    else_src = "\n    " + else_src;
                }

                str += std::format("\nelse {}", else_src);
            }

            return str;
        });

    table.emplace_back(of_type<expr::parenthesized>,
        [](std::string_view name, const flang::ast_node* n) {
            return std::format("({})", create_source(name, n->first()));
        });

    table.emplace_back(of_type<break_statement>,
        [](std::string_view, const flang::ast_node* n) {
            return std::string{ "break;" };
        });

    const static std::map<std::string_view, std::string, std::less<>> op_map = {
        {demangle<op::plus>(), "+"},
        {demangle<op::minus>(), "-"},
        {demangle<op::times>(), "*"},
        {demangle<op::divided>(), "/"},
        {demangle<op::equals>(), "=="},
        {demangle<op::not_equals>(), "!="},
        {demangle<op::less_than>(), "<"},
        {demangle<op::less_equal>(), "<="},
        {demangle<op::greater_than>(), ">"},
        {demangle<op::greater_equal>(), ">="},
        {demangle<op::b_and>(), "&&"},
        {demangle<op::b_or>(), "||"},
        {demangle<op::assignment>(), "="},
        {demangle<op::plus_assign>(), "+="},
        {demangle<op::minus_assign>(), "-="},
        {demangle<op::times_assign>(), "*="},
        {demangle<op::divided_assign>(), "/="},
    };

    // binary ops
    table.emplace_back(
        [](const flang::ast_node* node) { return op_map.contains(node->type()); },
        [](std::string_view name, const flang::ast_node* node) {
            return std::format("{} {} {}", create_source(name, node->nth(0)), op_map.at(node->type()), create_source(name, node->nth(1)));
        }
    );

    // unary
    table.emplace_back(
        of_type<op::un_negative, op::un_b_not>,
        [](std::string_view name, const flang::ast_node* node) {
            return std::format("{}{}", (node->is_type<op::un_negative>()) ? "-" : "!", create_source(name, node->first()));
        }
    );

    // literals
    table.emplace_back(
        of_type<lit::decimal, lit::integer, lit::boolean>,
        [](std::string_view name, const flang::ast_node* node) {
            if (node->is_type<lit::decimal>()) return std::format("static_cast<FloatT>({})", node->content());
            return node->content();
        }
    );

    // member access
    table.emplace_back(
        of_type<member_access>,
        [](std::string_view name, const flang::ast_node* node) {
            return std::format("{}.{}", create_source(name, node->nth(0)), create_source(name, node->nth(1)));
        }
    );

    // result, p, and weight
    table.emplace_back(
        of_type<variable> and with_content<"result", "p", "weight">,
        [](std::string_view name, const flang::ast_node* node) -> std::string {
            if (node->content() == "result") return "outp";
            if (node->content() == "weight") return std::format("v_{}", name);
            return "inp";
        }
    );

    // calls to randomness functions
    table.emplace_back(
        of_type<expr::call>
        and with_child(with_content<"rand01", "randgauss", "randbit">),
        [](std::string_view name, const flang::ast_node* node) {
            return std::format("rs->{}()", node->first()->content());
        }
    );

    // vec constructor
    table.emplace_back(
        of_type<expr::call> and with_child_at<0>(with_content<"vec2", "vec3">),
        [](std::string_view name, const flang::ast_node* node) {
            auto x = create_source(name, node->nth(1));
            auto y = create_source(name, node->nth(2));
            if (node->content() == "vec3") {
                auto z = create_source(name, node->nth(3));
                return std::format("vec3<FloatT>{{{}, {}, {}}}", x, y, z);
            }

            return std::format("vec2<FloatT>{{{}, {}}}", x, y);
        }
    );

    // calls
    table.emplace_back(
        of_type<expr::call>,
        [](std::string_view name, const flang::ast_node* node) {
            auto func_name = create_source(name, node->nth(0));
            if (node->size() == 1) {
                return func_name + "()";
            }
            auto args = std::string{};
            auto nargs = node->size() - 1;
            for (int i = 0; i < nargs; i++) {
                args += create_source(name, node->nth(i + 1));
                if (i + 1 != nargs) {
                    args += ", ";
                }
            }

            static const std::set<std::string> integer_functions = {
                "is_even"
            };

            if (integer_functions.contains(func_name)) {
                return std::format("fl::{}({})", func_name, args);
            }

            return std::format("fl::{}<FloatT>({})", func_name, args);
        }
    );

    // accessed members and function names
    table.emplace_back(
        of_type<variable>
        and (
            (with_parent(of_type<expr::call>) and of_rank<0>)
            or (with_parent(of_type<member_access>) and of_rank<1>)
            ),
        [](std::string_view name, const flang::ast_node* node) -> std::string {
            return node->content();
        }
    );

    // any other variable
    table.emplace_back(
        of_type<variable>,
        [](std::string_view name, const flang::ast_node* node) -> std::string {
            return std::format("u_{}__", node->content());
        }
    );

    return table;
}

std::string create_source(std::string_view name, const flang::ast_node* node) {

    const static std::vector<std::pair<flang::matcher, std::add_pointer_t<std::string(std::string_view, const flang::ast_node*)>>> outputters = make_table();

    for(const auto& [matcher, outputter] : outputters) {
        if (matcher(node)) {
			return outputter(name, node);
		}
	}

    SPDLOG_ERROR("no formatter defined for `{}`", node->type());
    return std::format("@{}@", node->type());
}

std::string strip_tabs(const std::string& src) {
    std::string ret = src;
    ret.erase(std::remove(ret.begin(), ret.end(), '\t'), ret.end());
    return ret;
}

auto extract_duplicates(const rfkt::flame& f) -> std::map<rfkt::hash_t, std::set<int>> {

    std::map<rfkt::hash_t, std::set<int>> shared_xforms;
    for (int i = 0; i <= f.xforms.size(); i++) {
        if (i == f.xforms.size() && !f.final_xform.has_value()) break;
        auto& xf = (i == f.xforms.size()) ? f.final_xform.value() : f.xforms[i];
        auto xhash = xf.hash();

        if (shared_xforms.contains(xhash))
            shared_xforms[xhash].insert(i);
        else {
            shared_xforms[xhash] = { i };
        }
    }
    return shared_xforms;
}

void collect_common(const rfkt::flamedb& fdb, const flang::ast* ast, std::vector<std::string_view>& storage) noexcept {
    using namespace flang::grammar;
    using namespace flang::matchers;

    constexpr static auto is_common_access = of_type<member_access> and with_child_at<0>(with_content<"common">);
    constexpr static auto vector_contains = [](const auto& vec, const auto& val) {
		return std::find(vec.begin(), vec.end(), val) != vec.end();
	};

    for (const auto& node : *ast) {
        if (!is_common_access(&node)) continue;
        std::string_view common_name = node.nth(1)->content();
        if (vector_contains(storage,common_name)) continue;

        const auto* common_source = &fdb.get_common_ast(common_name);
        collect_common(fdb, common_source, storage);
        storage.push_back(common_name);
    }
}

std::string rfkt::flame_compiler::make_source(const flamedb& fdb, const rfkt::flame& f) {

    auto info = json::object({
        {"xform_definitions", json::object()},
        {"xforms", json::array()},
        {"num_standard_xforms", f.xforms.size()}
    });

    auto& xfs = info["xforms"];
    auto& xfd = info["xform_definitions"];

    const auto shared_xforms = extract_duplicates(f);

    std::set<std::string_view> needed_variations;

    for (auto& [hash, children] : shared_xforms) {
        auto xf_def_js = json::object({
            {"vchain", json::array()}
        });

        auto& vc = xf_def_js["vchain"];

        const auto child_id = children.begin().operator*();
        const auto& child = (child_id == f.xforms.size()) ? f.final_xform.value() : f.xforms[child_id];

        for (auto& vl : child.vchain) {
            auto vl_def_js = json::object({
                {"variations", json::array()},
                {"common", json::array()}
             });

            std::vector<std::string_view> required_common;

            for(const auto& [name, vd] : vl) {
                auto vinfo = json::object({
                    {"name", name},
                    {"parameters", json::array()},
                    {"precalc", json::array()}
                });

                const auto& vdef = fdb.get_variation(name);

                for (const auto& [name, pdef] : vdef.parameters) {
                    if (pdef.tags.count("precalc")) {
                        vinfo["precalc"].push_back(name);
                    }
                    else {
                        vinfo["parameters"].push_back(name);
                    }
				}

                collect_common(fdb, fdb.get_variation_ast(name).apply, required_common);

                vl_def_js["variations"].push_back(std::move(vinfo));

                needed_variations.insert(name);
            };

            for(auto& cname: required_common) {
                if (!compiled_common.contains(cname)) {
                    const auto& src = fdb.get_common_ast(cname);
                    compiled_common.try_emplace(std::string{ cname }, create_source(cname, src.head()));
                }

                vl_def_js["common"].push_back(json::object({
                    {"name", cname},
                    {"source", compiled_common.find(cname)->second}
                    }));
            }

            vc.push_back(std::move(vl_def_js));
        }

        xfd[hash.str32()] = std::move(xf_def_js);
    }

    for (int i = 0; i <= f.xforms.size(); i++) {
        if (i == f.xforms.size() && !f.final_xform.has_value()) break;
        auto& xf = (i == f.xforms.size()) ? f.final_xform.value() : f.xforms[i];

        xfs.push_back(json::object({
            {"hash", xf.hash().str32()},
            {"id", (i == f.xforms.size()) ? std::string{"final"} : fmt::format("{}", i)}
            }));
    }

    for (auto& v : needed_variations) {
        if (compiled_variations.contains(v)) continue;

        const auto vsrc = fdb.get_variation_ast(v);

        std::pair<std::string, std::string> compiled;
        compiled.first = create_source(v, vsrc.apply->head());
        if (vsrc.precalc) {
            compiled.second = create_source(v, vsrc.precalc->head());
        }

        compiled_variations.emplace(v, std::move(compiled));
    }

    auto environment = [&compiled_variations=this->compiled_variations, &compiled_common=this->compiled_common, &fdb]() -> inja::Environment {
        auto env = inja::Environment{};

        env.set_expression("@", "@");
        env.set_statement("<#", "#>");

        env.add_callback("get_variation_source", 2, [&compiled_variations](inja::Arguments& args) {
            return expand_tabs(compiled_variations[args.at(0)->get<std::string>()].first, args.at(1)->get<std::string>());
        });

        env.add_callback("variation_has_precalc", 1, [&compiled_variations](inja::Arguments& args) {
            return !compiled_variations[args.at(0)->get<std::string>()].second.empty();
        });

        env.add_callback("get_precalc_source", 2, [&compiled_variations](inja::Arguments& args) {
            return expand_tabs(compiled_variations[args.at(0)->get<std::string>()].second, args.at(1)->get<std::string>());
        });

        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        return env;
    }();

    return environment.render_file("./assets/templates/flame.tpl", info);

}

std::string annotate_source(std::string src) {
    int linecount = 2;
    for (int i = src.find("\n"); i != std::string::npos; i = src.find("\n", i)) {
        auto linenum = fmt::format("{:>4}| ", linecount);
        src.insert(i + 1, linenum);
        i += linenum.size();
        linecount++;
    }

    return fmt::format("{:>4}| ", 1) + src;
}

void rfkt::flame_compiler::add_to_hash(rfkt::hash::state_t& state)
{
    state.update(rfkt::fs::last_modified("assets/kernels/refactor.cu"));
    state.update(rfkt::fs::last_modified("assets/kernels/include/refrakt/random.h"));
    state.update(rfkt::fs::last_modified("assets/kernels/include/refrakt/flamelib.h"));
}

auto rfkt::flame_compiler::get_flame_kernel(const flamedb& fdb, precision prec, const flame& f) -> result
{
    auto start = std::chrono::high_resolution_clock::now();

    if (fdb.hash() != last_flamedb_hash) {
        compiled_common.clear();
        compiled_variations.clear();
        last_flamedb_hash = fdb.hash();
    }

    auto src = make_source(fdb, f);
    auto [most_blocks, opts] = make_opts(prec, f);
    opts.header("flame_generated.h", src);

    auto rand_src = rfkt::fs::read_string("assets/kernels/include/refrakt/random.h");
    auto flamelib_src = rfkt::fs::read_string("assets/kernels/include/refrakt/flamelib.h");

    opts.header("refrakt/random.h", rand_src);
    opts.header("refrakt/flamelib.h", flamelib_src);
    opts.define("FLAMEDB_HASH", fdb.hash().str32());

    auto compile_result = km->compile(opts);
    auto duration_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1'000'000.0;

    auto r = result(
        annotate_source(src),
        std::move(compile_result.log)
    );

    //SPDLOG_INFO("{}", r.source);

    if (not compile_result.module.has_value()) {
        return r;
    }

    auto func = compile_result.module->kernel("bin");
    compile_result.module->kernel("print_debug_info").launch(1, 1)();

    auto max_blocks = func.max_blocks_per_mp(most_blocks.block) * cuda::context::current().device().mp_count();
    if (max_blocks < most_blocks.grid) {
        SPDLOG_ERROR("Kernel for {} needs {} blocks but only got {}", f.hash().str64(), most_blocks.grid, max_blocks);
        return r;
    }
    SPDLOG_INFO("Loaded flame kernel {}: {} temp. samples, {} flame params, {} regs, {} shared, {} local, {:.4} ms", f.hash().str64(), max_blocks, f.size_reals(), func.register_count(), func.shared_bytes(), func.local_bytes(), duration_ms);

    auto shuf_dev = compile_result.module.value()["shuf_bufs"];
    cuMemcpyDtoD(shuf_dev.ptr(), shuf_bufs[most_blocks.block].ptr(), shuf_dev.size());

    r.kernel = flame_kernel{ f.hash(), f.size_reals() + 13, std::move(compile_result.module.value()), std::pair<int, int>{most_blocks.grid, most_blocks.block}, catmull, gpuinfo::device::by_index(0).clock() };

    return r;
}

rfkt::cuda_buffer<unsigned short> make_shuffle_buffers(std::size_t ppts, std::size_t count) {
    using shuf_t = unsigned short;
    const auto shuf_size = ppts * count;

    auto shuf_local = std::vector<shuf_t>(shuf_size);
    auto engine = std::default_random_engine(0);

    for (shuf_t j = 0; j < ppts; j++) {
        shuf_local[j] = j;
    }

    std::shuffle(shuf_local.begin(), shuf_local.begin() + ppts, engine);

    for (int i = 1; i < count; i++) {
        auto start = shuf_local.begin() + i * ppts;
        auto end = shuf_local.begin() + (i + 1) * ppts;

        std::copy(shuf_local.begin(), shuf_local.begin() + ppts, start);
        std::shuffle(start, end, engine);
    }

    auto buf = rfkt::cuda_buffer<unsigned short>(shuf_local.size());
    buf.from_host(shuf_local);
    return buf;
}

rfkt::flame_compiler::flame_compiler(std::shared_ptr<ezrtc::compiler> k_manager): km(k_manager)
{
    num_shufs = 4096;

    exec_configs = cuda::context::current().device().concurrent_block_configurations();

    /*std::string check_kernel_name = "get_sizes<";
    for (const auto& conf : exec_configs) {
        check_kernel_name += std::format("{},", conf.block);
    }

    check_kernel_name[check_kernel_name.size() - 1] = '>';
  
    auto check_kernel_result_name = std::format("required_shared<{}>", exec_configs.size());

    auto check_result = km.compile(
        ezrtc::spec::source_file("sizing", "assets/kernels/size_info.cu")
        .variable(check_kernel_result_name)
        .kernel(check_kernel_name)
        .flag(ezrtc::compile_flag::default_device)
    );

    if (not check_result.module.has_value()) {
        SPDLOG_ERROR(check_result.log);
        exit(1);
    }

    check_result.module->kernel(check_kernel_name).launch(1, 1)();
    auto var = check_result.module.value()[check_kernel_result_name];
    std::vector<unsigned long long> shared_sizes{};
    shared_sizes.resize(exec_configs.size() * 2);
    cuMemcpyDtoH(shared_sizes.data(), var.ptr(), var.size());
    */

    for (auto& exec : exec_configs) {
        shuf_bufs[exec.block] = make_shuffle_buffers(exec.block, num_shufs);

        //fmt::print("{{{}, {}, {}}},\n", exec.grid, exec.block, exec.shared_per_block);

        {
            auto leftover = exec.shared_per_block - smem_per_block(precision::f32, 0, exec.block);
            if (leftover <= 0) continue;
            //SPDLOG_INFO("{}x{}xf32: {:> 5} leftover shared ({:> 5} floats)", exec.grid, exec.block, leftover, leftover / 4);
        }
        {
            auto leftover = exec.shared_per_block - smem_per_block(precision::f64, 0, exec.block);
            if (leftover <= 0) continue;
            //SPDLOG_INFO("{}x{}xf64: {:> 5} leftover shared ({:> 5} doubles)", exec.grid, exec.block, leftover, leftover / 8);
        }
    }

    const auto catmull_spec = ezrtc::spec::source_file("catmull", "assets/kernels/catmull.cu");

    auto result = km->compile(
        ezrtc::spec::
         source_file("catmull", "assets/kernels/catmull.cu")
        .kernel("generate_sample_coefficients")
        .flag(ezrtc::compile_flag::default_device)
        .flag(ezrtc::compile_flag::extra_device_vectorization)
    );

    if (not result.module.has_value()) {
        SPDLOG_ERROR(result.log);
        exit(1);
    }
    this->catmull = std::make_shared<ezrtc::cuda_module>(std::move(result.module.value()));
    auto func = (*catmull)();
    auto [s_grid, s_block] = func.suggested_dims();
    SPDLOG_INFO("Loaded catmull kernel: {} regs, {} shared, {} local, {}x{} suggested dims", func.register_count(), func.shared_bytes(), func.local_bytes(), s_grid, s_block);
}

auto rfkt::flame_compiler::make_opts(precision prec, const flame& f)->std::pair<cuda::execution_config, ezrtc::spec>
{
    auto flame_real_count = f.size_reals() + 13;
    auto flame_size_bytes = ((prec == precision::f32) ? sizeof(float) : sizeof(double)) * flame_real_count;
    auto flame_hash = f.hash();

    auto most_blocks_idx = 0;

    for (int i = exec_configs.size() - 1; i >= 0; i--) {
        auto& ec = exec_configs[i];
        if (smem_per_block(prec, flame_size_bytes, ec.block) <= ec.shared_per_block) {
            most_blocks_idx = i;
            break;
        }
    }

    //while (exec_configs[most_blocks_idx].grid > 300) {
    //    most_blocks_idx--;
    //}

    auto& most_blocks = exec_configs[most_blocks_idx];

    auto name = fmt::format("flame_{}_f{}_t{}_s{}", flame_hash.str64(), (prec == precision::f32) ? "32" : "64", most_blocks.grid, flame_real_count);

    auto opts = ezrtc::spec::source_file(name, "assets/kernels/refactor.cu");

    opts
        .flag(ezrtc::compile_flag::extra_device_vectorization)
        .flag(ezrtc::compile_flag::default_device)
        .flag(ezrtc::compile_flag::generate_line_info)
        .define("NUM_SHUF_BUFS", num_shufs)
        .define("THREADS_PER_BLOCK", most_blocks.block)
        .define("BLOCKS_PER_MP", most_blocks.grid / cuda::context::current().device().mp_count())
        .define("FLAME_SIZE_REALS", flame_real_count)
        .define("FLAME_SIZE_BYTES", flame_size_bytes)
        .define("TOTAL_THREADS", most_blocks.grid * most_blocks.block)
        .kernel("warmup")
        .kernel("bin")
        .kernel("print_debug_info")
        .variable("shuf_bufs")
        
        ;

    if (prec == precision::f64) opts.define("DOUBLE_PRECISION");
    else opts.flag(ezrtc::compile_flag::warn_double_usage);
    opts.flag(ezrtc::compile_flag::use_fast_math);

    return { most_blocks, opts };
}

template<std::size_t Size>
class pinned_ring_allocator_t {
public:
    pinned_ring_allocator_t() noexcept {
        cuMemAllocHost(&memory, Size);
    }

    ~pinned_ring_allocator_t() {
        cuMemFreeHost(&memory);
    }

    template<std::size_t Amount>
    void* reserve() {
        static_assert(Amount < Size, "Requested amount greater than allocator's size");

        if (Amount + offset >= Size) {
            offset = Amount;
            return memory;
        }

        auto old = offset;
        offset += Amount;
        return ((std::byte*)memory) + old;
    }

    template<typename T>
    T* reserve() {
        return (T*)reserve<sizeof(T)>();
    }

    template<typename T>
    std::span<T> reserve(std::size_t amount) {
        return { (T*)reserve(amount * sizeof(T)), amount };
    }

    template<typename T, std::size_t Amount>
    auto reserve()-> std::span<T> {
        return std::span{ (T*)reserve<sizeof(T)* Amount>(), Amount };
    }

    void* reserve(std::size_t amount) noexcept {
        assert(amount < Size);

        if (amount + offset >= Size) {
            offset = amount;
            return memory;
        }

        auto old = offset;
        offset += amount;
        return ((std::byte*)memory) + old;
    }

private:
    void* memory;
    std::size_t offset = 0;
};

auto rfkt::flame_kernel::bin(cuda_stream& stream, flame_kernel::saved_state & state, const bailout_args& bo) const -> std::future<bin_result>
{
    using counter_type = std::size_t;
    static constexpr auto counter_size = sizeof(counter_type);
    thread_local pinned_ring_allocator_t<counter_size * 512> pra{};

    struct stream_state_t {
        std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        decltype(start) end;
        std::size_t total_bins;
        std::size_t num_threads;

        cuda_buffer<std::size_t> qpx_dev;

        std::promise<flame_kernel::bin_result> promise{};
        std::span<std::size_t> qpx_host;
    };

    auto stream_state = new stream_state_t{};
    auto future = stream_state->promise.get_future();
    const auto num_counters = 2;
    const auto alloc_size = counter_size * num_counters;

    stream_state->qpx_host = pra.reserve<std::size_t>(num_counters);

    stream_state->total_bins = state.bin_dims.x * state.bin_dims.y;
    stream_state->num_threads = exec.first * exec.second;

    stream_state->qpx_dev = rfkt::cuda_buffer<std::size_t>{ num_counters, stream };
    stream_state->qpx_dev.clear(stream);

    auto klauncher = [&mod = this->mod, &stream, &exec = this->exec]<typename ...Ts>(Ts&&... args) {
        return mod("bin").launch(exec.first, exec.second, stream, true)(std::forward<Ts>(args)...);
    };

    CUDA_SAFE_CALL(cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->start = std::chrono::high_resolution_clock::now();
        }, stream_state));

    CUDA_SAFE_CALL(klauncher(
        state.shared.ptr(),
        (std::size_t)(bo.quality * stream_state->total_bins * 255.0),
        bo.iters,
        static_cast<std::uint64_t>(bo.millis) * 1'000'000,
        state.bins.ptr(), state.bin_dims.x, state.bin_dims.y,
        stream_state->qpx_dev.ptr(),
        stream_state->qpx_dev.ptr() + counter_size
    ));

    stream_state->qpx_dev.to_host(stream_state->qpx_host, stream);
    stream_state->qpx_dev.free_async(stream);

    CUDA_SAFE_CALL(cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->end = std::chrono::high_resolution_clock::now();
        //SPDLOG_INFO("{}", ss->qpx_host[0]);
        ss->promise.set_value(flame_kernel::bin_result{
            ss->qpx_host[0] / (ss->total_bins * 255.0),
            std::chrono::duration_cast<std::chrono::nanoseconds>(ss->end - ss->start).count() / 1'000'000.0,
            ss->qpx_host[1],
            ss->qpx_host[0] / 255,
            ss->total_bins,
            double(ss->qpx_host[1]) / (ss->num_threads)
        });

        delete ss;

    }, stream_state));

    return future;
}

auto rfkt::flame_kernel::warmup(cuda_stream& stream, std::span<double> samples, uint2 dims, std::uint32_t seed, std::uint32_t count) const -> flame_kernel::saved_state
{
    const auto sample_size = flame_size_reals + 256 * 3;
    const auto sample_count = samples.size() / sample_size;

    if (samples.size() % sample_size != 0) {
        SPDLOG_CRITICAL("Invalid sample size: {}", samples.size());
    }

    const auto nseg = static_cast<std::uint32_t>(sample_count - 3);
    const auto nreals = samples.size();

    thread_local pinned_ring_allocator_t<1024 * 1024 * 8> pra{};

    auto pinned_samples = pra.reserve<double>(nreals);

    std::copy(samples.begin(), samples.end(), pinned_samples.begin());


    /*std::size_t pack_index = 0;
    auto packer = [pinned_samples, &pack_index](double v) {
		pinned_samples[pack_index] = v;
		pack_index++;
	};

    for (const auto& sample : samples) {
        sample.make_screen_space_affine(dims.x, dims.y).pack(packer);
        sample.make_plane_space_affine(dims.y, dims.y).pack(packer);
        packer(0); // weight sum (calculated in kernel)

        sample.pack(packer);

        for (const auto& c : sample.palette) {
			packer(c[0]);
			packer(c[1]);
			packer(c[2]);
		}
    }

    assert(pack_index == nreals);*/

    auto samples_dev = rfkt::cuda_buffer<double>{ nreals, stream};
    samples_dev.from_host(pinned_samples, stream);

    auto segments_dev = rfkt::cuda_buffer<double>{ nreals * 4 * nseg, stream};

    const auto [grid, block] = this->catmull->kernel().suggested_dims();
    auto nblocks = (nseg * sample_size) / block;
    if ((nseg * samples.size()) % block > 0) nblocks++;
    CUDA_SAFE_CALL(
        this->catmull->kernel().launch(nblocks, block, stream, false)
        (
            samples_dev.ptr(),
            static_cast<std::uint32_t>(sample_size),
            nseg,
            segments_dev.ptr()
            ));


    auto state = flame_kernel::saved_state{ dims, this->saved_state_size(), stream };

    CUDA_SAFE_CALL(
        this->mod.kernel("warmup")
        .launch(this->exec.first, this->exec.second, stream, true)
        (
            nseg,
            segments_dev.ptr(),
            seed, count, dims.x, dims.y,
            state.shared.ptr()
            ));

    segments_dev.free_async(stream);
    samples_dev.free_async(stream);

    return state;
}
