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

#include <spdlog/spdlog.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/gpuinfo.h>

#include <inja/inja.hpp>

template<typename AdjMap, typename VertexID>
void order_recurse(const VertexID& vertex, const AdjMap& adj, std::set<VertexID>& visited, std::deque<VertexID>& ordering) {
    visited.insert(vertex);
    for (const auto& edge : adj.at(vertex)) {
        if (!visited.contains(edge)) order_recurse(edge, adj, visited, ordering);
    }
    ordering.push_front(vertex);
}

template<class VertexID, class AdjMap>
auto get_ordering(const AdjMap& adj) -> std::deque<VertexID> {
    auto ordering = std::deque<VertexID>{};
    auto visited = std::set<VertexID>{};

    for (const auto& [vertex, edges] : adj) {
        if (!visited.contains(vertex)) order_recurse(vertex, adj, visited, ordering);
    }

    return ordering;
}

std::string expand_tabs(const std::string& src, const std::string& tab_str = "\t") {
    std::string result = src;

    result.insert(0, tab_str);

    for (std::size_t i = 0; i < result.length(); i++) {
        if (result[i] == '\n') {
            result.insert(i + 1, tab_str);
            i++;
        }
    }

    return result;
}
std::string strip_tabs(const std::string& src) {
    std::string ret = src;
    ret.erase(std::remove(ret.begin(), ret.end(), '\t'), ret.end());
    return ret;
}

auto extract_common(const rfkt::flame& f) -> std::map<rfkt::hash_t, std::set<int>> {

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

auto make_struct(const rfkt::flame& f) -> std::string {

    auto info = json::object({
        {"xform_definitions", json::array()},
        {"xforms", json::array()},
        {"num_standard_xforms", f.xforms.size()}
    });

    auto& xfs = info["xforms"];
    auto& xfd = info["xform_definitions"];

    const auto common = extract_common(f);

    for (auto& [hash, children] : common) {
        auto xf_def_js = json::object({
            {"hash", hash.str32()},
            {"vchain", json::array()}
        });

        auto& vc = xf_def_js["vchain"];

        const auto child_id = children.begin().operator*();
        const auto& child = (child_id == f.xforms.size()) ? f.final_xform.value() : f.xforms[child_id];

        for (auto& vl : child.vchain) {
            auto vl_def_js = json::object({
                {"variations", json::array()},
                {"parameters", json::array()},
                {"precalc", json::array()},
                {"common", json::array()}
             });

            std::map<std::string, std::vector<std::string>> required_common;

            vl.for_each_variation([&](const auto& vdef, const auto& weight) {
                vl_def_js["variations"].push_back(
                    json::object({
                        {"name", vdef.name},
                        {"id", vdef.index}
                        })
                );

                for (const auto& common : vdef.dependencies) {
                    const auto& name = common.get().name;
                    if (required_common.count(name) > 0) continue;

                    required_common[name] = {};

                    for (auto& cdep : common.get().dependencies) {
                        required_common[name].push_back(cdep);
                    }
                }
            });

            auto ordering = get_ordering<std::string>(required_common);

            while (ordering.size()) {
                const auto& cdef = rfkt::flame_info::common(ordering.back());
                vl_def_js["common"].push_back(json::object({
                    {"name", cdef.name},
                    {"source", cdef.source}
                    }));
                ordering.pop_back();
            }
            
            vl.for_each_parameter([&](const auto& pdef, const auto& weight) {
                vl_def_js["parameters"].push_back(pdef.name);
            });

            vl.for_each_precalc([&](const auto& pdef) {
                vl_def_js["precalc"].push_back(pdef.name);
            });

            vc.push_back(std::move(vl_def_js));
        }

        xfd.push_back(std::move(xf_def_js));
    }

    for (int i = 0; i <= f.xforms.size(); i++) {
        if (i == f.xforms.size() && !f.final_xform.has_value()) break;
        auto& xf = (i == f.xforms.size()) ? f.final_xform.value() : f.xforms[i];

        xfs.push_back(json::object({
            {"hash", xf.hash().str32()},
            {"id", (i == f.xforms.size()) ? std::string{"final"} : fmt::format("{}", i)}
            }));
    }

    auto environment = []() -> inja::Environment {
        auto env = inja::Environment{};

        env.set_expression("@", "@");
        env.set_statement("<#", "#>");

        env.add_callback("get_variation_source", 2, [](inja::Arguments& args) {
            return expand_tabs(rfkt::flame_info::variation(args.at(0)->get<std::uint32_t>()).source, args.at(1)->get<std::string>());
        });

        env.add_callback("variation_has_precalc", 1, [](inja::Arguments& args) {
            return rfkt::flame_info::variation(args.at(0)->get<std::uint32_t>()).precalc_source.size() > 0;
        });

        env.add_callback("get_precalc_source", 2, [](inja::Arguments& args) {
            return expand_tabs(rfkt::flame_info::variation(args.at(0)->get<std::uint32_t>()).precalc_source, args.at(1)->get<std::string>());
        });

        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        return env;
    }();
    SPDLOG_INFO("{}", info.dump(1));
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

auto rfkt::flame_compiler::get_flame_kernel(precision prec, const flame& f) -> result
{
    auto src = make_struct(f);
    auto [most_blocks, opts] = make_opts(prec, f);
    opts.header("flame_generated.h", src);
    auto compile_result = km.compile(opts);


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
    SPDLOG_INFO("Loaded flame kernel: {} temp. samples, {} flame params, {} regs, {} shared, {} local.", max_blocks, f.real_count(), func.register_count(), func.shared_bytes(), func.local_bytes());

    auto shuf_dev = compile_result.module.value()["shuf_bufs"];
    cuMemcpyDtoD(shuf_dev.ptr(), shuf_bufs[most_blocks.block].ptr(), shuf_dev.size());

    r.kernel = flame_kernel{ f.hash(), f.real_count(), std::move(compile_result.module.value()), std::pair<int, int>{most_blocks.grid, most_blocks.block}, catmull, gpuinfo::device::by_index(0).clock() };

    return r;
}

rfkt::cuda_buffer<unsigned short> make_shuffle_buffers(std::size_t particles_per_temporal_sample, std::size_t num_shuf_buf) {
    using shuf_t = unsigned short;

    auto buf = rfkt::cuda_buffer<unsigned short>( particles_per_temporal_sample * num_shuf_buf);

    shuf_t* shuf_bufs_local = (shuf_t*)malloc(sizeof(shuf_t) * particles_per_temporal_sample * num_shuf_buf);

    if (shuf_bufs_local == nullptr) exit(1);

    for (shuf_t j = 0; j < particles_per_temporal_sample; j++) {
        shuf_bufs_local[j] = j;
    }

    for (int i = 1; i < num_shuf_buf; i++) {
        memcpy(shuf_bufs_local + i * particles_per_temporal_sample, shuf_bufs_local, sizeof(shuf_t) * particles_per_temporal_sample);
    }

    std::vector<std::thread> gen_threads;
    int bufs_per_thread = num_shuf_buf / 8;

    for (int thread_idx = 0; thread_idx < 8; thread_idx++) {
        gen_threads.emplace_back([=]() {
            auto engine = std::default_random_engine(thread_idx);

            int start = thread_idx * bufs_per_thread;
            int end = thread_idx * bufs_per_thread + bufs_per_thread;

            for (int i = start; i < end; i++) {
                std::shuffle(shuf_bufs_local + i * particles_per_temporal_sample, shuf_bufs_local + (i + 1) * particles_per_temporal_sample, engine);
            }
            });
    }

    for (auto& t : gen_threads) t.join();

    cuMemcpyHtoD(buf.ptr(), shuf_bufs_local, sizeof(shuf_t) * particles_per_temporal_sample * num_shuf_buf);
    free(shuf_bufs_local);
    return buf;
}

rfkt::flame_compiler::flame_compiler(ezrtc::compiler& k_manager): km(k_manager)
{
    num_shufs = 4096;

    exec_configs = cuda::context::current().device().concurrent_block_configurations();

    std::string check_kernel_name = "get_sizes<";
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


    for (auto& exec : exec_configs) {
        shuf_bufs[exec.block] = make_shuffle_buffers(exec.block, num_shufs);

        fmt::print("{{{}, {}, {}}},\n", exec.grid, exec.block, exec.shared_per_block);

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

    auto result = km.compile(
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
    auto flame_real_count = f.real_count();
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

    while (exec_configs[most_blocks_idx].grid > 500) {
        most_blocks_idx--;
    }

    //if (most_blocks_idx > 0) most_blocks_idx--;
    auto& most_blocks = exec_configs[most_blocks_idx];

    //flame_generated += fmt::format("__device__ unsigned int flame_size_reals() {{ return {}; }}\n", flame_real_count);
    //flame_generated += fmt::format("__device__ unsigned int flame_size_bytes() {{ return {}; }}\n", flame_size_bytes);

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

auto rfkt::flame_kernel::bin(CUstream stream, flame_kernel::saved_state & state, float target_quality, std::uint32_t ms_bailout, std::uint32_t iter_bailout) const -> std::future<bin_result>
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

    auto klauncher = [&mod = this->mod, stream, &exec = this->exec]<typename ...Ts>(Ts&&... args) {
        return mod("bin").launch(exec.first, exec.second, stream, true)(std::forward<Ts>(args)...);
    };

    CUDA_SAFE_CALL(cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->start = std::chrono::high_resolution_clock::now();
        }, stream_state));

    CUDA_SAFE_CALL(klauncher(
        state.shared.ptr(),
        (std::size_t)(target_quality * stream_state->total_bins * 255.0),
        iter_bailout,
        static_cast<std::uint64_t>(ms_bailout) * 1'000'000,
        state.bins.ptr(), state.bin_dims.x, state.bin_dims.y,
        stream_state->qpx_dev.ptr(),
        stream_state->qpx_dev.ptr() + counter_size,
        stream_state->qpx_dev.ptr() + 2 * counter_size
    ));

    stream_state->qpx_dev.to_host(stream_state->qpx_host, stream);
    stream_state->qpx_dev.free_async(stream);

    CUDA_SAFE_CALL(cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->end = std::chrono::high_resolution_clock::now();

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

auto rfkt::flame_kernel::warmup(CUstream stream, const flame& f, uint2 dims, double t, std::uint32_t nseg, double loops_per_frame, std::uint32_t seed, std::uint32_t count) const -> flame_kernel::saved_state
{
    const auto nreals = f.real_count() + 256ull * 3ull;
    const auto pack_size_reals = nreals * (nseg + 3ull);

    auto samples_dev = rfkt::cuda_buffer<double>{ pack_size_reals, stream };
    auto segments_dev = rfkt::cuda_buffer<double>{ pack_size_reals * 4 * nseg, stream };

    auto state = flame_kernel::saved_state{ dims, this->saved_state_size(), stream};

    thread_local pinned_ring_allocator_t<1024 * 1024 * 8> pra{};
    const auto pack = pra.reserve<double>(pack_size_reals);
    auto packer = [pack, counter = 0](double v) mutable {
        pack[counter] = v;
        counter++;
    };

    const auto seg_length = (loops_per_frame * 1.2) / (nseg);
    for (int pos = -1; pos < static_cast<int>(nseg) + 2; pos++) {
        f.pack(packer, dims, t + pos * seg_length);
    }

    samples_dev.from_host(pack, stream);

    const auto [grid, block] = this->catmull->kernel().suggested_dims();
    auto nblocks = (nseg * pack_size_reals) / block;
    if ((nseg * pack_size_reals) % block > 0) nblocks++;
    CUDA_SAFE_CALL(
        this->catmull->kernel().launch(nblocks, block, stream, false)
        (
            samples_dev.ptr(),
            static_cast<std::uint32_t>(nreals), 
            std::uint32_t{ nseg }, 
            segments_dev.ptr()
        ));



    CUDA_SAFE_CALL(this->mod.kernel("warmup")
        .launch(this->exec.first, this->exec.second, stream, true)
        (
            std::uint32_t{ nseg },
            segments_dev.ptr(),
            seed, count,
            state.shared.ptr()
            ));

    segments_dev.free_async(stream);
    samples_dev.free_async(stream);

    return state;
}

auto rfkt::flame_kernel::warmup(CUstream stream, std::span<const double> samples, uint2 dims, std::uint32_t seed, std::uint32_t count) const -> flame_kernel::saved_state
{
    assert(samples.size() % this->flame_size_reals == 0);
    assert(samples.size() / this->flame_size_reals > 3);

    const auto nseg = static_cast<std::uint32_t>(samples.size() / flame_size_reals - 3);

    thread_local pinned_ring_allocator_t<1024 * 1024 * 8> pra{};

    auto pinned_samples = pra.reserve<double>(samples.size());
    std::ranges::copy(samples, std::begin(pinned_samples));
    auto samples_dev = rfkt::cuda_buffer<double>{ samples.size(), stream};
    samples_dev.from_host(pinned_samples, stream);

    auto segments_dev = rfkt::cuda_buffer<double>{ samples.size() * 4 * nseg, stream};

    const auto [grid, block] = this->catmull->kernel().suggested_dims();
    auto nblocks = (nseg * samples.size()) / block;
    if ((nseg * samples.size()) % block > 0) nblocks++;
    CUDA_SAFE_CALL(
        this->catmull->kernel().launch(nblocks, block, stream, false)
        (
            samples_dev.ptr(),
            static_cast<std::uint32_t>(flame_size_reals),
            nseg ,
            segments_dev.ptr()
            ));


    auto state = flame_kernel::saved_state{ dims, this->saved_state_size(), stream };

    CUDA_SAFE_CALL(
        this->mod.kernel("warmup")
        .launch(this->exec.first, this->exec.second, stream, true)
        (
            nseg,
            segments_dev.ptr(),
            seed, count,
            state.shared.ptr()
            ));

    segments_dev.free_async(stream);
    samples_dev.free_async(stream);

    return state;
}
