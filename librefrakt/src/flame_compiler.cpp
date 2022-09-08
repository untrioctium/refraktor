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

template<typename AdjMap, typename VertexID>
void order_recurse(const VertexID& vertex, const AdjMap& adj, std::set<VertexID>& visited, std::stack<VertexID>& ordering) {
    visited.insert(vertex);
    for (const auto& edge : adj.at(vertex)) {
        if (!visited.contains(edge)) order_recurse(edge, adj, visited, ordering);
    }
    ordering.push(vertex);
}

template<class VertexID, class AdjMap>
auto get_ordering(const AdjMap& adj) -> std::stack<VertexID> {
    auto ordering = std::stack<VertexID>{};
    auto visited = std::set<VertexID>{};

    for (const auto& [vertex, edges] : adj) {
        if (!visited.contains(vertex)) order_recurse(vertex, adj, visited, ordering);
    }

    return ordering;
}

std::string expand_tabs(const std::string& src) {
    std::string result = src;

    result.insert(0, "\t");

    for (std::size_t i = 0; i < result.length(); i++) {
        if (result[i] == '\n') {
            result.insert(i + 1, "\t");
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

std::string create_vlink_source(const rfkt::vlink& vl, int offset) {
    std::string common_source;
    std::string var_sources;
    std::string setup_source = "nx = 0; ny = 0;\n";

    auto var_offset = offset;
    setup_source += fmt::format(
        "const auto& xaffine_a = state.flame[offset + {}];\n"
        "const auto& xaffine_d = state.flame[offset + {}];\n"
        "const auto& xaffine_b = state.flame[offset + {}];\n"
        "const auto& xaffine_e = state.flame[offset + {}];\n"
        "const auto& xaffine_c = state.flame[offset + {}];\n"
        "const auto& xaffine_f = state.flame[offset + {}];\n"
        "weight = xaffine_a * px + xaffine_b * py + xaffine_c;\n"
        "   py = xaffine_d * px + xaffine_e * py + xaffine_f;\n"
        "   px = weight;\n",
        offset, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5);
    var_offset += 6;


    auto var_count = vl.variations.size();
    auto param_offset = var_offset + var_count;
    std::map<std::string, std::vector<std::string>> required_common;

    for (auto& [var_idx, weight] : vl.variations) {
        auto& vdef = rfkt::flame_info::variation(var_idx);
        auto src = std::string{};
        for (auto& p : vdef.parameters) {
            src += fmt::format("const auto& {} = state.flame[offset + {}];\n", p.get().name, (param_offset++));
        }

        src += fmt::format("{}\n", vdef.source);

        static auto var_src_fmt =
            "//{}\n"
            "weight = state.flame[offset + {}];\n"
            "if(weight != 0.0) {{\n"
            "{}\n"
            "}}\n";

        var_sources += fmt::format(var_src_fmt, vdef.name, (var_offset++), expand_tabs(src));

        for (const auto& common : vdef.dependencies) {
            auto& name = common.get().name;
            required_common[name] = {};

            for (auto& cdep : common.get().dependencies) {
                required_common[name].push_back(cdep);
            }
        }
    }

    /*std::string end_sources =
        "#undef xaffine_a\n"
        "#undef xaffine_d\n"
        "#undef xaffine_b\n"
        "#undef xaffine_e\n"
        "#undef xaffine_c\n"
        "#undef xaffine_f";*/

    auto ordering = get_ordering<std::string>(required_common);

    while (ordering.size()) {
        const auto& cdef = rfkt::flame_info::common(ordering.top());
        common_source = fmt::format("Real xcommon_{} = {};\n{}", cdef.name, cdef.source, common_source);
        ordering.pop();
    }

    return setup_source + common_source + var_sources;
}

std::string create_xform_source(const rfkt::xform& xf) {
    std::string function_header = fmt::format("template<unsigned long long offset> __device__ void xform_{}( Real& px, Real& py, Real& nx, Real& ny, randctx* rs)", xf.hash().str32());

    std::string xform_src = "Real weight;\n";

    int offset = 4;
    for (auto& vl : xf.vchain) {
        xform_src += fmt::format(
            "{{\n{}\n}}\n"
            "px = nx;\n"
            "py = ny;\n", expand_tabs(create_vlink_source(vl, offset)));
        offset += vl.real_count();
    }

    return fmt::format("{}\n{{\n{}}}\n", function_header, expand_tabs(xform_src));
}

std::string create_flame_source(const rfkt::flame& f) {
    std::string src = "#define xcommon(name) xcommon_ ## name\n";

    std::map<rfkt::hash_t, std::set<int>> shared_xforms;
    std::vector<std::size_t> xform_offsets;
    std::size_t offset_counter = 7;

    for (int i = 0; i <= f.xforms.size(); i++) {
        if (i == f.xforms.size() && !f.final_xform.has_value()) continue;
        auto& xf = (i == f.xforms.size()) ? f.final_xform.value() : f.xforms[i];
        auto xhash = xf.hash();

        //fmt::print("{}\n", xhash.str32());

        if (shared_xforms.contains(xhash)) 
            shared_xforms[xhash].insert(i);
        else {
            src += create_xform_source(xf);
            shared_xforms[xhash] = { i };
        }

        xform_offsets.push_back(offset_counter);
        offset_counter += xf.real_count();
    }

    src +=
        "__device__ Real dispatch(int idx, Real& px, Real& py, Real& pc, Real& nx, Real& ny, Real& nc) {\n"
        "\tswitch(idx){\n"
        "\t\tdefault: return 0.0f;\n";

    for (auto& [hash, idxs] : shared_xforms) {
        for (auto idx : idxs) {
            src += fmt::format("\t\tcase {}:\n", idx);
            src += fmt::format(
                "\t\t{{\n"
                "\t\t\txform_{}<xform_offsets[{}]>(px, py, nx, ny, &my_rand()); \n"
                "\t\t\tnc = INTERP(pc, state.flame[xform_offsets[{}] + 1], state.flame[xform_offsets[{}] + 2]);\n"
                "\t\t\treturn state.flame[xform_offsets[{}] + 3];\n"
                "\t\t}}\n", 
                hash.str32(), idx, idx, idx, idx);
        }
    }
    src += "\t}\n}\n";

    std::string offsets = "constexpr static unsigned int xform_offsets[] = {";
    for (int i = 0; i <= f.xforms.size(); i++) {
        if (i == f.xforms.size() && !f.final_xform.has_value()) continue;

        if (i != 0) offsets += ",";
        offsets += fmt::format("{}", xform_offsets[i]);
    }
    offsets += "};\n";

    src +=
        "__device__ unsigned int select_xform(Real ratio){\n"
        "\tratio *= state.flame[6];\n"
        "\tReal rsum = 0.0f;\n"
        "\tunsigned char last_nonzero = 0;\n"
        "\tReal cur_weight;\n";
    for (int i = 0; i < f.xforms.size(); i++) {
        src += fmt::format("\tcur_weight = state.flame[xform_offsets[{0}]];\n", i);
        if (i + 1 != f.xforms.size())
            src += fmt::format("\tif( cur_weight != 0.0 && (rsum + cur_weight) >= ratio) return {0}; else {{ rsum += cur_weight; last_nonzero={0};}}\n", i);
        else src += fmt::format("\treturn (cur_weight != 0.0)? {}: last_nonzero;\n", i);
    }

    src += "}\n";

    return offsets + src;
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
    auto [most_blocks, opts] = make_opts(prec, f);
    auto [compile_result, handle] = km.compile_file("assets/kernels/refactor.cu", opts);


    auto r = result{
    .kernel = std::nullopt,
    .source = annotate_source(opts.get_header("flame_generated.h")),
    .log = std::move(compile_result.log)
    };


    if (!compile_result.success) {
        return r;
    }

    auto func = handle();
    //handle.kernel("print_debug_info").launch(1, 1)();
    auto max_blocks = func.max_blocks_per_mp(most_blocks.block) * cuda::context::current().device().mp_count();
    if (max_blocks < most_blocks.grid) {
        SPDLOG_ERROR("Kernel for {} needs {} blocks but only got {}", f.hash().str64(), most_blocks.grid, max_blocks);
        return r;
    }
    SPDLOG_INFO("Loaded flame kernel: {} temp. samples, {} flame params, {} regs, {} shared, {} local.", max_blocks, f.real_count(), func.register_count(), func.shared_bytes(), func.local_bytes());

    r.kernel = flame_kernel{ f.hash(), std::move(handle), prec, shuf_bufs[most_blocks.block], device_mhz, std::pair<int, int>{most_blocks.grid, most_blocks.block}, catmull };

    return r;
}

bool rfkt::flame_compiler::is_cached(precision prec, const flame& f)
{
    auto [blocks, opts] = make_opts(prec, f);
    return km.is_cached(opts);
}

std::shared_ptr<rfkt::cuda_buffer<unsigned short>> make_shuffle_buffers(std::size_t particles_per_temporal_sample, std::size_t num_shuf_buf) {
    using shuf_t = unsigned short;

    auto buf = std::make_shared<rfkt::cuda_buffer<unsigned short>>( particles_per_temporal_sample * num_shuf_buf);

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

    cuMemcpyHtoD(buf->ptr(), shuf_bufs_local, sizeof(shuf_t) * particles_per_temporal_sample * num_shuf_buf);
    free(shuf_bufs_local);
    return std::move(buf);
}

rfkt::flame_compiler::flame_compiler(kernel_manager& k_manager): km(k_manager)
{
    num_shufs = 4096;

    exec_configs = cuda::context::current().device().concurrent_block_configurations();

    for (auto& exec : exec_configs) {
        shuf_bufs[exec.block] = make_shuffle_buffers(exec.block, num_shufs);

        {
            auto leftover = exec.shared_per_block - smem_per_block(precision::f32, 0, exec.block);
            if (leftover <= 0) continue;
            SPDLOG_INFO("{}x{}xf32: {:> 5} leftover shared ({:> 5} floats)", exec.grid, exec.block, leftover, leftover / 4);
        }
        {
            auto leftover = exec.shared_per_block - smem_per_block(precision::f64, 0, exec.block);
            if (leftover <= 0) continue;
            SPDLOG_INFO("{}x{}xf64: {:> 5} leftover shared ({:> 5} doubles)", exec.grid, exec.block, leftover, leftover / 8);
        }
    }

    device_mhz = 1'965'000;

    auto [res, result] = km.compile_file("assets/kernels/catmull.cu", 
        compile_opts("catmull")
        .flag(compile_flag::extra_vectorization)
        .function("generate_sample_coefficients")
    );

    if (!res.success) {
        SPDLOG_ERROR(res.log);
        exit(1);
    }
    this->catmull = std::make_shared<cuda_module>(std::move(result));
    auto func = (*catmull)();
    auto [s_grid, s_block] = func.suggested_dims();
    SPDLOG_INFO("Loaded catmull kernel: {} regs, {} shared, {} local, {}x{} suggested dims", func.register_count(), func.shared_bytes(), func.local_bytes(), s_grid, s_block);
}

auto rfkt::flame_compiler::make_opts(precision prec, const flame& f)->std::pair<cuda::execution_config, compile_opts>
{
    std::string flame_generated = create_flame_source(f);

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

    //if (most_blocks_idx > 0) most_blocks_idx--;
    auto& most_blocks = exec_configs[2];

    //flame_generated += fmt::format("__device__ unsigned int flame_size_reals() {{ return {}; }}\n", flame_real_count);
    //flame_generated += fmt::format("__device__ unsigned int flame_size_bytes() {{ return {}; }}\n", flame_size_bytes);

    auto name = fmt::format("flame_{}_f{}_t{}_s{}", flame_hash.str64(), (prec == precision::f32) ? "32" : "64", most_blocks.grid, flame_real_count);

    auto opts = compile_opts(name)
        .flag(compile_flag::extra_vectorization)
        //.flag(compile_flag::use_fast_math)
        //.flag(compile_flag::relocatable_device_code)
        //.flag(compile_flag::line_info)
        .define("NUM_XFORMS", f.xforms.size())
        .define("NUM_SHUF_BUFS", num_shufs)
        .define("THREADS_PER_BLOCK", most_blocks.block)
        .define("BLOCKS_PER_MP", most_blocks.grid / cuda::context::current().device().mp_count())
        .define("HAS_FINAL_XFORM", f.final_xform.has_value())
        .define("FLAME_SIZE_REALS", flame_real_count)
        .define("FLAME_SIZE_BYTES", flame_size_bytes)
        .header("flame_generated.h", flame_generated)
        //.link(flamelib.get())
        .function("warmup")
        .function("bin")
        .function("print_debug_info");

    if (prec == precision::f64) opts.define("DOUBLE_PRECISION");
    else opts.flag(compile_flag::use_fast_math);

    return { most_blocks, opts };
}

template<std::size_t Size>
class pinned_ring_allocator_t {
public:
    pinned_ring_allocator_t() {
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

    template<typename T, std::size_t Amount>
    auto reserve()->std::array<T, Amount>* {
        return reserve<std::array<T, Amount>>();
    }

    void* reserve(std::size_t amount) {
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
    static constexpr auto num_counters = 2;
    static constexpr auto counters_size = sizeof(counter_type) * num_counters;
    thread_local pinned_ring_allocator_t<counters_size * 128> pra{};

    struct stream_state_t {
        std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        decltype(start) end;
        std::size_t total_bins;
        std::size_t num_threads;

        CUdeviceptr qp_dev;

        std::promise<flame_kernel::bin_result> promise{};
        std::array<std::size_t, num_counters>* qp_host;

    };

    auto stream_state = new stream_state_t{};
    auto future = stream_state->promise.get_future();
    stream_state->qp_host = pra.reserve<counter_type, num_counters>();

    stream_state->total_bins = state.bin_dims.x * state.bin_dims.y;
    stream_state->num_threads = exec.first * exec.second;

    cuMemAllocAsync(&stream_state->qp_dev, counters_size, stream);
    cuMemsetD32Async(stream_state->qp_dev, 0, counters_size / sizeof(unsigned int), stream);

    auto klauncher = [&mod = this->mod, stream, &exec = this->exec]<typename ...Ts>(Ts&&... args) {
        return mod("bin").launch(exec.first, exec.second, stream, cuda::context::current().device().cooperative_supported())(std::forward<Ts>(args)...);
    };

    cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->start = std::chrono::high_resolution_clock::now();
        }, stream_state);

    CUDA_SAFE_CALL(klauncher(
        state.shared.ptr(),
        shuf_bufs->ptr(),
        (std::size_t)(target_quality * stream_state->total_bins * 255.0),
        iter_bailout,
        (long long)(ms_bailout)*device_mhz,
        state.bins.ptr(), state.bin_dims.x, state.bin_dims.y,
        stream_state->qp_dev,
        stream_state->qp_dev + sizeof(counter_type)
    ));
    CUDA_SAFE_CALL(cuMemcpyDtoHAsync(stream_state->qp_host->data(), stream_state->qp_dev, counters_size, stream));
    CUDA_SAFE_CALL(cuMemFreeAsync(stream_state->qp_dev, stream));

    cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->end = std::chrono::high_resolution_clock::now();

        ss->promise.set_value(flame_kernel::bin_result{
            ss->qp_host->at(0) / (ss->total_bins * 255.0f),
            std::chrono::duration_cast<std::chrono::nanoseconds>(ss->end - ss->start).count() / 1'000'000.0f,
            ss->qp_host->at(1),
            ss->qp_host->at(0) / 255,
            ss->total_bins,
            double(ss->qp_host->at(1)) / (ss->num_threads)
        });

        delete ss;

    }, stream_state);

    return future;
}

auto rfkt::flame_kernel::warmup(CUstream stream, const flame& f, uint2 dims, double t, std::uint32_t nseg, double loops_per_frame, std::uint32_t seed, std::uint32_t count) const -> flame_kernel::saved_state
{
    auto pack_flame = [](const flame& f, uint2 dims, auto& packer, double t) {
        auto mat = f.make_screen_space_affine(dims.x, dims.y, t);
        packer(mat.a.sample(t));
        packer(mat.d.sample(t));
        packer(mat.b.sample(t));
        packer(mat.e.sample(t));
        packer(mat.c.sample(t));
        packer(mat.f.sample(t));

        auto sum = 0.0;
        for (const auto& xf : f.xforms) sum += xf.weight.sample(t);
        packer(sum);

        auto pack_xform = [&packer, &t](const xform& xf) {
            packer(xf.weight.sample(t));
            packer(xf.color.sample(t));
            packer(xf.color_speed.sample(t));
            packer(xf.opacity.sample(t));

            for (const auto& vl : xf.vchain) {

                auto affine = vl.affine.scale(vl.aff_mod_scale.sample(t)).rotate(vl.aff_mod_rotate.sample(t)).translate(vl.aff_mod_translate.first.sample(t), vl.aff_mod_translate.second.sample(t));

                packer(affine.a.sample(t));
                packer(affine.d.sample(t));
                packer(affine.b.sample(t));
                packer(affine.e.sample(t));
                packer(affine.c.sample(t));
                packer(affine.f.sample(t));

                for (const auto& [idx, value] : vl.variations) packer(value.sample(t));
                for (const auto& [idx, value] : vl.parameters) packer(value.sample(t));
            }
        };

        for (const auto& xf : f.xforms) pack_xform(xf);
        if (f.final_xform.has_value()) pack_xform(*f.final_xform);

        for (const auto& hsv : f.palette()) {
            packer(hsv[0].sample(t));
            packer(hsv[1].sample(t));
            packer(hsv[2].sample(t));
        }
    };

    auto warmup_impl = [&]<typename Real>() -> flame_kernel::saved_state {

        const auto nreals = f.real_count() + 256 * 3;
        const auto pack_size_reals = nreals * (nseg + 3);
        const auto pack_size_bytes = sizeof(Real) * pack_size_reals;

        CUdeviceptr samples_dev;
        cuMemAllocAsync(&samples_dev, pack_size_bytes, stream);

        CUdeviceptr segments_dev;
        cuMemAllocAsync(&segments_dev, pack_size_bytes * 4 * nseg, stream);
        auto state = flame_kernel::saved_state{ dims, this->saved_state_size(), stream };
        cuMemsetD32Async(state.bins.ptr(), 0, state.bin_dims.x * state.bin_dims.y * 4, stream);

        thread_local pinned_ring_allocator_t<1024 * 1024 * 4> pra{};
        auto pack = std::span<Real>{ (Real*)pra.reserve(pack_size_bytes), pack_size_reals };
        auto packer = [pack, counter = 0](double v) mutable {
            pack[counter] = static_cast<Real>(v);
            counter++;
        };

        const auto seg_length = loops_per_frame / nseg * 1.2;
        for (int pos = -1; pos < static_cast<int>(nseg) + 2; pos++) {
            pack_flame(f, dims, packer, t + pos * seg_length);
        }

        cuMemcpyHtoDAsync(samples_dev, pack.data(), pack_size_bytes, stream);

        auto [grid, block] = this->catmull->kernel().suggested_dims();
        auto nblocks = (nseg * pack_size_reals) / block;
        if ((nseg * pack_size_reals) % block > 0) nblocks++;
        this->catmull->kernel().launch(nblocks, block, stream, false)(
            samples_dev, static_cast<std::uint32_t>(nreals), std::uint32_t{ nseg }, segments_dev
            );
        cuMemFreeAsync(samples_dev, stream);

        this->mod.kernel("warmup")
            .launch(this->exec.first, this->exec.second, stream, true)
            (
                std::uint32_t{ nseg },
                segments_dev,
                this->shuf_bufs->ptr(),
                seed, count,
                state.shared.ptr()
                );

        cuMemFreeAsync(segments_dev, stream);
        return state;
    };

    if (this->prec == precision::f32)
        return warmup_impl.operator()<float>();
    else
        return warmup_impl.operator()<double>();
}
