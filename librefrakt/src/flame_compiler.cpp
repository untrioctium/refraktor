#include <stack>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <random>

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
    std::string setup_source = "result.x = 0; result.y = 0;\n";

    auto var_offset = offset;
    setup_source += fmt::format(
        "const auto& xaffine_a = state.flame[offset + {}];\n"
        "const auto& xaffine_d = state.flame[offset + {}];\n"
        "const auto& xaffine_b = state.flame[offset + {}];\n"
        "const auto& xaffine_e = state.flame[offset + {}];\n"
        "const auto& xaffine_c = state.flame[offset + {}];\n"
        "const auto& xaffine_f = state.flame[offset + {}];\n"
        "weight = xaffine_a * p.x + xaffine_b * p.y + xaffine_c;\n"
        "   p.y = xaffine_d * p.x + xaffine_e * p.y + xaffine_f;\n"
        "   p.x = weight;\n",
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
    std::string function_header = fmt::format("template<unsigned long long offset> __device__ void xform_{}( vec2& p, vec2& result, jsf32ctx* rs)", xf.hash().str32());

    std::string xform_src = "Real weight;\n";

    int offset = 4;
    for (auto& vl : xf.vchain) {
        xform_src += fmt::format(
            "{{\n{}\n}}\n"
            "p.x = result.x;\n"
            "p.y = result.y;\n", expand_tabs(create_vlink_source(vl, offset)));
        offset += vl.real_count();
    }

    return fmt::format("{}\n{{\n{}}}\n", function_header, expand_tabs(xform_src));
}

std::string create_flame_source(const rfkt::flame* f) {
    std::string src = "#define xcommon(name) xcommon_ ## name\n";

    std::map<rfkt::hash_t, std::set<int>> shared_xforms;
    std::vector<std::size_t> xform_offsets;
    std::size_t offset_counter = 7;

    for (int i = 0; i <= f->xforms.size(); i++) {
        if (i == f->xforms.size() && !f->final_xform.has_value()) continue;
        auto& xf = (i == f->xforms.size()) ? f->final_xform.value() : f->xforms[i];
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
        "__device__ Real dispatch(int idx, iterator& in, iterator& out) {\n"
        "\tswitch(idx){\n"
        "\t\tdefault: return 0.0f;\n";

    for (auto& [hash, idxs] : shared_xforms) {
        for (auto idx : idxs) {
            src += fmt::format("\t\tcase {}:\n", idx);
            src += fmt::format(
                "\t\t{{\n"
                "\t\t\txform_{}<xform_offsets[{}]>(in.position, out.position, &my_rand()); \n"
                "\t\t\tout.color = INTERP(in.color, state.flame[xform_offsets[{}] + 1], state.flame[xform_offsets[{}] + 2]);\n"
                "\t\t\treturn state.flame[xform_offsets[{}] + 3];\n"
                "\t\t}}\n", 
                hash.str32(), idx, idx, idx, idx);
        }
    }
    src += "\t}\n}\n";

    std::string offsets = "constexpr static unsigned int xform_offsets[] = {";
    for (int i = 0; i <= f->xforms.size(); i++) {
        if (i == f->xforms.size() && !f->final_xform.has_value()) continue;

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
    for (int i = 0; i < f->xforms.size(); i++) {
        src += fmt::format("\tcur_weight = state.flame[xform_offsets[{0}]];\n", i);
        if (i + 1 != f->xforms.size())
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

auto rfkt::flame_compiler::get_flame_kernel(precision prec, const flame* f) -> result
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
        SPDLOG_ERROR("Kernel for {} needs {} blocks but only got {}", f->hash().str64(), most_blocks.grid, max_blocks);
        return r;
    }
    SPDLOG_INFO("Loaded flame kernel: {} temp. samples, {} flame params, {} regs, {} shared, {} local.", max_blocks, f->real_count(), func.register_count(), func.shared_bytes(), func.local_bytes());

    r.kernel = flame_kernel{ f->hash(), std::move(handle), prec, shuf_bufs[most_blocks.block], device_mhz, std::pair<int, int>{most_blocks.grid, most_blocks.block}, catmull };

    return r;
}

/*auto rfkt::flame_compiler::get_flame_kernel_async(precision prec, const flame* f) -> std::future<decltype(get_flame_kernel_sync(prec, f))>
{
    auto ctx = cuda::context::current();
    return std::async(std::launch::async, [ctx, fc = this, prec, &f]() {
        ctx.make_current();
        return fc->get_flame_kernel_sync(prec, f);
    });
}*/

bool rfkt::flame_compiler::is_cached(precision prec, const flame* f)
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

   /* auto [result, lib] = km->ptx_from_file("assets/kernels/flamelib.cu", compile_opts("flamelib").flag(compile_flag::relocatable_device_code));

    if (!result.success) {
        SPDLOG_ERROR("Error compiling flamelib:\n{}", result.log);
        exit(1);
    }
    flamelib = lib;*/

    CUdevprop props;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    cuDeviceGetProperties(&props, dev);
    device_mhz = 1'900'000;

    auto [res, result] = km.compile_file("assets/kernels/catmull.cu", 
        compile_opts("catmull")
        .flag(compile_flag::extra_vectorization)
        .flag(compile_flag::use_fast_math)
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

auto rfkt::flame_compiler::make_opts(precision prec, const flame* f)->std::pair<cuda::execution_config, compile_opts>
{
    std::string flame_generated = create_flame_source(f);

    auto flame_real_count = f->real_count();
    auto flame_size_bytes = ((prec == precision::f32) ? sizeof(float) : sizeof(double)) * flame_real_count;
    auto flame_hash = f->hash();

    auto most_blocks_idx = 0;
    for (int i = exec_configs.size() - 1; i >= 0; i--) {
        auto& ec = exec_configs[i];
        if (smem_per_block(prec, flame_size_bytes, ec.block) <= ec.shared_per_block) {
            most_blocks_idx = i;
            break;
        }
    }

    if (most_blocks_idx > 0) most_blocks_idx--;
    auto& most_blocks = exec_configs[most_blocks_idx];

    //flame_generated += fmt::format("__device__ unsigned int flame_size_reals() {{ return {}; }}\n", flame_real_count);
    //flame_generated += fmt::format("__device__ unsigned int flame_size_bytes() {{ return {}; }}\n", flame_size_bytes);

    auto name = fmt::format("flame_{}_f{}_t{}_s{}", flame_hash.str64(), (prec == precision::f32) ? "32" : "64", most_blocks.grid, flame_real_count);

    auto opts = compile_opts(name)
        .flag(compile_flag::extra_vectorization)
        //.flag(compile_flag::use_fast_math)
        //.flag(compile_flag::relocatable_device_code)
        //.flag(compile_flag::line_info)
        .define("NUM_XFORMS", f->xforms.size())
        .define("NUM_SHUF_BUFS", num_shufs)
        .define("THREADS_PER_BLOCK", most_blocks.block)
        .define("BLOCKS_PER_MP", most_blocks.grid / cuda::context::current().device().mp_count())
        .define("HAS_FINAL_XFORM", f->final_xform.has_value())
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

/*template<typename Real>
struct vec2 {

    Real x;
    Real y;

    constexpr auto operator+(const vec2& rhs) const { return vec2<Real>{ x + rhs.x, y + rhs.y }; }
    constexpr auto operator-(const vec2& rhs) const { return vec2<Real>{ x - rhs.x, y - rhs.y }; }
    constexpr auto operator*(Real rhs) const { return vec2<Real>{ x* rhs, y* rhs }; }
    constexpr auto operator/(Real rhs) const { return vec2<Real>{ x / rhs, y / rhs }; }
    constexpr auto length() const { return hypot(x, y); }

};

template<typename Real>
constexpr auto operator*(Real lhs, const vec2<Real>& rhs) { return vec2<Real>{ lhs* rhs.x, lhs* rhs.y }; }

template<typename Real>
struct segment {
    vec2<Real> a;
    vec2<Real> b;
    vec2<Real> c;
    vec2<Real> d;

    constexpr auto sample(Real t) const {
        return a * t * t * t + b * t * t + c * t + d;
    }
};

template<typename Real>
constexpr auto generate_segment(const vec2<Real>& p0, const vec2<Real>& p1, const vec2<Real>& p2, const vec2<Real>& p3, Real tension, Real alpha) -> segment<Real> {

    Real t01 = pow((p1 - p0).length(), alpha);
    Real t12 = pow((p2 - p1).length(), alpha);
    Real t23 = pow((p3 - p2).length(), alpha);

    vec2<Real> m1 = (Real(1.0) - tension) *
        (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)));
    vec2<Real> m2 = (Real(1.0) - tension) *
        (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)));

    return segment<Real> {
        Real(2.0)* (p1 - p2) + m1 + m2,
        Real(-3.0)* (p1 - p2) - m1 - m1 - m2,
        m1,
        p1
    };
}*/

/*void rfkt::flame_kernel::bin(CUstream stream, const std::vector<flame>* samples, render_surface& rs, float target_quality, std::uint32_t ms_bailout, int iter_bailout, int warmup, std::uint32_t seed, thread_pool* pool, rfkt::callback<flame_kernel::bin_result>&& on_complete) const {

    auto stream_state = std::make_unique<stream_state_t>();
    stream_state->on_complete = std::move(on_complete);

    CUdeviceptr p_buf;
    CUdeviceptr qp_dev;
    CUdeviceptr coeff_buf;

    cuMemAllocAsync(&p_buf, 768 * samples->size(), stream);
    cuMemAllocAsync(&qp_dev, sizeof(unsigned long long) * 2, stream);
    cuMemsetD32Async(qp_dev, 0, sizeof(unsigned long long) * 2 / sizeof(unsigned int), stream);

    auto pack_pusher = []<typename Real>(const std::vector<flame>*samples, render_surface & rs, CUstream stream, kernel catmull) {
        auto pack = new std::vector<Real>{};
        unsigned int sample_size = samples->at(0).pack_length();
        unsigned int pack_size = sample_size * samples->size();

        if (samples->size() < 4) { throw(1); }

        unsigned int num_segments = samples->size() - 3;

        pack->reserve(pack_size);

        for (auto& f : *samples) {
            f.make_screen_space_affine(rs.width, rs.height).pack<Real>(*pack);
            f.pack<Real>(*pack);
        }

        CUdeviceptr f_buf;
        CUDA_SAFE_CALL(cuMemAllocAsync(&f_buf, pack_size * sizeof(Real), stream));
        CUDA_SAFE_CALL(cuMemcpyHtoDAsync(f_buf, pack->data(), sizeof(Real) * pack_size, stream));

        // keep the pack alive until the copy is complete
        cuLaunchHostFunc(stream, [](auto* data)
            {
                auto pack = (std::vector<Real>*) data;
                delete pack;
            }, pack);

        CUdeviceptr coeff_buf;
        CUDA_SAFE_CALL(cuMemAllocAsync(&coeff_buf, sample_size * 4 * sizeof(Real) * num_segments, stream));

        const auto num_threads = sample_size * num_segments;
        const auto [grid, block] = catmull.suggested_dims();

        CUDA_SAFE_CALL(catmull.launch(num_threads / block + ((num_threads % block > 0) ? 1 : 0), block)(f_buf, sample_size, num_segments, coeff_buf));
        CUDA_SAFE_CALL(cuMemFreeAsync(f_buf, stream));


        return coeff_buf;
    };

    if (prec == precision::f32) {
        coeff_buf = pack_pusher.operator()<float>(samples, rs, stream, (*catmull)());
    }
    else {
        coeff_buf = pack_pusher.operator()<double>(samples, rs, stream, (*catmull)());
    }

    auto pal_offset = 0u;

    for (auto& f : *samples) {
        CUDA_SAFE_CALL(cuMemcpyHtoDAsync(p_buf + pal_offset, f.palette_rgb.data(), 768, stream));
        pal_offset += 768;
    }

    stream_state->total_bins = std::size_t{ rs.width } *rs.height;
    stream_state->num_threads = exec.first * exec.second;

    auto klauncher = [&mod = this->mod, stream, &exec = this->exec](auto&&... args) {
        if (cuda::context::current().device().cooperative_supported()) {
            return mod().launch_cooperative(exec.first, exec.second, stream)(args...);
        }
        else
            return mod().launch(exec.first, exec.second, stream)(args...);
    };

    CUDA_SAFE_CALL(klauncher(
        coeff_buf,
        (unsigned int)(samples->size() - 3),
        p_buf,
        shuf_bufs->ptr(),
        (unsigned long long)(target_quality * rs.width * rs.height * 255ull),
        warmup,
        iter_bailout,
        (long long)(ms_bailout)*device_mhz,
        seed,
        rs.bins.ptr(),
        int(rs.width),
        int(rs.height),
        qp_dev,
        qp_dev + sizeof(unsigned long long)
        ));

    rfkt::cuda::host_func(stream, *pool,
        [stream_state = std::move(stream_state), qp_dev]()
    {
        unsigned long long qp_host[2];
        CUDA_SAFE_CALL(cuMemcpyDtoH(qp_host, qp_dev, sizeof(unsigned long long) * 2));
        CUDA_SAFE_CALL(cuMemFree(qp_dev));

        stream_state->end = std::chrono::high_resolution_clock::now();
        stream_state->on_complete(flame_kernel::bin_result{
            qp_host[0] / (stream_state->total_bins * 255.0f),
            std::chrono::duration_cast<std::chrono::nanoseconds>(stream_state->end - stream_state->start).count() / 1'000'000.0f,
            qp_host[1],
            double(qp_host[1]) / (stream_state->num_threads),
            });

    });
    CUDA_SAFE_CALL(cuMemFreeAsync(coeff_buf, stream));
    CUDA_SAFE_CALL(cuMemFreeAsync(p_buf, stream));


}*/

auto rfkt::flame_kernel::bin(CUstream stream, flame_kernel::saved_state & state, float target_quality, std::uint32_t ms_bailout, std::uint32_t iter_bailout) const -> bin_result
{
    struct stream_state_t {
        std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        decltype(start) end;
        std::size_t total_bins;
        std::size_t num_threads;

        CUdeviceptr qp_dev;
    } stream_state;
    stream_state.total_bins = state.bin_dims.x * state.bin_dims.y;
    stream_state.num_threads = exec.first * exec.second;

    using counter_type = std::size_t;
    constexpr auto num_counters = 2;
    constexpr auto counters_size = sizeof(counter_type) * num_counters;
    cuMemAllocAsync(&stream_state.qp_dev, counters_size, stream);
    cuMemsetD32Async(stream_state.qp_dev, 0, counters_size / sizeof(unsigned int), stream);

    auto klauncher = [&mod = this->mod, stream, &exec = this->exec]<typename ...Ts>(Ts&&... args) {
        return mod("bin").launch(exec.first, exec.second, stream, cuda::context::current().device().cooperative_supported())(std::forward<Ts>(args)...);
    };

    cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->start = std::chrono::high_resolution_clock::now();
        }, & stream_state);
    CUDA_SAFE_CALL(klauncher(
        state.shared.ptr(),
        shuf_bufs->ptr(),
        (std::size_t)(target_quality * stream_state.total_bins * 255.0),
        iter_bailout,
        (long long)(ms_bailout)*device_mhz,
        state.bins.ptr(), state.bin_dims.x, state.bin_dims.y,
        stream_state.qp_dev,
        stream_state.qp_dev + sizeof(counter_type)
    ));
    cuLaunchHostFunc(stream, [](void* ptr) {
        auto* ss = (stream_state_t*)ptr;
        ss->end = std::chrono::high_resolution_clock::now();
        }, & stream_state);

    cuStreamSynchronize(stream);
    std::size_t qp_host[num_counters] = {};
    CUDA_SAFE_CALL(cuMemcpyDtoH(qp_host, stream_state.qp_dev, counters_size));
    CUDA_SAFE_CALL(cuMemFree(stream_state.qp_dev));

    //stream_state.end = 
    return flame_kernel::bin_result{
        qp_host[0] / (stream_state.total_bins * 255.0f),
        std::chrono::duration_cast<std::chrono::nanoseconds>(stream_state.end - stream_state.start).count() / 1'000'000.0f,
        qp_host[1],
        qp_host[0] / 255,
        stream_state.total_bins,
        double(qp_host[1]) / (stream_state.num_threads)
    };
}
