#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <librefrakt/util/filesystem.hpp>
#include <librefrakt/flame_types.hpp>
#include <librefrakt/flame_info.hpp>
#include <librefrakt/anima.hpp>
#include <librefrakt/flame_compiler.hpp>
#include <librefrakt/util/cuda.hpp>
#include <librefrakt/util/stb.hpp>

#include <librefrakt/interface/denoiser.hpp>
#include <librefrakt/interface/jpeg_encoder.hpp>
#include <librefrakt/image/tonemapper.hpp>
#include <librefrakt/image/converter.hpp>

#include <sqlite3.h>

#include <print>

namespace py = pybind11;

struct context {
    std::unique_ptr<rfkt::flamedb> flamedb;
    std::unique_ptr<rfkt::function_table> functions;
    std::unique_ptr<roccu::context> cuda_ctx;
    std::unique_ptr<rfkt::flame_compiler> flame_compiler;
    std::shared_ptr<ezrtc::compiler> kernel_manager;

    std::unique_ptr<rfkt::denoiser> denoiser;
    std::unique_ptr<rfkt::denoiser> upscaling_denoiser;
    std::unique_ptr<rfkt::tonemapper> tonemapper;
    std::unique_ptr<rfkt::converter> converter;
    std::unique_ptr<rfkt::jpeg_encoder> jpeg_encoder;
    std::unique_ptr<roccu::gpu_stream> stream;
    std::unique_ptr<roccu::gpu_event> event;

    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db = {nullptr, sqlite3_close};

    std::map<rfkt::hash_t, rfkt::flame_compiler::result> cached_kernels;
};

static std::unique_ptr<context> ctx = nullptr;

#define EXPOSE_VECTOR2_TYPE(Base, T) \
    py::class_<T>(m, #T) \
        .def(py::init<Base>()) \
        .def_readwrite("x", &T::x) \
        .def_readwrite("y", &T::y)

#define EXPOSE_VECTOR3_TYPE(Base, T) \
    py::class_<T>(m, #T) \
        .def(py::init<Base>()) \
        .def_readwrite("x", &T::x) \
        .def_readwrite("y", &T::y) \
        .def_readwrite("z", &T::z)

#define EXPOSE_VECTOR4_TYPE(Base, T) \
    py::class_<T>(m, #T) \
        .def(py::init<Base>()) \
        .def_readwrite("x", &T::x) \
        .def_readwrite("y", &T::y) \
        .def_readwrite("z", &T::z) \
        .def_readwrite("w", &T::w)

rfkt::flame_kernel::bin_result render_image(const rfkt::flame& flame, std::string_view output_path, unsigned int width, unsigned int height, double t, double fps, double seconds_per_loop, double quality_bailout, unsigned int millis_bailout, bool denoise, std::set<std::string> flags, bool superscale, unsigned int min_warps_per_block) {

    py::gil_scoped_release release;
    ctx->cuda_ctx->make_current();


    auto dev_l2 = roccu::context::current().device().l2_cache_size();
    auto bins_width = width;
    auto bins_height = height;

    if(superscale) {
        // find the largest integer multiple that fits in the L2 cache
        constexpr static auto normal_bytes_per_bin = 16;
        constexpr static auto hot_cold_bytes_per_bin = 8;

        auto bins_size_normal = bins_width * bins_height * normal_bytes_per_bin;
        auto bins_size_hot_cold = bins_width * bins_height * hot_cold_bytes_per_bin;

        SPDLOG_INFO("Device L2: {:.2f}MB", dev_l2 / 1024.0 / 1024);
        SPDLOG_INFO("Normal size: {:.2f}MB", bins_size_normal / 1024.0 / 1024);
        SPDLOG_INFO("Hot/cold size: {:.2f}MB", bins_size_hot_cold / 1024.0 / 1024);

        auto bins_normal_multiple = static_cast<unsigned int>(std::sqrt(dev_l2 / bins_size_normal));
        auto bins_hot_cold_multiple = static_cast<unsigned int>(std::sqrt(dev_l2 / bins_size_hot_cold));

        SPDLOG_INFO("Normal multiple: {}", bins_normal_multiple);
        SPDLOG_INFO("Hot/cold multiple: {}", bins_hot_cold_multiple);

        if(bins_hot_cold_multiple > bins_normal_multiple) {
            bins_width *= bins_hot_cold_multiple;
            bins_height *= bins_hot_cold_multiple;
            flags.insert("HOT_COLD");
        } else if(bins_normal_multiple > 0) {
            bins_width *= bins_normal_multiple;
            bins_height *= bins_normal_multiple;
        }
    }

    SPDLOG_INFO("Bins size: {}x{}", bins_width, bins_height);

    auto compile_result = ctx->flame_compiler->get_flame_kernel(*ctx->flamedb, rfkt::precision::f32, flame, flags, min_warps_per_block);

    if (!compile_result.kernel) {
        throw std::runtime_error(compile_result.log);
    }

    if(compile_result.log.size() > 0) {
        SPDLOG_ERROR("Compilation log: {}", compile_result.log);
    }

    auto loops_per_frame = 1.0 / (fps * seconds_per_loop);

    auto samples = std::vector<double>{};
    auto packer = [&samples](double v) { samples.push_back(v); };
    auto invoker = ctx->functions->make_invoker();
    auto offset = 1.2 * loops_per_frame;
    flame.pack_samples(packer, invoker, t - offset * loops_per_frame, offset, 4, bins_width, bins_height);

    auto state = compile_result.kernel->warmup(*ctx->stream, samples, rfkt::uint2{bins_width, bins_height}, 0xdeadbeef, 100);

    auto quality_scale = state.cold_bins.area() / (width * height);
    auto bin_result = compile_result.kernel->bin(*ctx->stream, state, {.millis = millis_bailout, .quality = quality_bailout / quality_scale}).get();

    auto tonemapped = roccu::gpu_image<rfkt::half3>(width, height, *ctx->stream);
    auto denoised = roccu::gpu_image<rfkt::half3>(width, height, *ctx->stream);
    auto converted = roccu::gpu_image<rfkt::uchar3>(width, height, *ctx->stream);

    auto tm_args = rfkt::tonemapper::args_t{
        .quality = bin_result.quality * quality_scale,
        .gamma = flame.gamma.sample(t, invoker),
        .brightness = flame.brightness.sample(t, invoker),
        .vibrancy = flame.vibrancy.sample(t, invoker),
        .hdr = false
    };

    ctx->tonemapper->run(state.cold_bins, state.hot_bins, tonemapped, tm_args, *ctx->stream);
    if(denoise) {
        ctx->denoiser->denoise(tonemapped, denoised, *ctx->event);
    } else {
        denoised = std::move(tonemapped);
    }
    ctx->converter->to_uchar3(denoised, converted, *ctx->stream);
    auto fut = ctx->jpeg_encoder->encode_image(converted, 100, *ctx->stream);
    auto output = fut.get()();
    rfkt::fs::write(rfkt::fs::path(output_path), (const char*)output.data(), output.size());

    return bin_result;
}

rfkt::flame_kernel::bin_result render_image_interpolated(const rfkt::interpolator& interpolator, double mix, std::string_view output_path, unsigned int width, unsigned int height, double t, double fps, double seconds_per_loop, double quality_bailout, unsigned int millis_bailout, bool denoise, bool upscale, std::set<std::string> flags, std::uint32_t iter_bailout, unsigned int min_warps_per_block) {
    py::gil_scoped_release release;
    ctx->cuda_ctx->make_current();

    auto hash = interpolator.left_flame().hash();
    auto cache_search = ctx->cached_kernels.find(hash);

    if(cache_search == ctx->cached_kernels.end()) {
        auto compile_result = ctx->flame_compiler->get_flame_kernel(*ctx->flamedb, rfkt::precision::f32, interpolator.left_flame(), flags, min_warps_per_block);
        if (!compile_result.kernel) {
            throw std::runtime_error(compile_result.log);
        }
        cache_search = ctx->cached_kernels.emplace(hash, std::move(compile_result)).first;
    }

    auto& compile_result = cache_search->second;

    unsigned int out_width = width;
    unsigned int out_height = height;

    if(upscale) {
        width /= 2;
        height /= 2;
    }

    auto loops_per_frame = 1.0 / (fps * seconds_per_loop);

    auto samples = std::vector<double>{};
    auto packer = [&samples](double v) { samples.push_back(v); };
    auto invoker = ctx->functions->make_invoker();
    auto offset = 1.1 * loops_per_frame;
    interpolator.pack_samples(packer, invoker, t - offset, offset, 4, width, height, mix);

    auto state = compile_result.kernel->warmup(*ctx->stream, samples, rfkt::uint2{width, height}, 0xdeadbeef, 100);
    auto bin_result = compile_result.kernel->bin(*ctx->stream, state, {.millis = millis_bailout, .quality = quality_bailout, .iters = iter_bailout}).get();

    auto tonemapped = roccu::gpu_image<rfkt::half3>(width, height, *ctx->stream);
    auto denoised = roccu::gpu_image<rfkt::half3>(out_width, out_height, *ctx->stream);
    auto converted = roccu::gpu_image<rfkt::uchar3>(out_width, out_height, *ctx->stream);

    auto tm_args = rfkt::tonemapper::args_t{
        .quality = bin_result.quality,
        .gamma = interpolator.interp_anima(&rfkt::flame::gamma, invoker, t, mix),
        .brightness = interpolator.interp_anima(&rfkt::flame::brightness, invoker, t, mix),
        .vibrancy = interpolator.interp_anima(&rfkt::flame::vibrancy, invoker, t, mix),
        .hdr = false
    };

    ctx->tonemapper->run(state.cold_bins, state.hot_bins, tonemapped, tm_args, *ctx->stream);
    if(denoise) {
        auto& dn = upscale ? ctx->upscaling_denoiser : ctx->denoiser;
        dn->denoise(tonemapped, denoised, *ctx->event);
    } else {
        denoised = std::move(tonemapped);
    }
    ctx->converter->to_uchar3(denoised, converted, *ctx->stream);
    auto fut = ctx->jpeg_encoder->encode_image(converted, 100, *ctx->stream);
    auto output = fut.get()();
    rfkt::fs::write(rfkt::fs::path(output_path), (const char*)output.data(), output.size());

    return bin_result;
}

auto make_histogram(const rfkt::flame& flame, unsigned int width, unsigned int height, double t, double fps, double seconds_per_loop, double quality_bailout, unsigned int millis_bailout, unsigned int iters_bailout, unsigned int warmup_iterations,std::uint32_t seed, std::set<std::string> flags, unsigned int min_warps_per_block) 
    -> std::tuple<py::array_t<float>, rfkt::flame_kernel::bin_result> {

        ctx->cuda_ctx->make_current();
    auto compile_result = ctx->flame_compiler->get_flame_kernel(*ctx->flamedb, rfkt::precision::f32, flame, flags, min_warps_per_block);

    if (!compile_result.kernel) {
        throw std::runtime_error(compile_result.log);
    }

    auto loops_per_frame = 1.0 / (fps * seconds_per_loop);

    auto samples = std::vector<double>{};
    auto packer = [&samples](double v) { samples.push_back(v); };
    auto invoker = ctx->functions->make_invoker();
    auto offset = 1.2 * loops_per_frame;
    flame.pack_samples(packer, invoker, t - offset * loops_per_frame, offset, 4, width, height);

    auto state = compile_result.kernel->warmup(*ctx->stream, samples, rfkt::uint2{width, height}, 0xdeadbeef, warmup_iterations);
    auto bin_result = compile_result.kernel->bin(*ctx->stream, state, {.millis = millis_bailout, .quality = quality_bailout, .iters = iters_bailout}).get();

    auto local = new std::vector<rfkt::float4>(width * height);
    state.cold_bins.to_host(*local);

    auto capsule = py::capsule(local, [](void* ptr) { delete static_cast<std::vector<rfkt::float4>*>(ptr); });
    return {
        py::array_t<float>(
            {height, width, 4u},
            {width * 4 * sizeof(float), 4 * sizeof(float), sizeof(float)},
            (float*)local->data(),
            capsule
        ),
        bin_result
    };
}

auto make_benchmark_samples(const rfkt::flame& flame, unsigned int width, unsigned int height, double t, double fps, double seconds_per_loop, double quality_bailout, unsigned int millis_bailout, std::set<std::string> flags, int nsamples, std::size_t iter_bailout, unsigned int min_warps_per_block) -> std::vector<rfkt::flame_kernel::bin_result> {

    py::gil_scoped_release release;

    auto compile_result = ctx->flame_compiler->get_flame_kernel(*ctx->flamedb, rfkt::precision::f32, flame, flags, min_warps_per_block);

    if (!compile_result.kernel) {
        throw std::runtime_error(compile_result.log);
    }

    auto loops_per_frame = 1.0 / (fps * seconds_per_loop);

    auto samples = std::vector<double>{};
    auto packer = [&samples](double v) { samples.push_back(v); };
    auto invoker = ctx->functions->make_invoker();
    auto offset = 1.2 * loops_per_frame;
    flame.pack_samples(packer, invoker, t - offset * loops_per_frame, offset, 4, width, height);

    auto state = compile_result.kernel->warmup(*ctx->stream, samples, rfkt::uint2{width, height}, 0xdeadbeef, 100);

    auto bin_futures = std::vector<std::future<rfkt::flame_kernel::bin_result>>{};
    for(int i = 0; i < nsamples + 1; i++) {
        bin_futures.push_back(compile_result.kernel->bin(*ctx->stream, state, {.millis = millis_bailout, .quality = quality_bailout, .iters = static_cast<std::uint32_t>(iter_bailout)}));
    }

    auto bin_results = std::vector<rfkt::flame_kernel::bin_result>{};
    for(auto& future : bin_futures) {
        bin_results.push_back(future.get());
    }

    // discard the first sample
    bin_results.erase(bin_results.begin());

    return bin_results;
}



PYBIND11_MODULE(_pyrefrakt, m, py::mod_gil_not_used()) {

    using namespace rfkt;

    m.def("enable_logging", [](bool enable) {
        spdlog::set_level(enable ? spdlog::level::info : spdlog::level::off);
    });

    m.def("initialize", [](const std::string& config_path, const std::string& assets_path) {

        //spdlog::set_level(spdlog::level::off);

        if (ctx) {
            return;
        }

        ctx = std::make_unique<context>();

        SPDLOG_INFO("Initializing CUDA");
        ctx->cuda_ctx = std::make_unique<roccu::context>(rfkt::cuda::init());
    
        SPDLOG_INFO("Initializing flame database");
        ctx->flamedb = std::make_unique<rfkt::flamedb>();
        rfkt::fs::set_assets_directory(rfkt::fs::path(assets_path));
        rfkt::initialize(*ctx->flamedb, config_path);

        SPDLOG_INFO("Initializing function table");
        ctx->functions = std::make_unique<rfkt::function_table>();
        ctx->functions->add_or_update("increase", {
            {{"per_loop", {rfkt::func_info::arg_t::decimal, 360.0}}},
            "return iv + t * per_loop"
        });
        ctx->functions->add_or_update("sine", {
            {
                {"frequency", {rfkt::func_info::arg_t::decimal, 1.0}},
                {"amplitude", {rfkt::func_info::arg_t::decimal, 1.0}},
                {"phase", {rfkt::func_info::arg_t::decimal, 0.0}},
                {"sharpness", {rfkt::func_info::arg_t::decimal, 0.0}},
                {"absolute", {rfkt::func_info::arg_t::boolean, false}}
            },
            "local v = math.sin(t * frequency * math.pi * 2.0 + math.rad(phase))\n"
            "if sharpness > 0 then v = math.copysign(1.0, v) * (math.abs(v) ^ sharpness) end\n"
            "if absolute then v = math.abs(v) end\n"
            "return iv + v * amplitude\n"
        });


        SPDLOG_INFO("Initializing kernel cache");
        auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>((rfkt::fs::user_local_directory() / "kernel.sqlite3").string());
        auto guarded = std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache);
        auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(guarded);
        ctx->kernel_manager = std::make_shared<ezrtc::compiler>(zlib);


        SPDLOG_INFO("Initializing flame compiler");
        ctx->flame_compiler = std::make_unique<rfkt::flame_compiler>(ctx->kernel_manager.get());


        ctx->stream = std::make_unique<roccu::gpu_stream>();
        ctx->event = std::make_unique<roccu::gpu_event>();


        ctx->tonemapper = std::make_unique<rfkt::tonemapper>(*ctx->kernel_manager);
        ctx->converter = std::make_unique<rfkt::converter>(*ctx->kernel_manager);

        ctx->denoiser = rfkt::denoiser::make("rfkt::optix_denoise", uint2{1024, 1024}, rfkt::denoiser_flag::tiled, *ctx->stream);
        ctx->upscaling_denoiser = rfkt::denoiser::make("rfkt::optix_denoise", uint2{1024, 1024}, rfkt::denoiser_flag::upscale | rfkt::denoiser_flag::tiled, *ctx->stream);
        ctx->jpeg_encoder = rfkt::jpeg_encoder::make("rfkt::nvjpeg_encode", *ctx->stream);

        if(!ctx->jpeg_encoder) {
            throw std::runtime_error("Failed to initialize jpeg encoder");
        }

        sqlite3* dbptr = nullptr;
        if(sqlite3_open(fmt::format("{}/sheep.db", assets_path).c_str(), &dbptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to open database");
        }
        ctx->db = {dbptr, &sqlite3_close};
    });

    m.def("import_flam3", [](std::string_view path) {
        py::gil_scoped_release release;
        auto data = rfkt::fs::read_string(rfkt::fs::path(path));
        auto result = rfkt::import_flam3(*ctx->flamedb, data);
        if(!result) {
            throw std::runtime_error(result.error());
        }
        return std::move(result.value());
    });

    m.def("get_sheep", [](int gen, int idx) -> rfkt::flame {
        py::gil_scoped_release release;
        sqlite3_stmt* stmt;
        constexpr static std::string_view sql = "SELECT content FROM flames WHERE generation = ? AND id = ?";
        if(sqlite3_prepare_v2(ctx->db.get(), sql.data(), sql.size(), &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare statement");
        }
        sqlite3_bind_int(stmt, 1, gen);
        sqlite3_bind_int(stmt, 2, idx);
        if(sqlite3_step(stmt) != SQLITE_ROW) {
            throw std::runtime_error(fmt::format("No flame found for generation {} and id {}", gen, idx));
        }


        auto content = std::string_view{reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)), static_cast<std::size_t>(sqlite3_column_bytes(stmt, 0))};
        auto result = rfkt::import_flam3(*ctx->flamedb, content);
        if(!result) {
            throw std::runtime_error(result.error());
        }
        return std::move(result.value());
    });

    m.def("precompile", [](const rfkt::flame& flame, std::set<std::string> flags, unsigned int min_warps_per_block) {
        py::gil_scoped_release release;
        ctx->cuda_ctx->make_current();
        ctx->flame_compiler->get_flame_kernel(*ctx->flamedb, rfkt::precision::f32, flame, flags, min_warps_per_block);
    });

    m.def("interpolate", [](const rfkt::flame& left, const rfkt::flame& right, bool by_weight) {
        py::gil_scoped_release release;
        return rfkt::interpolator(left, right, *ctx->flamedb, by_weight);
    });

    m.def("make_source", [](const rfkt::flame& flame) {
        py::gil_scoped_release release;
        return ctx->flame_compiler->make_source(*ctx->flamedb, flame);
    });

    m.def("render_image", &render_image,
        py::arg("flame"),
        py::arg("output_path"),
        py::arg("width"),
        py::arg("height"),
        py::arg("t") = 0.0,
        py::arg("fps") = 30.0,
        py::arg("seconds_per_loop") = 5.0,
        py::arg("quality_bailout") = 128.0,
        py::arg("millis_bailout") = 2000,
        py::arg("denoise") = true,
        py::arg("flags") = std::set<std::string>{},
        py::arg("superscale") = false,
        py::arg("min_warps_per_block") = 4
    );

    m.def("render_image_interpolated", &render_image_interpolated,
        py::arg("interpolator"),
        py::arg("mix"),
        py::arg("output_path"),
        py::arg("width"),
        py::arg("height"),
        py::arg("t") = 0.0,
        py::arg("fps") = 30.0,
        py::arg("seconds_per_loop") = 5.0,
        py::arg("quality_bailout") = 128.0,
        py::arg("millis_bailout") = 2000,
        py::arg("denoise") = true,
        py::arg("upscale") = false,
        py::arg("flags") = std::set<std::string>{},
        py::arg("iter_bailout") = 4'000'000'000,
        py::arg("min_warps_per_block") = 4);

    m.def("make_histogram", &make_histogram,
        py::arg("flame"),
        py::arg("width"),
        py::arg("height"),
        py::arg("t") = 0.0,
        py::arg("fps") = 30.0,
        py::arg("seconds_per_loop") = 5.0,
        py::arg("quality_bailout") = 128.0,
        py::arg("millis_bailout") = 2000,
        py::arg("iters_bailout") = 4'000'000'000,
        py::arg("warmup_iterations") = 100,
        py::arg("seed") = 0xdeadbeef,
        py::arg("flags") = std::set<std::string>{},
        py::arg("min_warps_per_block") = 4);

    m.def("make_benchmark_samples", &make_benchmark_samples,
        py::arg("flame"),
        py::arg("width"),
        py::arg("height"),
        py::arg("t") = 0.0,
        py::arg("fps") = 30.0,
        py::arg("seconds_per_loop") = 5.0,
        py::arg("quality_bailout") = 128.0,
        py::arg("millis_bailout") = 2000,
        py::arg("flags") = std::set<std::string>{},
        py::arg("nsamples") = 10,
        py::arg("iter_bailout") = 4'000'000'000,
        py::arg("min_warps_per_block") = 4);

    EXPOSE_VECTOR2_TYPE(int, int2);
    EXPOSE_VECTOR3_TYPE(int, int3);
    EXPOSE_VECTOR4_TYPE(int, int4);
    EXPOSE_VECTOR2_TYPE(unsigned int, uint2);
    EXPOSE_VECTOR3_TYPE(unsigned int, uint3);
    EXPOSE_VECTOR4_TYPE(unsigned int, uint4);
    EXPOSE_VECTOR2_TYPE(float, float2);
    EXPOSE_VECTOR3_TYPE(float, float3);
    EXPOSE_VECTOR4_TYPE(float, float4);
    EXPOSE_VECTOR2_TYPE(double, double2);
    EXPOSE_VECTOR3_TYPE(double, double3);
    EXPOSE_VECTOR4_TYPE(double, double4);

    py::class_<rfkt::flame_kernel::bin_result>(m, "bin_result")
        .def_readwrite("quality", &rfkt::flame_kernel::bin_result::quality)
        .def_readwrite("elapsed_ms", &rfkt::flame_kernel::bin_result::elapsed_ms)
        .def_readwrite("total_passes", &rfkt::flame_kernel::bin_result::total_passes)
        .def_readwrite("total_draws", &rfkt::flame_kernel::bin_result::total_draws)
        .def_readwrite("total_bins", &rfkt::flame_kernel::bin_result::total_bins)
        .def_readwrite("passes_per_thread", &rfkt::flame_kernel::bin_result::passes_per_thread)
        .def_readwrite("max_density", &rfkt::flame_kernel::bin_result::max_density)
        .def_readwrite("max_sm_time", &rfkt::flame_kernel::bin_result::max_sm_time);

    py::class_<rfkt::interpolator>(m, "interpolator");

    py::class_<rfkt::anima>(m, "anima")
        .def(py::init<double>())
        .def("serialize", [](const rfkt::anima& a) { return a.serialize().dump(); })
        .def_readwrite("t0", &rfkt::anima::t0);

    py::class_<rfkt::affine>(m, "affine")
        .def(py::init<double, double, double, double, double, double>())
        .def("serialize", [](const rfkt::affine& a) { return a.serialize().dump(); })
        .def_readwrite("a", &rfkt::affine::a)
        .def_readwrite("b", &rfkt::affine::b)
        .def_readwrite("c", &rfkt::affine::c)
        .def_readwrite("d", &rfkt::affine::d)
        .def_readwrite("e", &rfkt::affine::e)
        .def_readwrite("f", &rfkt::affine::f)
        .def("rotated", &rfkt::affine::rotated)
        .def("scaled", &rfkt::affine::scaled)
        .def("translated", [](rfkt::affine& a, double x, double y) { return a.translated(x, y); })
        .def_static("identity", &rfkt::affine::identity)
        .def("lookup", &rfkt::affine::lookup, py::return_value_policy::reference_internal);

    py::class_<rfkt::vardata>(m, "vardata")
        .def("__getitem__", [](rfkt::vardata& v, std::string_view name) { return v[name]; }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](rfkt::vardata& v, std::string_view name, const rfkt::anima& value) { v[name] = value; })
        .def("serialize", [](const rfkt::vardata& v) { return v.serialize().dump(); })
        .def("lookup", &rfkt::vardata::lookup, py::return_value_policy::reference_internal)
        .def("__iter__", [](rfkt::vardata& v) { return py::make_iterator(v.begin(), v.end()); }, py::keep_alive<0, 1>())
        .def("__len__", [](rfkt::vardata& v) { return v.size_parameters(); })
        .def("__contains__", [](rfkt::vardata& v, std::string_view name) { return v.has_parameter(name); })
        .def_readwrite("weight", &rfkt::vardata::weight);

    py::class_<rfkt::vlink>(m, "vlink")
        .def("__getitem__", [](rfkt::vlink& v, std::string_view name) { return v[name]; }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](rfkt::vlink& v, std::string_view name, const rfkt::vardata& value) { v[name] = value; })
        .def("__iter__", [](rfkt::vlink& v) { return py::make_iterator(v.begin(), v.end()); }, py::keep_alive<0, 1>())
        .def("__len__", [](rfkt::vlink& v) { return v.size_variations(); })
        .def("__contains__", [](rfkt::vlink& v, std::string_view name) { return v.has_variation(name); })
        .def("serialize", [](const rfkt::vlink& v) { return v.serialize().dump(); })
        .def("lookup", &rfkt::vlink::lookup, py::return_value_policy::reference_internal)
        .def_readwrite("transform", &rfkt::vlink::transform)
        .def_readwrite("mod_x", &rfkt::vlink::mod_x)
        .def_readwrite("mod_y", &rfkt::vlink::mod_y)
        .def_readwrite("mod_scale", &rfkt::vlink::mod_scale)
        .def_readwrite("mod_rotate", &rfkt::vlink::mod_rotate)
        .def_static("identity", &rfkt::vlink::identity);

    py::class_<rfkt::xform>(m, "xform")
        .def_readwrite("weight", &rfkt::xform::weight)
        .def_readwrite("color", &rfkt::xform::color)
        .def_readwrite("color_speed", &rfkt::xform::color_speed)
        .def_readwrite("opacity", &rfkt::xform::opacity)
        .def("__len__", [](rfkt::xform& x) { return x.vchain.size(); })
        .def("__getitem__", [](rfkt::xform& x, int idx) -> std::optional<rfkt::vlink*> { 
            if(idx < 0 || idx >= x.vchain.size()) return std::nullopt;
            return &x.vchain[idx]; 
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](rfkt::xform& x, int idx, const rfkt::vlink& value) { x.vchain[idx] = value; })
        .def("__iter__", [](rfkt::xform& x) { return py::make_iterator(x.vchain.begin(), x.vchain.end()); }, py::keep_alive<0, 1>())
        .def("__len__", [](rfkt::xform& x) { return x.vchain.size(); })
        .def("serialize", [](const rfkt::xform& x) { return x.serialize().dump(); })
        .def("lookup", &rfkt::xform::lookup, py::return_value_policy::reference_internal)
        .def_static("identity", &rfkt::xform::identity);

    py::class_<rfkt::flame>(m, "flame")
        .def_readwrite("center_x", &rfkt::flame::center_x)
        .def_readwrite("center_y", &rfkt::flame::center_y)
        .def_readwrite("scale", &rfkt::flame::scale)
        .def_readwrite("rotate", &rfkt::flame::rotate)
        .def_readwrite("gamma", &rfkt::flame::gamma)
        .def_readwrite("brightness", &rfkt::flame::brightness)
        .def("__getitem__", [](rfkt::flame& f, int idx) -> std::optional<rfkt::xform*> { 
            if(idx < 0 || idx >= f.xforms().size()) return std::nullopt;
            return &f.xforms()[idx]; 
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](rfkt::flame& f, int idx, const rfkt::xform& value) { f.xforms()[idx] = value; })
        .def("__iter__", [](rfkt::flame& f) { return py::make_iterator(f.xforms().begin(), f.xforms().end()); }, py::keep_alive<0, 1>())
        .def("__len__", [](rfkt::flame& f) { return f.xforms().size(); })
        .def("serialize", [](const rfkt::flame& f) { return f.serialize().dump(); })
        .def("lookup", &rfkt::flame::lookup, py::return_value_policy::reference_internal)
        .def("hash", [](const rfkt::flame& f) { return f.hash().str64(); })
        .def("value_hash", [](const rfkt::flame& f) { return f.value_hash().str64(); });
}