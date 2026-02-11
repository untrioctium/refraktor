#include "local_render_queue.hpp"
#include "kernel_compile_queue.hpp"
#include "variation_database.hpp"
#include "animation_database.hpp"

#include <QDebug>
#include <QtConcurrent/QtConcurrent>

#include <roccu.hpp>

#include <chrono>

constexpr static auto tile_dimensions = rfkt::uint2{512, 512};

LocalRenderQueue::LocalRenderQueue(QObject* parent)
    : QObject(parent)
    , m_renderPool(this)
    , m_tonemapper(*KernelCompileQueue::kernelManagerInstance())
    , m_denoiser(rfkt::denoiser::make(
          "rfkt::optix_denoise",
          tile_dimensions,
          rfkt::denoiser_flag::tiled,
          m_stream))
    , m_upscaleDenoiser(rfkt::denoiser::make(
          "rfkt::optix_denoise",
          tile_dimensions,
          rfkt::denoiser_flag::tiled | rfkt::denoiser_flag::upscale,
          m_stream))
    , m_converter(*KernelCompileQueue::kernelManagerInstance())
{
    m_renderPool.setMaxThreadCount(1);
    m_renderPool.setExpiryTimeout(-1);

    auto ctx = roccu::context::current();
    m_renderPool.start([ctx]() {
        ctx.make_current();
    });
}

LocalRenderQueue::~LocalRenderQueue() {
    m_renderPool.waitForDone();
}

LocalRenderQueue* LocalRenderQueue::instance() {
    static LocalRenderQueue* s_instance = nullptr;
    if (!s_instance) {
        s_instance = new LocalRenderQueue();
    }
    return s_instance;
}

LocalRenderQueue* LocalRenderQueue::create(QQmlEngine*, QJSEngine*) {
    return instance();
}

QFuture<QImage> LocalRenderQueue::requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) {
    qDebug() << "Requesting render to QImage. Params:";
    qDebug() << "FPS: " << params.fps;
    qDebug() << "Seconds per loop: " << params.secondsPerLoop;
    qDebug() << "Target quality: " << params.targetQuality;
    qDebug() << "Max render millis: " << params.maxRenderMillis;
    qDebug() << "Denoise: " << params.denoise;
    qDebug() << "Dimensions: " << params.dims.x << "x" << params.dims.y;
    qDebug() << "Time: " << params.t;

    auto kernel_future = KernelCompileQueue::instance()->requestCompile(
        VariationDatabase::instance()->db(), f, rfkt::precision::f32);
    auto invoker = AnimationDatabase::instance()->table().make_invoker();

    std::vector<double> samples = {};
    auto packer = [&samples](double v) { samples.push_back(v); };

    auto loops_per_frame = 1.0 / (params.fps * params.secondsPerLoop);

    auto dims = params.dims;
    if (params.upscale) {
        dims.x /= 2;
        dims.y /= 2;
    }

    f.pack_sample(packer, invoker, params.t - 1.2 * loops_per_frame, dims.x, dims.y);
    f.pack_sample(packer, invoker, params.t, dims.x, dims.y);
    f.pack_sample(packer, invoker, params.t + 1.2 * loops_per_frame, dims.x, dims.y);
    f.pack_sample(packer, invoker, params.t + 2.4 * loops_per_frame, dims.x, dims.y);

    struct gbv_t {
        double gamma = 1.0;
        double brightness = 0.0;
        double vibrancy = 0.0;
    };

    gbv_t gbv;
    gbv.gamma = f.gamma.sample(params.t, invoker);
    gbv.brightness = f.brightness.sample(params.t, invoker);
    gbv.vibrancy = f.vibrancy.sample(params.t, invoker);

    return kernel_future.then(&m_renderPool,[
         samples = std::move(samples),
         gbv = std::move(gbv),
         params = params,
         &stream = this->m_stream,
         &tm = this->m_tonemapper,
         dn = params.upscale ? this->m_upscaleDenoiser.get() : this->m_denoiser.get(),
         &dn_event = this->m_dnEvent,
         &conv = this->m_converter](rfkt::flame_compiler::result kernel_result) mutable {

            auto start = std::chrono::high_resolution_clock::now();

            if (!kernel_result.kernel.has_value()) {
                qInfo() << "Failed to compile kernel: " << kernel_result.log;
                return QImage();
            }

            auto bin_dims = params.dims;
            if (params.upscale) {
                bin_dims.x /= 2;
                bin_dims.y /= 2;
            }

            auto tonemapped = roccu::gpu_image<rfkt::half3>(bin_dims.x, bin_dims.y, stream);
            auto denoised = roccu::gpu_image<rfkt::half3>(params.dims.x, params.dims.y, stream);
            auto converted = roccu::gpu_image<rfkt::uchar4>(params.dims.x, params.dims.y, stream);

            auto& kernel = kernel_result.kernel.value();

            auto state = kernel.warmup(stream, samples, bin_dims, 0xdeadbeef, 100);
            auto bin_result = kernel.bin(stream, state, {.millis = params.maxRenderMillis, .quality = params.targetQuality}).get();

            tm.run(state.bins, tonemapped, {bin_result.quality, gbv.gamma, gbv.brightness, gbv.vibrancy}, stream);

            if (params.denoise) {
                auto time = dn->denoise(tonemapped, denoised, dn_event).get();
                qDebug() << "Denoising time: " << time * 1000.0 << "ms";
            } else {
                denoised = std::move(tonemapped);
            }

            conv.to_uchar4(denoised, converted, stream);

            stream.sync();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            qDebug() << "Render time: " << duration << "ms";
            qDebug() << "Bin time: " << bin_result.elapsed_ms << "ms";

            auto host_converted = converted.to_host_flat();

            return QImage(
                reinterpret_cast<uchar*>(host_converted.data()),
                static_cast<int>(params.dims.x),
                static_cast<int>(params.dims.y),
                QImage::Format_RGBA8888).copy();
        });
}
