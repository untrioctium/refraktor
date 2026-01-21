#include "flame_render_queue.hpp"
#include "kernel_compile_queue.hpp"
#include "variation_database.hpp"
#include "animation_database.hpp"

#include <QDebug>
#include <QtConcurrent/QtConcurrent>

#include <roccu.hpp>

#include <chrono>

FlameRenderQueue::FlameRenderQueue(QObject* parent)
    : QObject(parent)
    , m_renderPool(this)
    , m_tonemapper(*KernelCompileQueue::kernelManagerInstance())
    , m_denoiser(rfkt::denoiser::make(
          "rfkt::optix_denoise",
          uint2{512, 512},
          rfkt::denoiser_flag::tiled,
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

FlameRenderQueue::~FlameRenderQueue() {
    m_renderPool.waitForDone();
}

FlameRenderQueue* FlameRenderQueue::instance() {
    static FlameRenderQueue* s_instance = nullptr;
    if (!s_instance) {
        s_instance = new FlameRenderQueue();
    }
    return s_instance;
}

FlameRenderQueue* FlameRenderQueue::create(QQmlEngine*, QJSEngine*) {
    return instance();
}

QFuture<QImage> FlameRenderQueue::requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) {
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

    f.pack_sample(packer, invoker, params.t - 1.2 * loops_per_frame, params.dims.x, params.dims.y);
    f.pack_sample(packer, invoker, params.t, params.dims.x, params.dims.y);
    f.pack_sample(packer, invoker, params.t + 1.2 * loops_per_frame, params.dims.x, params.dims.y);
    f.pack_sample(packer, invoker, params.t + 2.4 * loops_per_frame, params.dims.x, params.dims.y);

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
         dn = this->m_denoiser.get(),
         &dn_event = this->m_dnEvent,
         &conv = this->m_converter](rfkt::flame_compiler::result&& kernel_result) mutable {

            auto start = std::chrono::high_resolution_clock::now();

            if (!kernel_result.kernel.has_value()) {
                qDebug() << "Failed to compile kernel: " << kernel_result.log;
                return QImage();
            }

            auto tonemapped = roccu::gpu_image<half3>(params.dims, stream);
            auto denoised = roccu::gpu_image<half3>(params.dims, stream);
            auto converted = roccu::gpu_image<uchar4>(params.dims, stream);

            auto& kernel = kernel_result.kernel.value();

            auto state = kernel.warmup(stream, samples, params.dims, 0xdeadbeef, 100);
            auto bin_result = kernel.bin(stream, state, {.millis = params.maxRenderMillis, .quality = params.targetQuality}).get();

            tm.run(state.bins, tonemapped, {bin_result.quality, gbv.gamma, gbv.brightness, gbv.vibrancy}, stream);

            if (params.denoise) {
                dn->denoise(tonemapped, denoised, dn_event);
                stream.wait_for(dn_event);
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
                params.dims.x,
                params.dims.y,
                QImage::Format_RGBA8888).copy();
        });
}
