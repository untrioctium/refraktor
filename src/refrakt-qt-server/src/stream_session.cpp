#include "stream_session.hpp"

#include <QCoreApplication>
#include <QJsonDocument>

#include <spdlog/spdlog.h>

#include <librefrakt/util/filesystem.hpp>
#include <librefrakt/util.hpp>

#include <services/kernel_compile_queue.hpp>
#include <services/variation_database.hpp>
#include <services/animation_database.hpp>

// ---------------------------------------------------------------------------
// StreamSession
// ---------------------------------------------------------------------------

StreamSession::StreamSession(
    std::unique_ptr<QWebSocket> socket,
    roccu::context ctx,
    QObject* parent)
    : QObject(parent)
    , m_socket(std::move(socket))
    , m_ctx(ctx)
{
    connect(m_socket.get(), &QWebSocket::textMessageReceived,
            this, &StreamSession::onTextMessage);
    connect(m_socket.get(), &QWebSocket::disconnected,
            this, &StreamSession::onDisconnected);
    connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
            this, &StreamSession::stopRendering);

    SPDLOG_INFO("Stream session created");
}

StreamSession::~StreamSession()
{
    stopRendering();
    SPDLOG_INFO("Stream session destroyed");
}

void StreamSession::stopRendering()
{
    m_stopFlag.store(true, std::memory_order_release);

    if (m_sendTimer) {
        m_sendTimer->stop();
    }

    if (m_renderThread.joinable()) {
        m_renderThread.join();
    }
}

void StreamSession::onTextMessage(const QString& message)
{
    auto doc = QJsonDocument::fromJson(message.toUtf8());
    if (!doc.isObject()) {
        SPDLOG_WARN("Could not parse JSON from WebSocket message");
        return;
    }

    auto obj = doc.object();
    auto cmd = obj.value("cmd").toString();

    if (cmd == u"begin") {
        onBegin(obj);
    }
}

void StreamSession::onBegin(const QJsonObject& data)
{
    if (m_worker) {
        SPDLOG_WARN("begin received but session already rendering");
        return;
    }

    auto width = static_cast<unsigned int>(data.value("width").toInt(1280));
    auto height = static_cast<unsigned int>(data.value("height").toInt(720));
    auto upscale = data.value("upscale").toBool(false);
    auto secondsPerLoop = data.value("loop_length").toDouble(5.0);
    auto embellish = data.value("embellish").toBool(false);

    m_fps = static_cast<unsigned int>(data.value("fps").toInt(30));
    auto loopsPerFrame = 1.0 / (secondsPerLoop * m_fps);
    auto maxBinTime = 1000.0 / m_fps - 6.0;

    auto& fdb = VariationDatabase::instance()->db();
    auto& ft = AnimationDatabase::instance()->table();

    auto flamePath = [&]() -> rfkt::fs::path {
        auto flameField = data.value("flame").toString();
        if (flameField.isEmpty()) {
            auto localFlames = rfkt::fs::list(
                "assets/flames_test", rfkt::fs::filter::has_extension(".flam3"));
            return localFlames[std::rand() % localFlames.size()];
        }
        return std::format(
            "assets/flames_test/electricsheep.{}.flam3",
            flameField.toStdString());
    }();

    auto flameResult = rfkt::import_flam3(fdb, rfkt::fs::read_string(flamePath));
    if (!flameResult) {
        SPDLOG_WARN("Could not import {}: {}", flamePath.string(), flameResult.error());
        return;
    }

    auto flame = std::move(flameResult.value());

    if (embellish) {
        flame.for_each_xform([&](auto xid, rfkt::xform& xf) {
            if (xid == -1) return;
            for (auto& vlink : xf.vchain) {
                if (!vlink.mod_rotate.call_info) {
                    vlink.mod_rotate.call_info = ft.make_default("increase");
                    vlink.mod_rotate.call_info->args["per_loop"] =
                        5.0 * ((rand() & 1) ? -1 : 1);
                }
            }
        });
    }

    SPDLOG_INFO("Compiling kernel for {} ({}x{})", flamePath.string(), width, height);

    auto future = KernelCompileQueue::instance()->requestCompile(
        fdb, flame, rfkt::precision::f32);

    rfkt::uint2 outputDims{width, height};
    rfkt::uint2 binDims = outputDims;
    if (upscale) {
        binDims.x /= 2;
        binDims.y /= 2;
    }

    future.then(this, [this,
                       flame = std::move(flame),
                       outputDims,
                       binDims,
                       loopsPerFrame,
                       maxBinTime,
                       upscale](rfkt::flame_compiler::result result) mutable {

        if (!result.kernel.has_value()) {
            SPDLOG_WARN("Kernel compilation failed:\n{}", result.log);
            return;
        }

        SPDLOG_INFO("Kernel compiled, starting render ({}x{} -> {}x{} @ {} fps)",
                     binDims.x, binDims.y, outputDims.x, outputDims.y, m_fps);

        StreamRenderWorker::Config config{
            .flame = std::move(flame),
            .kernel = std::move(result.kernel.value()),
            .ctx = m_ctx,
            .outputDims = outputDims,
            .binDims = binDims,
            .fps = m_fps,
            .loopsPerFrame = loopsPerFrame,
            .maxBinTime = maxBinTime,
            .upscale = upscale,
        };

        m_worker = std::make_unique<StreamRenderWorker>(
            std::move(config), m_chunks, m_stopFlag);

        m_renderThread = std::thread([this]() { m_worker->run(); });

        m_sendTimer = new QTimer(this);
        m_sendTimer->setTimerType(Qt::PreciseTimer);
        connect(m_sendTimer, &QTimer::timeout, this, &StreamSession::sendFrame);
        m_sendTimer->start(static_cast<int>(1000 / m_fps));
    });
}

void StreamSession::sendFrame()
{
    QByteArray data;
    if (m_chunks.try_dequeue(data)) {
        m_socket->sendBinaryMessage(data);
    } else {
        SPDLOG_INFO("Buffer miss");
    }
}

void StreamSession::onDisconnected()
{
    SPDLOG_INFO("WebSocket disconnected, tearing down stream session");
    stopRendering();
    deleteLater();
}

// ---------------------------------------------------------------------------
// StreamRenderWorker
// ---------------------------------------------------------------------------

StreamRenderWorker::StreamRenderWorker(
    Config&& config,
    moodycamel::BlockingReaderWriterCircularBuffer<QByteArray>& chunks,
    std::atomic_bool& stopFlag)
    : m_config(std::move(config))
    , m_chunks(chunks)
    , m_stopFlag(stopFlag)
{
}

StreamRenderWorker::~StreamRenderWorker()
{
    SPDLOG_INFO("Render worker destroyed");
}

void StreamRenderWorker::run()
{
    m_config.ctx.make_current_if_not();

    auto* km = KernelCompileQueue::kernelManagerInstance();
    m_tonemapper.emplace(*km);
    m_converter.emplace(*km);

    auto dnFlags = m_config.upscale
        ? rfkt::denoiser_flag::upscale
        : rfkt::denoiser_flag::none;

    m_denoiser = rfkt::denoiser::make(
        "rfkt::optix_denoise", m_config.binDims, dnFlags, m_stream);

    m_tonemapped = roccu::gpu_image<rfkt::half3>(
        m_config.binDims.x, m_config.binDims.y, m_stream);
    m_denoised = roccu::gpu_image<rfkt::half3>(
        m_config.outputDims.x, m_config.outputDims.y, m_stream);

    m_encoder = std::make_unique<eznve::encoder>(
        eznve::config::for_streaming(
            eznve::uint2{m_config.outputDims.x, m_config.outputDims.y},
            eznve::uint2{m_config.fps, 1},
            eznve::codec::h264),
        m_config.ctx,
        [](std::string_view msg) {});

    m_start = std::chrono::high_resolution_clock::now();

    SPDLOG_INFO("Render worker started");
    while (!m_stopFlag.load(std::memory_order_acquire)) {
        renderLoop();
    }
    SPDLOG_INFO("Render worker stopping");
}

void StreamRenderWorker::renderLoop()
{
    const auto t = m_totalFrames * m_config.loopsPerFrame;
    auto& ft = AnimationDatabase::instance()->table();
    auto invoker = ft.make_invoker();

    std::vector<double> samples;
    auto packer = [&samples](double v) { samples.push_back(v); };
    m_config.flame.pack_samples(
        packer, invoker,
        t - 1.0 * m_config.loopsPerFrame,
        1.0 * m_config.loopsPerFrame,
        4,
        static_cast<int>(m_config.binDims.x),
        static_cast<int>(m_config.binDims.y));

    auto state = m_config.kernel.warmup(
        m_stream, samples, m_config.binDims, 0xdeadbeef, 64);

    auto effectiveMaxBinTime = m_config.maxBinTime - m_ppTimeEstimate;

    auto frameQuality = 0.0;
    auto subpasses = 0;

    while (frameQuality < (.95 * m_targetQuality.value_or(100)) && subpasses < 2) {
        if (subpasses > 0) {
            SPDLOG_INFO("Repairing frame ({}/10 buffered, wanted {:.4}, got {:.4})",
                        m_chunks.size_approx(), m_targetQuality.value(), frameQuality);
        }

        auto result = m_config.kernel.bin(m_stream, state, {
            .iters = 1'000'000,
            .millis = static_cast<std::uint32_t>(std::max(1.0, effectiveMaxBinTime)),
            .quality = m_targetQuality.value_or(1000) - frameQuality
        }).get();

        if (m_totalFrames % m_config.fps == 0) {
            SPDLOG_INFO("{:.4} m draws/ms",
                static_cast<double>(result.total_draws) / 1'000'000.0 / result.elapsed_ms);
        }

        frameQuality += result.quality;
        if (!m_targetQuality) m_targetQuality = frameQuality;
        subpasses++;
    }

    if (m_stopFlag.load(std::memory_order_acquire)) return;

    if (m_targetQuality.has_value() &&
        (frameQuality < m_targetQuality.value() * .95 || subpasses > 1)) {
        auto buffered = m_chunks.size_approx();
        if (buffered <= 4)      m_targetQuality.value() *= .975;
        else if (buffered <= 8) m_targetQuality.value() *= .99;
        else                    m_targetQuality.value() *= .995;
    }

    rfkt::timer ppTimer;

    auto gamma = m_config.flame.gamma.sample(t, invoker);
    auto brightness = m_config.flame.brightness.sample(t, invoker);
    auto vibrancy = m_config.flame.vibrancy.sample(t, invoker);

    m_tonemapper->run(
        state.cold_bins, state.hot_bins, m_tonemapped,
        {frameQuality, gamma, brightness, vibrancy},
        m_stream);

    if (m_denoiser) {
        m_denoiser->denoise(m_tonemapped, m_denoised, m_dnEvent).get();
    } else {
        m_denoised = std::move(m_tonemapped);
        m_tonemapped = roccu::gpu_image<rfkt::half3>(
            m_config.binDims.x, m_config.binDims.y, m_stream);
    }

    roccu::gpu_image_view<rfkt::uchar4> encoderView{
        m_encoder->buffer(),
        m_encoder->width(),
        m_encoder->height()
    };

    m_converter->to_uchar4(m_denoised, encoderView, m_stream);
    m_stream.sync();

    auto chunks = m_encoder->submit_frame(m_totalFrames % m_config.fps == 0 ? eznve::frame_flag::idr : eznve::frame_flag::none);

    auto ppTimeMs = ppTimer.count() * 1000.0;
    m_ppTimeEstimate = m_ppTimeEstimate * 0.9 + ppTimeMs * 0.1;

    if (!chunks.empty()) {
        QByteArray aggregated;
        for (auto& c : chunks) {
            aggregated.append(c.data.data(), static_cast<qsizetype>(c.data.size()));
        }

        if (!m_chunks.try_enqueue(std::move(aggregated))) {
            if (m_targetQuality.has_value() && subpasses == 1
                && m_targetQuality.value() < 200) {
                m_targetQuality.value() *= 1.01;
            }
            while (!m_chunks.wait_enqueue_timed(aggregated, 100)) {
                if (m_stopFlag.load(std::memory_order_acquire)) return;
            }
        }
    }

    m_totalFrames++;

    if (m_totalFrames % m_config.fps == 0 && m_totalFrames > 0) {
        SPDLOG_INFO("buffer: {}/10, target: {:.4}, {:.4} mbps, pp: {:.2}ms",
                    m_chunks.size_approx(),
                    m_targetQuality.value_or(0),
                    m_encoder->total_bytes() / secsSinceStart() / 1'000'000.0 * 8.0,
                    m_ppTimeEstimate);
    }
}

double StreamRenderWorker::secsSinceStart() const
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - m_start).count() / 1'000'000.0;
}
