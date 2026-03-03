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
    roccu::context ppCtx,
    QObject* parent)
    : QObject(parent)
    , m_socket(std::move(socket))
    , m_ctx(ctx)
    , m_ppCtx(ppCtx)
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

    if (m_worker) {
        m_worker->requestStop();
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
    } else if (cmd == u"next_flame") {
        if (m_worker) m_worker->requestSkip();
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
    auto displayLoops = data.value("display_loops").toDouble(4.0);
    auto transitionLoops = data.value("transition_loops").toDouble(1.0);
    auto bitrateKbps = static_cast<unsigned int>(data.value("bitrate_kbps").toInt(25000));

    m_fps = static_cast<unsigned int>(data.value("fps").toInt(30));
    auto loopsPerFrame = 1.0 / (secondsPerLoop * m_fps);
    auto maxBinTime = 1000.0 / m_fps - 6.0;

    auto& fdb = VariationDatabase::instance()->db();
    auto& ft = AnimationDatabase::instance()->table();

    auto embellishFlame = [&](rfkt::flame& f) {
        f.for_each_xform([&](auto xid, rfkt::xform& xf) {
            if (xid == -1) return;
            for (auto& vlink : xf.vchain) {
                if (!vlink.mod_rotate.call_info) {
                    vlink.mod_rotate.call_info = ft.make_default("increase");
                    vlink.mod_rotate.call_info->args["per_loop"] =
                        5.0 * ((rand() & 1) ? -1 : 1);
                }
            }
        });
    };

    auto loadFlame = [&](const rfkt::fs::path& path) -> std::optional<rfkt::flame> {
        auto result = rfkt::import_flam3(fdb, rfkt::fs::read_string(path));
        if (!result) {
            SPDLOG_WARN("Could not import {}: {}", path.string(), result.error());
            return std::nullopt;
        }
        auto f = std::move(result.value());
        if (embellish) embellishFlame(f);
        return f;
    };

    auto localFlames = rfkt::fs::list(
        "assets/flames_stream", rfkt::fs::filter::has_extension(".flam3"));

    auto flamePath = [&]() -> rfkt::fs::path {
        auto flameField = data.value("flame").toString();
        if (flameField.isEmpty()) {
            return localFlames[std::rand() % localFlames.size()];
        }
        return fmt::format(
            "assets/flames_stream/electricsheep.{}.flam3",
            flameField.toStdString());
    }();

    auto flame = loadFlame(flamePath);
    if (!flame) return;

    auto nextFlamePath = localFlames[std::rand() % localFlames.size()];
    auto nextFlame = loadFlame(nextFlamePath);
    if (!nextFlame) return;

    auto interp = rfkt::interpolator(*flame, *nextFlame, fdb, false);

    SPDLOG_INFO("Compiling transition kernel for {} -> {} ({}x{})",
                flamePath.string(), nextFlamePath.string(), width, height);

    auto future = KernelCompileQueue::instance()->requestCompile(
        fdb, interp.left_flame(), rfkt::precision::f32);

    rfkt::uint2 outputDims{width, height};
    rfkt::uint2 binDims = outputDims;
    if (upscale) {
        binDims.x /= 2;
        binDims.y /= 2;
    }

    future.then(this, [this,
                       flame = std::move(*flame),
                       nextFlame = std::move(*nextFlame),
                       interp = std::move(interp),
                       outputDims,
                       binDims,
                       loopsPerFrame,
                       maxBinTime,
                       upscale,
                       embellish,
                       displayLoops,
                       transitionLoops, bitrateKbps](rfkt::flame_compiler::result result) mutable {

        if (!result.kernel.has_value()) {
            SPDLOG_WARN("Kernel compilation failed:\n{}", result.log);
            return;
        }

        SPDLOG_INFO("Kernel compiled, starting render ({}x{} -> {}x{} @ {} fps)",
                     binDims.x, binDims.y, outputDims.x, outputDims.y, m_fps);

        StreamRenderWorker::Config config{
            .currentFlame = std::move(flame),
            .pendingFlame = std::move(nextFlame),
            .interpolator = std::move(interp),
            .kernel = std::move(result.kernel.value()),
            .ctx = m_ctx,
            .ppCtx = m_ppCtx,
            .outputDims = outputDims,
            .binDims = binDims,
            .fps = m_fps,
            .loopsPerFrame = loopsPerFrame,
            .maxBinTime = maxBinTime,
            .upscale = upscale,
            .embellish = embellish,
            .displayLoops = displayLoops,
            .transitionLoops = transitionLoops,
            .bitrateKbps = bitrateKbps,
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

void StreamRenderWorker::requestSkip()
{
    m_skipRequested.store(true, std::memory_order_release);
}

void StreamRenderWorker::requestStop()
{
    m_handoff_cv.notify_all();
}

StreamRenderWorker::~StreamRenderWorker()
{
    SPDLOG_INFO("Render worker destroyed");
}

rfkt::flame StreamRenderWorker::loadRandomFlame()
{
    auto& fdb = VariationDatabase::instance()->db();
    auto& ft = AnimationDatabase::instance()->table();

    auto localFlames = rfkt::fs::list(
        "assets/flames_stream", rfkt::fs::filter::has_extension(".flam3"));
    auto flamePath = localFlames[std::rand() % localFlames.size()];

    auto flameResult = rfkt::import_flam3(fdb, rfkt::fs::read_string(flamePath));
    if (!flameResult) {
        SPDLOG_WARN("Could not import {}: {}", flamePath.string(), flameResult.error());
        return m_currentFlame;
    }

    auto flame = std::move(flameResult.value());

    if (m_config.embellish) {
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

    SPDLOG_INFO("Loaded next flame: {}", flamePath.string());
    return flame;
}

void StreamRenderWorker::beginNextFlamePreparation()
{
    auto& fdb = VariationDatabase::instance()->db();

    auto nextFlame = loadRandomFlame();
    auto interp = rfkt::interpolator(m_currentFlame, nextFlame, fdb, false);

    auto kernelFuture = KernelCompileQueue::instance()->requestCompile(
        fdb, interp.left_flame(), rfkt::precision::f32);

    m_nextPrep = NextFlamePrep{
        .originalFlame = std::move(nextFlame),
        .interpolator = std::move(interp),
        .kernelFuture = std::move(kernelFuture),
    };

    SPDLOG_INFO("Began preparation for next flame transition");
}

void StreamRenderWorker::advancePhase(double t)
{
    if (m_nextPrep && m_nextPrep->kernelFuture.isFinished()) {
        auto result = m_nextPrep->kernelFuture.takeResult();
        if (result.kernel.has_value()) {
            m_pendingFlame = std::move(m_nextPrep->originalFlame);
            m_activeKernel = std::move(result.kernel.value());
            m_interpolator = std::move(m_nextPrep->interpolator);
            m_mix = 0.0;
            SPDLOG_INFO("Transition kernel ready, swapped (mix=0, seamless)");
        } else {
            SPDLOG_WARN("Transition kernel compilation failed:\n{}", result.log);
        }
        m_nextPrep.reset();
    }

    if (m_phase == Phase::Display) {
        bool skip = m_skipRequested.load(std::memory_order_acquire);
        bool timeElapsed = (t - m_phaseStartT) >= m_config.displayLoops;
        if ((timeElapsed || skip) && m_pendingFlame) {
            if (skip) m_skipRequested.store(false, std::memory_order_release);
            m_phase = Phase::Transition;
            m_phaseStartT = t;
            SPDLOG_INFO("Entering transition phase at t={:.4}{}", t, skip ? " (skip requested)" : "");
        }
    } else {
        auto linear = std::clamp((t - m_phaseStartT) / m_config.transitionLoops, 0.0, 1.0);
        m_mix = linear * linear * linear * (linear * (linear * 6.0 - 15.0) + 10.0);

        if (m_mix >= 1.0) {
            if (m_pendingFlame) {
                m_currentFlame = std::move(*m_pendingFlame);
                m_pendingFlame.reset();
            }
            m_phase = Phase::Display;
            m_phaseStartT = t;
            SPDLOG_INFO("Transition complete, entering display phase at t={:.4}", t);
            beginNextFlamePreparation();
        }
    }
}

void StreamRenderWorker::run()
{
    m_sameDevice = (m_config.ctx == m_config.ppCtx);

    m_currentFlame = std::move(m_config.currentFlame);
    m_pendingFlame = std::move(m_config.pendingFlame);
    m_interpolator = std::move(m_config.interpolator);
    m_activeKernel = std::move(m_config.kernel);

    // GPU A resources
    m_config.ctx.make_current();
    auto* km = KernelCompileQueue::kernelManagerInstance();
    m_tonemapper.emplace(*km);
    m_streamA = roccu::gpu_stream{};
    m_tonemapped_a = roccu::gpu_image<rfkt::half3>(
        m_config.binDims.x, m_config.binDims.y, m_streamA);
    m_tonemapDone = roccu::gpu_event{};

    // GPU B resources
    m_config.ppCtx.make_current();
    m_converter.emplace(*km);
    m_streamB = roccu::gpu_stream{};
    m_dnEvent = roccu::gpu_event{};

    auto dnFlags = m_config.upscale
        ? rfkt::denoiser_flag::upscale
        : rfkt::denoiser_flag::none;

    m_denoiser = rfkt::denoiser::make(
        "rfkt::optix_denoise", m_config.outputDims, dnFlags, m_streamB);

    m_denoised_b = roccu::gpu_image<rfkt::half3>(
        m_config.outputDims.x, m_config.outputDims.y, m_streamB);

    if (!m_sameDevice) {
        m_tonemapped_b = roccu::gpu_image<rfkt::half3>(
            m_config.binDims.x, m_config.binDims.y, m_streamB);
    }

    auto encConfig = eznve::config::for_streaming(
        eznve::uint2{m_config.outputDims.x, m_config.outputDims.y},
        eznve::uint2{m_config.fps, 1},
        eznve::codec::h264);
    encConfig.bitrate_kbps = m_config.bitrateKbps;

    m_encoder = std::make_unique<eznve::encoder>(
        encConfig, m_config.ppCtx, [](std::string_view) {});

    m_start = std::chrono::high_resolution_clock::now();

    SPDLOG_INFO("Render worker started (sameDevice={})", m_sameDevice);

    auto ppThread = std::thread([this]() { postProcessLoop(); });
    binningLoop();

    m_handoff_cv.notify_all();
    ppThread.join();

    SPDLOG_INFO("Render worker stopping");
}

void StreamRenderWorker::binningLoop()
{
    m_config.ctx.make_current();

    while (!m_stopFlag.load(std::memory_order_acquire)) {
        const auto t = m_totalFrames * m_config.loopsPerFrame;

        advancePhase(t);

        auto& ft = AnimationDatabase::instance()->table();
        auto invoker = ft.make_invoker();

        // 1. Warmup + bin current frame
        std::vector<double> samples;
        auto packer = [&samples](double v) { samples.push_back(v); };

        m_interpolator->pack_samples(
            packer, invoker,
            t - 1.0 * m_config.loopsPerFrame,
            1.0 * m_config.loopsPerFrame,
            4,
            static_cast<int>(m_config.binDims.x),
            static_cast<int>(m_config.binDims.y),
            m_mix);

        auto mp_count = roccu::context::current().device().mp_count();
        auto half_sm_blocks = m_activeKernel.blocks_per_sm() * (mp_count / 2);

        auto state = m_activeKernel.warmup(
            m_streamA, samples, m_config.binDims, 0xdeadbeef, 64, 1, static_cast<int>(half_sm_blocks));

        auto frameQuality = 0.0;
        auto subpasses = 0;

        while (frameQuality < (.95 * m_targetQuality.value_or(100)) && subpasses < 2) {
            if (subpasses > 0) {
                SPDLOG_INFO("Repairing frame ({}/10 buffered, wanted {:.4}, got {:.4})",
                            m_chunks.size_approx(), m_targetQuality.value(), frameQuality);

                if (m_chunks.size_approx() <= 4) {
                    SPDLOG_INFO("Skipping repair; buffer is too small");
                    break;
                }
            }

            auto result = m_activeKernel.bin(m_streamA, state, {
                .iters = 1'000'000,
                .millis = static_cast<std::uint32_t>(m_config.maxBinTime),
                .quality = subpasses == 0 ? m_targetQuality.value_or(100) * 1.2 : m_targetQuality.value_or(1000) - frameQuality
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

        // 2. Tonemap current frame
        auto gamma = m_interpolator->interp_anima(&rfkt::flame::gamma, invoker, t, m_mix);
        auto brightness = m_interpolator->interp_anima(&rfkt::flame::brightness, invoker, t, m_mix);
        auto vibrancy = m_interpolator->interp_anima(&rfkt::flame::vibrancy, invoker, t, m_mix);

        m_tonemapper->run(
            state.cold_bins, state.hot_bins, m_tonemapped_a,
            {frameQuality, gamma, brightness, vibrancy},
            m_streamA);

        m_streamA.record(m_tonemapDone);

        // 3. Hand off to post-processing thread
        {
            std::unique_lock lock(m_handoff_mutex);
            m_handoff_cv.wait(lock, [this] {
                return !m_pendingPP.has_value() || m_stopFlag.load(std::memory_order_acquire);
            });
            if (m_stopFlag.load(std::memory_order_acquire)) return;

            m_pendingPP = PostProcessFrame{
                .quality = frameQuality,
                .gamma = gamma,
                .brightness = brightness,
                .vibrancy = vibrancy,
                .frameNumber = m_totalFrames,
            };
        }
        m_handoff_cv.notify_one();

        m_totalFrames++;

        if (m_totalFrames % m_config.fps == 0 && m_totalFrames > 0) {
            SPDLOG_INFO("buffer: {}/10, target: {:.4}, {:.4} mbps, phase: {}, mix: {:.3}",
                        m_chunks.size_approx(),
                        m_targetQuality.value_or(0),
                        m_encoder->total_bytes() / secsSinceStart() / 1'000'000.0 * 8.0,
                        m_phase == Phase::Display ? "display" : "transition",
                        m_mix);
        }
    }
}

void StreamRenderWorker::postProcessLoop()
{
    m_config.ppCtx.make_current();

    while (!m_stopFlag.load(std::memory_order_acquire)) {
        PostProcessFrame frame{};
        {
            std::unique_lock lock(m_handoff_mutex);
            m_handoff_cv.wait(lock, [this] {
                return m_pendingPP.has_value() || m_stopFlag.load(std::memory_order_acquire);
            });
            if (m_stopFlag.load(std::memory_order_acquire)) break;

            frame = std::move(*m_pendingPP);
            m_pendingPP.reset();
        }
        m_handoff_cv.notify_one();

        m_streamB.wait_for(m_tonemapDone);

        auto& denoiseInput = m_sameDevice ? m_tonemapped_a : m_tonemapped_b;

        if (!m_sameDevice) {
            ROCCU_SAFE_CALL(cuMemcpyPeerAsync(
                m_tonemapped_b.ptr(), m_config.ppCtx,
                m_tonemapped_a.ptr(), m_config.ctx,
                m_tonemapped_a.size_bytes(), m_streamB));
        }

        roccu::gpu_image_view<rfkt::uchar4> encoderView{
            m_encoder->buffer(),
            m_encoder->width(),
            m_encoder->height()
        };

        if (m_denoiser) {
            m_denoiser->denoise(denoiseInput, m_denoised_b, m_dnEvent);
            m_converter->to_uchar4(m_denoised_b, encoderView, m_streamB);
        } else {
            m_converter->to_uchar4(denoiseInput, encoderView, m_streamB);
        }

        m_streamB.sync();

        auto idrFlag = frame.frameNumber % m_config.fps == 0
            ? eznve::frame_flag::idr
            : eznve::frame_flag::none;

        auto chunks = m_encoder->submit_frame(idrFlag);

        if (!chunks.empty()) {
            QByteArray aggregated;
            for (auto& c : chunks) {
                aggregated.append(c.data.data(), static_cast<qsizetype>(c.data.size()));
            }

            if (!m_chunks.try_enqueue(std::move(aggregated))) {
                if (m_targetQuality.has_value()
                    && m_targetQuality.value() < 200) {
                    m_targetQuality.value() *= 1.05;
                }
                while (!m_chunks.wait_enqueue_timed(aggregated, 100)) {
                    if (m_stopFlag.load(std::memory_order_acquire)) return;
                }
            }
        }
    }
}

double StreamRenderWorker::secsSinceStart() const
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - m_start).count() / 1'000'000.0;
}
