#pragma once

#include <QObject>
#include <QTimer>
#include <QWebSocket>
#include <QByteArray>
#include <QJsonObject>
#include <QFuture>

#include <atomic>
#include <optional>
#include <memory>
#include <chrono>
#include <thread>

#include <roccu.hpp>
#include <roccu_cpp_types.hpp>

#include <librefrakt/flame_compiler.hpp>
#include <librefrakt/flame_types.hpp>
#include <librefrakt/image/tonemapper.hpp>
#include <librefrakt/image/converter.hpp>
#include <librefrakt/interface/denoiser.hpp>
#include <librefrakt/anima.hpp>
#include <eznve.hpp>

#include <readerwritercircularbuffer.h>

class StreamRenderWorker {
public:
    struct Config {
        rfkt::flame currentFlame;
        rfkt::flame pendingFlame;
        std::optional<rfkt::interpolator> interpolator;
        rfkt::flame_kernel kernel;
        roccu::context ctx;
        rfkt::uint2 outputDims;
        rfkt::uint2 binDims;
        unsigned int fps = 30;
        double loopsPerFrame = 0.0;
        double maxBinTime = 30.0;
        bool upscale = false;
        bool denoise = true;
        bool embellish = false;
        double displayLoops = 4.0;
        double transitionLoops = 1.0;
        unsigned int bitrateKbps = 25000;
    };

    StreamRenderWorker(
        Config&& config,
        moodycamel::BlockingReaderWriterCircularBuffer<QByteArray>& chunks,
        std::atomic_bool& stopFlag);

    ~StreamRenderWorker();

    StreamRenderWorker(const StreamRenderWorker&) = delete;
    StreamRenderWorker& operator=(const StreamRenderWorker&) = delete;
    StreamRenderWorker(StreamRenderWorker&&) = delete;
    StreamRenderWorker& operator=(StreamRenderWorker&&) = delete;

    void run();
    void requestSkip();

private:
    void renderLoop();
    void advancePhase(double t);
    rfkt::flame loadRandomFlame();
    void beginNextFlamePreparation();
    double secsSinceStart() const;

    Config m_config;

    enum class Phase { Display, Transition };
    Phase m_phase = Phase::Display;
    double m_phaseStartT = 0.0;
    double m_mix = 0.0;

    rfkt::flame m_currentFlame;
    rfkt::flame_kernel m_activeKernel;
    std::optional<rfkt::interpolator> m_interpolator;
    std::optional<rfkt::flame> m_pendingFlame;

    struct NextFlamePrep {
        rfkt::flame originalFlame;
        std::optional<rfkt::interpolator> interpolator;
        QFuture<rfkt::flame_compiler::result> kernelFuture;
    };
    std::optional<NextFlamePrep> m_nextPrep;

    roccu::gpu_stream m_stream{};
    roccu::gpu_stream m_ppStream{};
    std::optional<rfkt::tonemapper> m_tonemapper;
    std::unique_ptr<rfkt::denoiser> m_denoiser;
    std::optional<rfkt::converter> m_converter;
    std::unique_ptr<eznve::encoder> m_encoder;

    roccu::gpu_image<rfkt::half3> m_tonemapped;
    roccu::gpu_image<rfkt::half3> m_denoised;
    roccu::gpu_event m_dnEvent{};

    struct PrevFrame {
        roccu::gpu_image<rfkt::float4> cold_bins;
        roccu::gpu_image<rfkt::half4> hot_bins;
        double quality = 0.0;
        double gamma = 0.0;
        double brightness = 0.0;
        double vibrancy = 0.0;
        std::int64_t frameNumber = 0;
    };
    std::optional<PrevFrame> m_prevFrame;

    moodycamel::BlockingReaderWriterCircularBuffer<QByteArray>& m_chunks;
    std::atomic_bool& m_stopFlag;

    std::int64_t m_totalFrames = 0;
    std::optional<double> m_targetQuality;
    double m_syncWaitEstimate = 0.0;
    std::atomic_bool m_skipRequested{false};

    std::chrono::high_resolution_clock::time_point m_start;
};

class StreamSession : public QObject {
    Q_OBJECT
public:
    StreamSession(
        std::unique_ptr<QWebSocket> socket,
        roccu::context ctx,
        QObject* parent = nullptr);

    ~StreamSession() override;

    StreamSession(const StreamSession&) = delete;
    StreamSession& operator=(const StreamSession&) = delete;
    StreamSession(StreamSession&&) = delete;
    StreamSession& operator=(StreamSession&&) = delete;

private slots:
    void onTextMessage(const QString& message);
    void sendFrame();
    void onDisconnected();

private:
    void onBegin(const QJsonObject& data);
    void stopRendering();

    std::unique_ptr<QWebSocket> m_socket;
    roccu::context m_ctx;

    std::thread m_renderThread;
    std::unique_ptr<StreamRenderWorker> m_worker;
    QTimer* m_sendTimer = nullptr;

    moodycamel::BlockingReaderWriterCircularBuffer<QByteArray> m_chunks{10};
    std::atomic_bool m_stopFlag{false};

    unsigned int m_fps = 30;
};
