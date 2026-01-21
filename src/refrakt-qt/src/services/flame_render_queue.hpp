#pragma once

#include <QObject>
#include <QThreadPool>
#include <QFuture>
#include <QImage>
#include <QQmlEngine>

#include <librefrakt/flame_types.hpp>
#include <librefrakt/image/tonemapper.hpp>
#include <librefrakt/image/converter.hpp>
#include <librefrakt/interface/denoiser.hpp>

#include <roccu_cpp_types.hpp>

#include <qqmlintegration.h>

#include <memory>

struct RenderParams {
    uint2 dims = {640, 480};
    int fps = 30;
    double t = 0.0;
    double secondsPerLoop = 5.0;
    double targetQuality = 16;
    std::uint32_t maxRenderMillis = 100;
    bool denoise = true;
};

class FlameRenderQueue : public QObject {
    Q_OBJECT
    QML_SINGLETON

public:
    explicit FlameRenderQueue(QObject* parent = nullptr);
    ~FlameRenderQueue() override;

    FlameRenderQueue(const FlameRenderQueue&) = delete;
    FlameRenderQueue(FlameRenderQueue&&) = delete;
    FlameRenderQueue& operator=(const FlameRenderQueue&) = delete;
    FlameRenderQueue& operator=(FlameRenderQueue&&) = delete;

    static FlameRenderQueue* instance();
    static FlameRenderQueue* create(QQmlEngine*, QJSEngine*);

    QFuture<QImage> requestRenderToQImage(const rfkt::flame& f, const RenderParams& params);

private:
    QThreadPool m_renderPool;

    roccu::gpu_stream m_stream{};
    roccu::gpu_event m_dnEvent{};

    rfkt::tonemapper m_tonemapper;
    std::unique_ptr<rfkt::denoiser> m_denoiser;
    rfkt::converter m_converter;
};
