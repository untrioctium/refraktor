#pragma once

#include <QObject>
#include <QImage>
#include <QFuture>

#include <librefrakt/flame_types.hpp>
#include <librefrakt/vector_types.hpp>

struct RenderParams {
    rfkt::uint2 dims = {640, 480};
    int fps = 30;
    double t = 0.0;
    double secondsPerLoop = 5.0;
    double targetQuality = 16;
    std::uint32_t maxRenderMillis = 100;
    bool denoise = true;
    bool upscale = false;
};

struct FlameRenderService {
    virtual QFuture<QImage> requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) = 0;

    virtual ~FlameRenderService() = default;
};