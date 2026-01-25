#pragma once

#include <QObject>
#include <QThreadPool>
#include <QFuture>
#include <QImage>
#include <QQmlEngine>

#include "services/flame_render_service.hpp"

#include <librefrakt/flame_types.hpp>
#include <librefrakt/image/tonemapper.hpp>
#include <librefrakt/image/converter.hpp>
#include <librefrakt/interface/denoiser.hpp>
#include <librefrakt/vector_types.hpp>


#include <roccu_cpp_types.hpp>

#include <qqmlintegration.h>

#include <memory>

class LocalRenderQueue : public QObject, public FlameRenderService {
    Q_OBJECT
    QML_SINGLETON

public:
    explicit LocalRenderQueue(QObject* parent = nullptr);
    ~LocalRenderQueue() override;

    LocalRenderQueue(const LocalRenderQueue&) = delete;
    LocalRenderQueue(LocalRenderQueue&&) = delete;
    LocalRenderQueue& operator=(const LocalRenderQueue&) = delete;
    LocalRenderQueue& operator=(LocalRenderQueue&&) = delete;

    static LocalRenderQueue* instance();
    static LocalRenderQueue* create(QQmlEngine*, QJSEngine*);

    QFuture<QImage> requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) override;

private:
    QThreadPool m_renderPool;

    roccu::gpu_stream m_stream{};
    roccu::gpu_event m_dnEvent{};

    rfkt::tonemapper m_tonemapper;
    std::unique_ptr<rfkt::denoiser> m_denoiser;
    std::unique_ptr<rfkt::denoiser> m_upscaleDenoiser;
    rfkt::converter m_converter;
};
