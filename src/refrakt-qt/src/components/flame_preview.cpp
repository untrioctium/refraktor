#include "flame_preview.hpp"

#include "services/variation_database.hpp"
#include "flame_render_queue.hpp"

#include <QFileSystemWatcher>
#include <QTimer>
#include <QFile>
#include <QPainter>
#include <QQuickWindow>

FlamePreview::FlamePreview(QQuickItem* parent)
    : QQuickPaintedItem(parent)
    , m_fileWatcher(new QFileSystemWatcher(this))
    , m_debounceTimer(new QTimer(this))
{
    // Debounce rapid file changes (e.g., during save)
    m_debounceTimer->setSingleShot(true);
    m_debounceTimer->setInterval(100);
    connect(m_debounceTimer, &QTimer::timeout, this, &FlamePreview::startRender);

    connect(m_fileWatcher, &QFileSystemWatcher::fileChanged, this, [this](const QString& path) {
        // Re-add the file to watch (some editors remove and recreate files on save)
        if (!m_fileWatcher->files().contains(path) && QFile::exists(path)) {
            m_fileWatcher->addPath(path);
        }
        // Debounce the render
        m_debounceTimer->start();
    });
}

FlamePreview::~FlamePreview() {
    cancelPendingRender();
}

void FlamePreview::paint(QPainter* painter) {
    if (m_image.isNull()) return;

    // Image is rendered at physical pixel size with DPR set,
    // so Qt will draw it at the correct logical size
    painter->drawImage(0, 0, m_image);
}

QString FlamePreview::source() const {
    return m_source;
}

void FlamePreview::setSource(const QString& path) {
    if (m_source == path) return;

    // Stop watching old file
    if (!m_source.isEmpty() && m_fileWatcher->files().contains(m_source)) {
        m_fileWatcher->removePath(m_source);
    }

    m_source = path;
    emit sourceChanged();

    if (!m_source.isEmpty()) {
        // Start watching new file
        if (QFile::exists(m_source)) {
            m_fileWatcher->addPath(m_source);
        }
        scheduleRender();
    } else {
        m_status = Null;
        m_image = QImage();
        emit statusChanged();
        update();
    }
}

FlamePreview::Status FlamePreview::status() const {
    return m_status;
}

QString FlamePreview::errorString() const {
    return m_errorString;
}

qreal FlamePreview::time() const {
    return m_time;
}

bool FlamePreview::upscale() const {
    return m_upscale;
}

void FlamePreview::setUpscale(bool u) {
    if (m_upscale == u) return;
    m_upscale = u;
    emit upscaleChanged();
    scheduleRender();
}

void FlamePreview::setTime(qreal t) {
    if (qFuzzyCompare(m_time, t)) return;
    m_time = t;
    emit timeChanged();
    scheduleRender();
}

qreal FlamePreview::quality() const {
    return m_quality;
}

void FlamePreview::setQuality(qreal q) {
    if (qFuzzyCompare(m_quality, q)) return;
    m_quality = q;
    emit qualityChanged();
    scheduleRender();
}

bool FlamePreview::denoise() const {
    return m_denoise;
}

void FlamePreview::setDenoise(bool d) {
    if (m_denoise == d) return;
    m_denoise = d;
    emit denoiseChanged();
    scheduleRender();
}

qreal FlamePreview::secondsPerLoop() const {
    return m_secondsPerLoop;
}

void FlamePreview::setSecondsPerLoop(qreal s) {
    if (qFuzzyCompare(m_secondsPerLoop, s)) return;
    m_secondsPerLoop = s;
    emit secondsPerLoopChanged();
    scheduleRender();
}

quint32 FlamePreview::maxRenderMillis() const {
    return m_maxRenderMillis;
}

void FlamePreview::setMaxRenderMillis(quint32 ms) {
    if (m_maxRenderMillis == ms) return;
    m_maxRenderMillis = ms;
    emit maxRenderMillisChanged();
    scheduleRender();
}

void FlamePreview::reload() {
    if (!m_source.isEmpty()) {
        startRender();
    }
}

void FlamePreview::geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) {
    QQuickPaintedItem::geometryChange(newGeometry, oldGeometry);

    // Re-render if size changed
    if (newGeometry.size() != oldGeometry.size() &&
        newGeometry.width() > 0 && newGeometry.height() > 0) {
        scheduleRender();
    }
}

void FlamePreview::scheduleRender() {
    // Only schedule if we have a valid source and are visible
    if (m_source.isEmpty()) return;
    m_debounceTimer->start();
}

void FlamePreview::cancelPendingRender() {
    if (m_watcher) {
        m_watcher->cancel();
        disconnect(m_watcher, nullptr, this, nullptr);
        m_watcher->deleteLater();
        m_watcher = nullptr;
    }
    m_debounceTimer->stop();
}

void FlamePreview::startRender() {
    cancelPendingRender();

    if (m_source.isEmpty()) {
        return;  // No source yet, not an error
    }

    // Don't render if size is invalid or no window
    if (width() <= 0 || height() <= 0 || !window()) {
        return;
    }

    // Get device pixel ratio for high-DPI rendering
    qreal dpr = window()->effectiveDevicePixelRatio();

    // Read file contents
    QFile file(m_source);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        setError(QString("Failed to open file: %1").arg(file.errorString()));
        return;
    }

    QString content = QString::fromUtf8(file.readAll());
    file.close();

    // Parse the flame
    auto& fdb = VariationDatabase::instance()->db();
    auto flameOpt = rfkt::import_flam3(fdb, content.toStdString());

    if (!flameOpt.has_value()) {
        setError("Failed to parse flame file");
        return;
    }

    m_status = Loading;
    m_errorString.clear();
    emit statusChanged();
    emit errorStringChanged();
    emit loadingStarted();

    // Build render params using physical pixel size
    RenderParams params;
    params.dims = {
        static_cast<uint>(width() * dpr),
        static_cast<uint>(height() * dpr)
    };
    params.t = m_time;
    params.secondsPerLoop = m_secondsPerLoop;
    params.targetQuality = m_quality;
    params.maxRenderMillis = m_maxRenderMillis;
    params.denoise = m_denoise;
    params.upscale = m_upscale;
    // Store DPR to apply when image arrives
    m_pendingDpr = dpr;

    // Request render
    auto future = FlameRenderQueue::instance()->requestRenderToQImage(
        flameOpt.value(), params);

    m_watcher = new QFutureWatcher<QImage>(this);
    connect(m_watcher, &QFutureWatcher<QImage>::finished, this, [this]() {
        if (!m_watcher) return;

        QImage result = m_watcher->result();
        m_watcher->deleteLater();
        m_watcher = nullptr;

        if (result.isNull()) {
            setError("Render failed - kernel compilation error");
            return;
        }

        // Set device pixel ratio so Qt draws at correct logical size
        result.setDevicePixelRatio(m_pendingDpr);

        m_image = result;
        m_status = Ready;
        emit statusChanged();
        emit loadingFinished();
        update();
    });
    m_watcher->setFuture(future);
}

void FlamePreview::setError(const QString& error) {
    m_status = Error;
    m_errorString = error;
    m_image = QImage();
    emit statusChanged();
    emit errorStringChanged();
    emit loadingFailed(error);
    update();
}
