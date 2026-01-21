#pragma once

#include <QQuickPaintedItem>
#include <QImage>
#include <QFutureWatcher>

#include <qqmlintegration.h>

class QFileSystemWatcher;
class QTimer;

class FlamePreview : public QQuickPaintedItem {
    Q_OBJECT
    QML_ELEMENT

    // Core properties
    Q_PROPERTY(QString source READ source WRITE setSource NOTIFY sourceChanged)
    Q_PROPERTY(Status status READ status NOTIFY statusChanged)
    Q_PROPERTY(QString errorString READ errorString NOTIFY errorStringChanged)

    // Render parameters
    Q_PROPERTY(qreal time READ time WRITE setTime NOTIFY timeChanged)
    Q_PROPERTY(qreal quality READ quality WRITE setQuality NOTIFY qualityChanged)
    Q_PROPERTY(bool denoise READ denoise WRITE setDenoise NOTIFY denoiseChanged)
    Q_PROPERTY(qreal secondsPerLoop READ secondsPerLoop WRITE setSecondsPerLoop NOTIFY secondsPerLoopChanged)
    Q_PROPERTY(quint32 maxRenderMillis READ maxRenderMillis WRITE setMaxRenderMillis NOTIFY maxRenderMillisChanged)

public:
    enum Status { Null, Loading, Ready, Error };
    Q_ENUM(Status)

    explicit FlamePreview(QQuickItem* parent = nullptr);
    ~FlamePreview() override;

    FlamePreview(const FlamePreview&) = delete;
    FlamePreview(FlamePreview&&) = delete;
    FlamePreview& operator=(const FlamePreview&) = delete;
    FlamePreview& operator=(FlamePreview&&) = delete;

    void paint(QPainter* painter) override;

    // Property accessors
    QString source() const;
    void setSource(const QString& path);

    Status status() const;
    QString errorString() const;

    qreal time() const;
    void setTime(qreal t);

    qreal quality() const;
    void setQuality(qreal q);

    bool denoise() const;
    void setDenoise(bool d);

    qreal secondsPerLoop() const;
    void setSecondsPerLoop(qreal s);

    quint32 maxRenderMillis() const;
    void setMaxRenderMillis(quint32 ms);

    Q_INVOKABLE void reload();

protected:
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override;

signals:
    void sourceChanged();
    void statusChanged();
    void errorStringChanged();
    void timeChanged();
    void qualityChanged();
    void denoiseChanged();
    void secondsPerLoopChanged();
    void maxRenderMillisChanged();

    // Semantic signals for QML convenience
    void loadingStarted();
    void loadingFinished();
    void loadingFailed(const QString& error);

private:
    void scheduleRender();
    void cancelPendingRender();
    void startRender();
    void setError(const QString& error);

    QString m_source;
    Status m_status = Null;
    QString m_errorString;

    qreal m_time = 0.0;
    qreal m_quality = 16.0;
    bool m_denoise = true;
    qreal m_secondsPerLoop = 5.0;
    quint32 m_maxRenderMillis = 100;

    QImage m_image;
    QFutureWatcher<QImage>* m_watcher = nullptr;
    QFileSystemWatcher* m_fileWatcher = nullptr;
    QTimer* m_debounceTimer = nullptr;
    qreal m_pendingDpr = 1.0;
};
