#include <QGuiApplication> 
#include <QCursor>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QFuture>
#include <QFutureWatcher>
#include <QTConcurrent/QtConcurrent>
#include <QQuickPaintedItem>
#include <QQuickWindow>
#include <QPainter>
#include <QFileSystemWatcher>
#include <QFile>
#include <QTimer>

#include <librefrakt/flame_compiler.hpp>
#include <librefrakt/image/tonemapper.hpp>
#include <librefrakt/interface/denoiser.hpp>
#include <librefrakt/image/converter.hpp>
#include <librefrakt/anima.hpp>

#include <qqmlintegration.h>
#include <qtmetamacros.h>
#include <readerwriterqueue.h>

#include <roccu_cpp_types.hpp>

class CursorHelper : public QObject {
    Q_OBJECT
public:
    explicit CursorHelper(QObject* parent = nullptr) : QObject(parent) {}
    
    Q_INVOKABLE QPointF cursorPos() {
        return QCursor::pos();
    }
    
    Q_INVOKABLE void setCursorPos(qreal x, qreal y) {
        QCursor::setPos(QPoint(static_cast<int>(x), static_cast<int>(y)));
    }
};

class VariationDatabase : public QObject {
    Q_OBJECT
    QML_SINGLETON

    Q_PROPERTY(int variationCount READ variationCount CONSTANT)
    Q_PROPERTY(QStringList variationNames READ variationNames CONSTANT)

public:

    rfkt::flamedb& db() { return m_db; }
    const rfkt::flamedb& db() const { return m_db; }

    int variationCount() const { return static_cast<int>(m_db.variations().size());}
    QStringList variationNames() const { return m_variationNames; }

    Q_INVOKABLE bool hasVariation(const QString& name) const { return m_db.is_variation(name.toStdString()); }

    static VariationDatabase* create(QQmlEngine*, QJSEngine*) { return instance();}

    static VariationDatabase* instance() {
        static VariationDatabase* instance = nullptr;
        if(!instance) {
            instance = new VariationDatabase();
        }
        return instance;
    }

    static void initialize(const QString& configPath) {
        rfkt::initialize(instance()->m_db, configPath.toStdString());
    }

private:

    explicit VariationDatabase(QObject* parent = nullptr) : QObject(parent) {}

    rfkt::flamedb m_db;
    QStringList m_variationNames;
};

class AnimationDatabase : public QObject {
    Q_OBJECT
    QML_SINGLETON

public:

    rfkt::function_table& table() { return m_table; }

    static AnimationDatabase* instance() {
        static AnimationDatabase* instance = nullptr;
        if(!instance) {
            instance = new AnimationDatabase();
        }
        return instance;
    }

private:

    explicit AnimationDatabase(QObject* parent = nullptr) : QObject(parent) {}
    
    rfkt::function_table m_table;
};

class KernelCompileQueue : public QObject {
    Q_OBJECT
    QML_SINGLETON

public:

    explicit KernelCompileQueue(QObject* parent = nullptr) : 
        QObject(parent),
        m_kernelManager(std::make_shared<ezrtc::compiler>(
            std::make_shared<ezrtc::cache_adaptors::zlib>(
                std::make_shared<ezrtc::cache_adaptors::guarded>(
                    std::make_shared<ezrtc::sqlite_cache>(
                        (rfkt::fs::user_local_directory() / "kernel.sqlite3").string().c_str()
                    )
                )
            )
        )),
        m_flameCompiler(m_kernelManager)
    {
        m_compilePool.setMaxThreadCount(1);
        m_compilePool.setExpiryTimeout(-1);

        auto ctx = roccu::context::current();
        m_compilePool.start([ctx]() {
            ctx.make_current();
        });
    }

    static KernelCompileQueue* instance() {
        static KernelCompileQueue* instance = nullptr;
        if(!instance) {
            instance = new KernelCompileQueue();
        }
        return instance;
    }

    QFuture<ezrtc::compiler::result> requestCompile( ezrtc::spec&& spec) {
        return QtConcurrent::run(&m_compilePool, [this, spec = std::move(spec)]() mutable {
            return m_kernelManager->compile(std::move(spec));
        });
    }

    QFuture<rfkt::flame_compiler::result> requestCompile(
        const rfkt::flamedb& fdb, 
        const rfkt::flame& f, 
        rfkt::precision prec) 
    {
        auto thunk = m_flameCompiler.prepare_flame_kernel(fdb, prec, f);
        return QtConcurrent::run(&m_compilePool, [thunk = std::move(thunk)]() mutable {
            return thunk();
        });
    }

    static std::shared_ptr<ezrtc::compiler> kernelManagerInstance() {
        return instance()->m_kernelManager;
    }

private:

    std::shared_ptr<ezrtc::compiler> m_kernelManager;
    QThreadPool m_compilePool;
    rfkt::flame_compiler m_flameCompiler;
};

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
    explicit FlameRenderQueue(QObject* parent = nullptr)
        : QObject(parent), 
          renderPool(this), 
          tm(*KernelCompileQueue::kernelManagerInstance()), 
          dn(rfkt::denoiser::make("rfkt::optix_denoise", uint2{512, 512}, rfkt::denoiser_flag::tiled, stream)), 
          conv(*KernelCompileQueue::kernelManagerInstance())
    {
        renderPool.setMaxThreadCount(1);
        renderPool.setExpiryTimeout(-1);

        auto ctx = roccu::context::current();

        renderPool.start([ctx]() {
            ctx.make_current();
        });
    }

    ~FlameRenderQueue() override {
        renderPool.waitForDone();
    }

    FlameRenderQueue(const FlameRenderQueue&) = delete;
    FlameRenderQueue(FlameRenderQueue&&) = delete;
    FlameRenderQueue& operator=(const FlameRenderQueue&) = delete;
    FlameRenderQueue& operator=(FlameRenderQueue&&) = delete;

    static FlameRenderQueue* instance() {
        static FlameRenderQueue* s_instance = nullptr;
        if (!s_instance) {
            s_instance = new FlameRenderQueue();
        }
        return s_instance;
    }

    static FlameRenderQueue* create(QQmlEngine*, QJSEngine*) {
        return instance();
    }

    QFuture<QImage> requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) {

        qDebug() << "Requesting render to QImage. Params:";
        qDebug() << "FPS: " << params.fps;
        qDebug() << "Seconds per loop: " << params.secondsPerLoop;
        qDebug() << "Target quality: " << params.targetQuality;
        qDebug() << "Max render millis: " << params.maxRenderMillis;
        qDebug() << "Denoise: " << params.denoise;
        qDebug() << "Dimensions: " << params.dims.x << "x" << params.dims.y;
        qDebug() << "Time: " << params.t;

        auto kernel_future = KernelCompileQueue::instance()->requestCompile(VariationDatabase::instance()->db(), f, rfkt::precision::f32);
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

        return QtConcurrent::run(&renderPool, 
            [kernel_future = std::move(kernel_future), 
               samples = std::move(samples), 
               gbv = std::move(gbv),
               params = params,
               &stream = this->stream,
               &tm = this->tm,
               dn = this->dn.get(),
               &dn_event = this->dn_event,
               &conv = this->conv]() mutable {

                auto start = std::chrono::high_resolution_clock::now();

                auto tonemapped = roccu::gpu_image<half3>(params.dims, stream);
                auto denoised = roccu::gpu_image<half3>(params.dims, stream);
                auto converted = roccu::gpu_image<uchar4>(params.dims, stream);

                auto kernel_result = kernel_future.takeResult();

                if(!kernel_result.kernel.has_value()) {
                    qDebug() << "Failed to compile kernel: " << kernel_result.log;
                    return QImage();
                }

                auto& kernel = kernel_result.kernel.value();

                auto state = kernel.warmup(stream, samples, params.dims, 0xdeadbeef, 100);
                auto bin_result = kernel.bin(stream, state, { .millis = params.maxRenderMillis, .quality = params.targetQuality }).get();

                tm.run(state.bins, tonemapped, { bin_result.quality, gbv.gamma, gbv.brightness, gbv.vibrancy }, stream);

                if(params.denoise) {
                    dn->denoise(tonemapped, denoised, dn_event);
                    stream.wait_for(dn_event);
                }
                else {
                    denoised = std::move(tonemapped);
                }

                conv.to_uchar4(denoised, converted, stream);

                stream.sync();

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                qDebug() << "Render time: " << duration << "ms";

                auto host_converted = converted.to_host_flat();

                return QImage(reinterpret_cast<uchar*>(host_converted.data()), params.dims.x, params.dims.y, QImage::Format_RGBA8888).copy();

        });

    }

private:
    QThreadPool renderPool;

    roccu::gpu_stream stream{};
    roccu::gpu_event dn_event{};
    
    rfkt::tonemapper tm;
    std::unique_ptr<rfkt::denoiser> dn;
    rfkt::converter conv;

};

class FlamePreview : public QQuickPaintedItem {
    Q_OBJECT

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

    explicit FlamePreview(QQuickItem* parent = nullptr)
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

    ~FlamePreview() override {
        cancelPendingRender();
    }

    FlamePreview(const FlamePreview&) = delete;
    FlamePreview(FlamePreview&&) = delete;
    FlamePreview& operator=(const FlamePreview&) = delete;
    FlamePreview& operator=(FlamePreview&&) = delete;

    void paint(QPainter* painter) override {
        if (m_image.isNull()) return;

        // Image is rendered at physical pixel size with DPR set,
        // so Qt will draw it at the correct logical size
        painter->drawImage(0, 0, m_image);
    }

    // Property accessors
    QString source() const { return m_source; }
    void setSource(const QString& path) {
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

    Status status() const { return m_status; }
    QString errorString() const { return m_errorString; }

    qreal time() const { return m_time; }
    void setTime(qreal t) {
        if (qFuzzyCompare(m_time, t)) return;
        m_time = t;
        emit timeChanged();
        scheduleRender();
    }

    qreal quality() const { return m_quality; }
    void setQuality(qreal q) {
        if (qFuzzyCompare(m_quality, q)) return;
        m_quality = q;
        emit qualityChanged();
        scheduleRender();
    }

    bool denoise() const { return m_denoise; }
    void setDenoise(bool d) {
        if (m_denoise == d) return;
        m_denoise = d;
        emit denoiseChanged();
        scheduleRender();
    }

    qreal secondsPerLoop() const { return m_secondsPerLoop; }
    void setSecondsPerLoop(qreal s) {
        if (qFuzzyCompare(m_secondsPerLoop, s)) return;
        m_secondsPerLoop = s;
        emit secondsPerLoopChanged();
        scheduleRender();
    }

    quint32 maxRenderMillis() const { return m_maxRenderMillis; }
    void setMaxRenderMillis(quint32 ms) {
        if (m_maxRenderMillis == ms) return;
        m_maxRenderMillis = ms;
        emit maxRenderMillisChanged();
        scheduleRender();
    }

    Q_INVOKABLE void reload() {
        if (!m_source.isEmpty()) {
            startRender();
        }
    }

protected:
    void geometryChange(const QRectF& newGeometry, const QRectF& oldGeometry) override {
        QQuickPaintedItem::geometryChange(newGeometry, oldGeometry);

        // Re-render if size changed
        if (newGeometry.size() != oldGeometry.size() && 
            newGeometry.width() > 0 && newGeometry.height() > 0) {
            scheduleRender();
        }
    }

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
    void scheduleRender() {
        // Only schedule if we have a valid source and are visible
        if (m_source.isEmpty()) return;
        m_debounceTimer->start();
    }

    void cancelPendingRender() {
        if (m_watcher) {
            m_watcher->cancel();
            disconnect(m_watcher, nullptr, this, nullptr);
            m_watcher->deleteLater();
            m_watcher = nullptr;
        }
        m_debounceTimer->stop();
    }

    void startRender() {
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

    void setError(const QString& error) {
        m_status = Error;
        m_errorString = error;
        m_image = QImage();
        emit statusChanged();
        emit errorStringChanged();
        emit loadingFailed(error);
        update();
    }

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

int main(int argc, char *argv[])
{

    for(auto& dn_info: rfkt::denoiser::names()) {
        qDebug() << "Denoiser: " << dn_info;
    }

    auto ctx = rfkt::cuda::init();
    auto dev = ctx.device();
    qDebug() << "Using device: " << dev.name();

    try {
        VariationDatabase::initialize("config");
    }
    catch(const std::exception& e) {
        qDebug() << "Failed to initialize flame system: " << e.what();
        return 1;
    }
    catch(const flang::parse_error& e) {
        qDebug() << "Failed to parse variations file: " << e.what();
        return 1;
    }

    AnimationDatabase::instance()->table().add_or_update("increase", {
        {{"per_loop", {rfkt::func_info::arg_t::decimal, 360.0}}},
        "return iv + t * per_loop"
    });

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    qmlRegisterType<FlamePreview>("Refrakt", 1, 0, "FlamePreview");

    const QUrl url(QStringLiteral("qrc:/qt/qml/Refrakt/qml/main.qml"));
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection
    );

    engine.load(url);

    return app.exec();
}

#include "main.moc"