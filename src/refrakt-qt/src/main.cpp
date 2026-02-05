#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QDebug>

//#include <librefrakt/util/cuda.hpp>

#include "services/flame_directory_service.hpp"
#include "services/local_flame_directory.hpp"
#include "services/remote_flame_directory.hpp"
#include "services/variation_database.hpp"
#include "services/animation_database.hpp"
#include "components/flame_preview.hpp"
#include "services/service_locator.hpp"

#include <stacktrace>

void qtMessageHandler(QtMsgType type, const QMessageLogContext& ctx, const QString& msg)
{

    auto source_info = spdlog::source_loc{ ctx.file, ctx.line, ctx.function };
    auto logger = spdlog::default_logger_raw();
    switch (type) {
    case QtDebugMsg:    logger->log(source_info, spdlog::level::debug, "[Qt] {}", msg.toStdString()); break;
    case QtInfoMsg:     logger->log(source_info, spdlog::level::info, "[Qt] {}", msg.toStdString()); break;
    case QtWarningMsg:  logger->log(source_info, spdlog::level::warn, "[Qt] {}", msg.toStdString()); break;
    case QtCriticalMsg: logger->log(source_info, spdlog::level::err, "[Qt] {}", msg.toStdString()); break;
    case QtFatalMsg:    logger->log(source_info, spdlog::level::critical, "[Qt] {}", msg.toStdString()); std::abort();
    }

    auto trace = std::stacktrace::current();
    for (const auto& frame : trace) {
        if(frame.source_file().contains("refraktor\\src")) {
            spdlog::info("[{}:{}]", frame.source_file(), frame.source_line(), frame.description());
        }
    }
}

int main(int argc, char* argv[])
{



    qInstallMessageHandler(qtMessageHandler);
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    qRegisterMetaType<FlameDirectoryService::RequestID>("RequestID");

    //auto ctx = rfkt::cuda::init();
    //auto dev = ctx.device();
    //qDebug() << "Using device: " << dev.name();

    try {
        VariationDatabase::initialize("config");
    } catch (const std::exception& e) {
        qDebug() << "Failed to initialize flame system: " << e.what();
        return 1;
    } catch (const flang::parse_error& e) {
        qDebug() << "Failed to parse variations file: " << e.what();
        return 1;
    }

    auto sl = ServiceLocator::instance();
    sl->provide<FlameDirectoryService>(new RemoteFlameDirectory());

    AnimationDatabase::instance()->table().add_or_update("increase", {
        {{"per_loop", {rfkt::func_info::arg_t::decimal, 360.0}}},
        "return iv + t * per_loop"
    });

    qmlRegisterType<FlamePreview>("Refrakt", 1, 0, "FlamePreview");

    qmlRegisterSingletonType<FlameDirectoryService>("Refrakt", 1, 0, "FlameDirectoryService", [](QQmlEngine* engine, QJSEngine* scriptEngine) -> QObject* {
        return ServiceLocator::instance()->get<FlameDirectoryService>();
    });

    const QUrl url(QStringLiteral("qrc:/qt/qml/Refrakt/qml/main.qml"));
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject* obj, const QUrl& objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}
