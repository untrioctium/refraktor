#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QDebug>

//#include <librefrakt/util/cuda.hpp>

#include "services/variation_database.hpp"
#include "services/animation_database.hpp"
#include "components/flame_preview.hpp"

void qtMessageHandler(QtMsgType type, const QMessageLogContext&, const QString& msg)
{
    auto str = msg.toStdString();
    switch (type) {
    case QtDebugMsg:    SPDLOG_DEBUG("[Qt] {}", str); break;
    case QtInfoMsg:     SPDLOG_INFO("[Qt] {}", str); break;
    case QtWarningMsg:  SPDLOG_WARN("[Qt] {}", str); break;
    case QtCriticalMsg: SPDLOG_ERROR("[Qt] {}", str); break;
    case QtFatalMsg:    SPDLOG_CRITICAL("[Qt] {}", str); std::abort();
    }
}

int main(int argc, char* argv[])
{

    qInstallMessageHandler(qtMessageHandler);
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
        [url](QObject* obj, const QUrl& objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}
