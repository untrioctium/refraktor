#include <QCoreApplication>
#include <QTcpServer>
#include <QHttpServerRouterRule>
#include <QHttpServer>
#include <QHttpServerResponse>
#include <QHttpServerWebSocketUpgradeResponse>
#include <QHostAddress>
#include <QJsonObject>
#include <QJsonArray>
#include <QBuffer>
#include <QWebSocket>

#include <ctime>

#include <spdlog/spdlog.h>

#include <librefrakt/util/cuda.hpp>
#include <librefrakt/util/filesystem.hpp>

#include <services/local_render_queue.hpp>
#include <services/variation_database.hpp>
#include <services/animation_database.hpp>
#include <services/local_flame_directory.hpp>

#include "stream_session.hpp"

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
    std::srand(std::time(nullptr));
    // print working directory
    qDebug() << "Working directory: " << rfkt::fs::working_directory().string();
    qInstallMessageHandler(qtMessageHandler);

    auto ctx = rfkt::cuda::init();
    auto dev = ctx.device();
    qDebug() << "Using device: " << dev.name();

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


    QCoreApplication app(argc, argv);

    QHttpServer server;

    server.route("/render", QHttpServerRequest::Method::Post, [](const QHttpServerRequest& request) {

        qInfo() << "GET /render";

        auto paramsJson = QJsonDocument::fromJson(request.body()).object();

        auto paramsObj = paramsJson["params"].toObject();

        auto renderParams = RenderParams {
            .dims = {static_cast<unsigned int>(paramsObj["width"].toInt()), static_cast<unsigned int>(paramsObj["height"].toInt())},
            .fps = paramsObj["fps"].toInt(),
            .t = paramsObj["time"].toDouble(),
            .secondsPerLoop = paramsObj["loop_speed"].toDouble(),
            .targetQuality = paramsObj["quality"].toDouble(),
            .maxRenderMillis = static_cast<std::uint32_t>(paramsObj["bin_time"].toInt()),
            .denoise = paramsObj["denoise"].toBool(),
            .upscale = paramsObj["upscale"].toBool(),
        };

        auto flame = rfkt::flame::deserialize(
            nlohmann::json::parse(paramsJson["flame"].toString().toStdString()), 
            AnimationDatabase::instance()->table(), 
            VariationDatabase::instance()->db());

        if (!flame) {
            return QtFuture::makeReadyValueFuture<QHttpServerResponse>(QJsonObject{
                {"error", "Failed to deserialize flame"}
            });
        }

        return LocalRenderQueue::instance()->requestRenderToQImage(*flame, renderParams)
            .then([](QImage image) {
                QByteArray jpegData{};
                QBuffer buffer(&jpegData);
                buffer.open(QBuffer::WriteOnly);
                image.save(&buffer, "JPEG");
                buffer.close();
        
                return QHttpServerResponse(jpegData);
            });


    });

    server.route("/variations", QHttpServerRequest::Method::Get, [](const QHttpServerRequest& request) {

        qInfo() << "GET /variations";

        return QHttpServerResponse(QString::fromStdString(VariationDatabase::instance()->db().serialize()));
    });

    auto flameDirectory = LocalFlameDirectory::instance();
    server.route("/flames", QHttpServerRequest::Method::Get, [flameDirectory](const QHttpServerRequest& request) {
        qInfo() << "GET /flames";
        return flameDirectory->listFlamesAsync().then([](QStringList flames) {
            auto response = QHttpServerResponse(QJsonArray::fromStringList(flames));
            QHttpHeaders headers;
            headers.append("Content-Type", "application/json");
            response.setHeaders(headers);
            return response;
        });
    });

    server.route("/flames/<arg>", QHttpServerRequest::Method::Get, [flameDirectory](const QString& name, const QHttpServerRequest& request) {
        qInfo() << "GET /flames/" << name;
        
        return flameDirectory->getFlameAsync(name)
            .then([](FlameInfo flame) {
                auto response = QHttpServerResponse(flame.data);
                QHttpHeaders headers;

                if (flame.format == FlameDirectoryService::Format::XML) {
                    headers.append("Content-Type", "application/xml");
                } else {
                    headers.append("Content-Type", "application/json");
                }

                response.setHeaders(headers);
                return response;
            });
        });


    server.route("/bananas", []() {
        auto html = rfkt::fs::read_string("assets/static/stream.html");
        return QHttpServerResponse("text/html", QByteArray::fromStdString(std::string(html)));
    });

    server.route("/health", []() {
        return QHttpServerResponse(QJsonObject{
            {"status", "ok"}
        });
    });

    server.addWebSocketUpgradeVerifier(
        &server, [](const QHttpServerRequest& request) {
            if (request.url().path() == u"/stream")
                return QHttpServerWebSocketUpgradeResponse::accept();
            return QHttpServerWebSocketUpgradeResponse::passToNext();
        });

    QObject::connect(&server, &QHttpServer::newWebSocketConnection, [&server, ctx]() {
        auto socket = server.nextPendingWebSocketConnection();
        if (!socket) return;

        new StreamSession(std::move(socket), ctx, &server);
    });

    constexpr static auto port = 3000;
    auto tcpServer = QTcpServer();

    if (!tcpServer.listen(QHostAddress::Any, port)) {
        SPDLOG_ERROR("Failed to listen on port {}", port);
        return 1;
    }

    if(!server.bind(&tcpServer)) {
        SPDLOG_ERROR("Failed to bind server to TCP server");
        return 1;
    }

    SPDLOG_INFO("Server listening on http://localhost:{}", port);

    return app.exec();
}
