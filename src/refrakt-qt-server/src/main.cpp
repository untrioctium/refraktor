#include <QCoreApplication>
#include <QTcpServer>
#include <QHttpServerRouterRule>
#include <QHttpServer>
#include <QHttpServerResponse>
#include <QHostAddress>
#include <QJsonObject>

#include <spdlog/spdlog.h>

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    QHttpServer server;

    server.route("/", []() {
        return "Hello, World!";
    });

    server.route("/health", []() {
        return QHttpServerResponse(QJsonObject{
            {"status", "ok"}
        });
    });

    auto port = 3000;
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
