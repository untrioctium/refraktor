#include "remote_flame_directory.hpp"

#include <QThreadPool>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QtConcurrent>
#include <QJsonObject>
#include <QJsonArray>

static constexpr const char* BASE_URL = "http://localhost:3000";

RemoteFlameDirectory::RequestID RemoteFlameDirectory::listFlames() {
    auto requestID = makeRequestID();

    QNetworkRequest request(QUrl(QString::fromUtf8(BASE_URL) + "/flames"));
    QNetworkReply* reply = m_networkManager.get(request);

    QFuture<void> future = QtFuture::connect(reply, &QNetworkReply::finished).then([this, requestID, reply]() -> void {
        reply->deleteLater();
        
        if (reply->error() != QNetworkReply::NoError) {
            emit error(requestID, reply->errorString());
        }
        auto body = reply->readAll();
        auto json = QJsonDocument::fromJson(body).array();
        QStringList flames{};
        for (const auto& name: json) {
            flames.append(name.toString());
        }
        emit flamesListed(requestID, flames);
    });

    return requestID;
}

RemoteFlameDirectory::RequestID RemoteFlameDirectory::getFlame(const QString& name) {
    auto requestID = makeRequestID();

    QThreadPool::globalInstance()->start([this, requestID, name]() -> void {
        QNetworkRequest request(QUrl(QString::fromUtf8(BASE_URL) + "/flames/" + name));
        QNetworkReply* reply = m_networkManager.get(request);

        if (reply->error() != QNetworkReply::NoError) {
            emit error(requestID, reply->errorString());
        }

        auto body = reply->readAll();

        auto contentType = reply->header(QNetworkRequest::ContentTypeHeader).toString();
        auto format = contentType.contains("application/xml") ? Format::XML : Format::JSON;

        emit flameLoaded(requestID, {QString::fromUtf8(body), format});
    });

    return requestID;
}

QFuture<FlameInfo> RemoteFlameDirectory::getFlameAsync(const QString& name) {

    QNetworkRequest request(QUrl(QString::fromUtf8(BASE_URL) + "/flames/" + name));
    QNetworkReply* reply = m_networkManager.get(request);

    return QtFuture::connect(reply, &QNetworkReply::finished).then([this, reply]() -> FlameInfo {
        
        reply->deleteLater();
        
        auto body = reply->readAll();
        auto contentType = reply->header(QNetworkRequest::ContentTypeHeader).toString();
        auto format = contentType.contains("application/xml") ? FlameDirectoryService::Format::XML : FlameDirectoryService::Format::JSON;
        return {QString::fromUtf8(body), format};
    });
}

QFuture<QStringList> RemoteFlameDirectory::listFlamesAsync() {
    return QtConcurrent::run([this]() -> QStringList {
        QNetworkRequest request(QUrl(QString::fromUtf8(BASE_URL) + "/flames"));
        QNetworkReply* reply = m_networkManager.get(request);
        
        QJsonDocument json = QJsonDocument::fromJson(reply->readAll());
        QJsonArray array = json.array();
        QStringList flames;
        for (const auto& name : array) {
            flames.append(name.toString());
        }
        return flames;
    });
}