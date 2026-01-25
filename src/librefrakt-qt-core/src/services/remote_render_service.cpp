#include "remote_render_service.hpp"

#include <QJsonObject>
#include <QJsonDocument>
#include <QNetworkReply>

QFuture<QImage> RemoteRenderService::requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) {

    QNetworkRequest request(QUrl("http://localhost:3000/render"));

    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonObject paramsJson;
    paramsJson["width"] = static_cast<int>(params.dims.x);
    paramsJson["height"] = static_cast<int>(params.dims.y);
    paramsJson["quality"] = params.targetQuality;
    paramsJson["time"] = params.t;
    paramsJson["fps"] = params.fps;
    paramsJson["loop_speed"] = params.secondsPerLoop;
    paramsJson["bin_time"] = static_cast<int>(params.maxRenderMillis);
    paramsJson["denoise"] = params.denoise;
    paramsJson["upscale"] = params.upscale;

    auto flameJsonRaw = QString::fromStdString(f.serialize().dump());

    QJsonObject requestJson;
    requestJson["params"] = paramsJson;
    requestJson["flame"] = flameJsonRaw;

    QByteArray requestJsonRaw = QJsonDocument(requestJson).toJson(QJsonDocument::Compact);

    QNetworkReply* reply = m_networkManager.post(request, requestJsonRaw);

    return QtFuture::connect(reply, &QNetworkReply::finished).then([reply]() {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            qWarning() << "Remote render service error:" << reply->errorString();
            return QImage();
        }

        auto jpegData = reply->readAll();
        return QImage::fromData(jpegData).copy();
    });
}

RemoteRenderService* RemoteRenderService::instance() {
    static RemoteRenderService* s_instance = nullptr;
    if (!s_instance) {
        s_instance = new RemoteRenderService();
    }
    return s_instance;
}