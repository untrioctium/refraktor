#pragma once

#include "services/flame_directory_service.hpp"

#include <QNetworkAccessManager>

class RemoteFlameDirectory : public FlameDirectoryService {
    Q_OBJECT

public:

    explicit RemoteFlameDirectory(QObject* parent = nullptr) : FlameDirectoryService(parent) {}

    RequestID listFlames() override;
    RequestID getFlame(const QString& name) override;

    QFuture<FlameInfo> getFlameAsync(const QString& name) override;
    QFuture<QStringList> listFlamesAsync() override;

    static RemoteFlameDirectory* instance();

private:

    QNetworkAccessManager m_networkManager;
};