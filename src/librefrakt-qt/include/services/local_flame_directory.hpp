#pragma once

#include "services/flame_directory_service.hpp"

#include <QFuture>

class LocalFlameDirectory : public FlameDirectoryService {
    Q_OBJECT

public:

    explicit LocalFlameDirectory(QObject* parent = nullptr) : FlameDirectoryService(parent) {}

    RequestID listFlames() override;
    RequestID getFlame(const QString& name) override;

    QFuture<QStringList> listFlamesAsync() override;
    QFuture<FlameInfo> getFlameAsync(const QString& name);

    static LocalFlameDirectory* instance();

};