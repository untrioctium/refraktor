#include "services/local_flame_directory.hpp"

#include <QDir>
#include <QtConcurrent>
#include <QDebug>

LocalFlameDirectory::RequestID LocalFlameDirectory::listFlames() {
    auto requestID = makeRequestID();
    QtConcurrent::run([this]() {
        QStringList flames;
        for (const auto& entry : QDir("assets/flames_test").entryList()) {
            if (entry.endsWith(".flam3")) {
                flames.append(entry.split("/").last());
            }
        }
        return flames;
    }).then(this, [this, requestID](QStringList flames) {
        // Now on main thread (this object's thread)
        emit flamesListed(requestID, flames);
    });
    return requestID;
}

LocalFlameDirectory::RequestID LocalFlameDirectory::getFlame(const QString& name) {
    auto requestID = makeRequestID();
    QtConcurrent::run([this, name, requestID]() {
        return QString::fromUtf8(QFile("assets/flames_test/" + name).readAll());
    }).then(this, [this, requestID](QString flame) {
        emit flameLoaded(requestID, {flame, Format::XML});
    });
    return requestID;
}

QFuture<QStringList> LocalFlameDirectory::listFlamesAsync() {
    return QtConcurrent::run([this]() {
        QStringList flames;
        for (const auto& entry : QDir("assets/flames_test").entryList()) {
            if (entry.endsWith(".flam3")) {
                flames.append(entry.split("/").last());
            }
        }
        return flames;
    });
}

QFuture<FlameInfo> LocalFlameDirectory::getFlameAsync(const QString& name) {
    return QtConcurrent::run([this, name]() {
        auto path = "assets/flames_test/" + name;
        auto file = QFile(path);
        if (!file.open(QIODevice::ReadOnly)) {
            qWarning() << "Failed to open file " << path << ": " << file.errorString();
            return FlameInfo{QString(), FlameDirectoryService::Format::XML};
        }
        auto data = file.readAll();
        file.close();
        return FlameInfo{QString::fromUtf8(data), FlameDirectoryService::Format::XML};
    });
}

LocalFlameDirectory* LocalFlameDirectory::instance() {
    static LocalFlameDirectory* s_instance = nullptr;
    if (!s_instance) {
        s_instance = new LocalFlameDirectory();
    }
    return s_instance;
}