#pragma once

#include <QStringList>
#include <QFuture>
#include <qtmetamacros.h>

class FlameInfo;

class FlameDirectoryService : public QObject {

    Q_OBJECT
public:

    using RequestID = quint64;

    enum class Format {
        XML,
        JSON
    };
    Q_ENUM(Format)

    explicit FlameDirectoryService(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~FlameDirectoryService() = default;

    Q_INVOKABLE virtual RequestID listFlames() = 0;
    Q_INVOKABLE virtual RequestID getFlame(const QString& name) = 0;

    virtual QFuture<FlameInfo> getFlameAsync(const QString& name) = 0;
    virtual QFuture<QStringList> listFlamesAsync() = 0;

signals:
    void flamesListed(RequestID requestID, const QStringList& flames);
    void flameLoaded(RequestID requestID, const FlameInfo& flame);

    void error(RequestID requestID, const QString& error);

protected:

    static quint64 makeRequestID() {
        static std::atomic_uint64_t requestID = 0;
        return requestID.fetch_add(1);
    }

};

class FlameInfo {
    Q_GADGET

    Q_PROPERTY(QString data MEMBER data CONSTANT)
    Q_PROPERTY(FlameDirectoryService::Format format MEMBER format CONSTANT)

public:
    QString data;
    FlameDirectoryService::Format format;
};