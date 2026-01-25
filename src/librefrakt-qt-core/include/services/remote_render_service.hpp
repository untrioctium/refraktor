#include "services/flame_render_service.hpp"

#include <QQmlEngine>
#include <QNetworkAccessManager>

class RemoteRenderService : public QObject, public FlameRenderService {
    Q_OBJECT

public:

    explicit RemoteRenderService(QObject* parent = nullptr) : QObject(parent) {}
    ~RemoteRenderService() override = default;

    RemoteRenderService(const RemoteRenderService&) = delete;
    RemoteRenderService(RemoteRenderService&&) = delete;
    RemoteRenderService& operator=(const RemoteRenderService&) = delete;
    RemoteRenderService& operator=(RemoteRenderService&&) = delete;

    static RemoteRenderService* instance();
    static RemoteRenderService* create(QQmlEngine*, QJSEngine*);

    QFuture<QImage> requestRenderToQImage(const rfkt::flame& f, const RenderParams& params) override;

private:

    QNetworkAccessManager m_networkManager;
};