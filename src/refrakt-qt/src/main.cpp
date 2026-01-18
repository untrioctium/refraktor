#include <QGuiApplication> 
#include <QCursor>
#include <QQmlApplicationEngine>
#include <QUrl>

class CursorHelper : public QObject {
    Q_OBJECT
public:
    explicit CursorHelper(QObject* parent = nullptr) : QObject(parent) {}
    
    Q_INVOKABLE QPointF cursorPos() {
        return QCursor::pos();
    }
    
    Q_INVOKABLE void setCursorPos(qreal x, qreal y) {
        QCursor::setPos(QPoint(x, y));
    }
};

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;

    qmlRegisterSingletonType<CursorHelper>("Refrakt", 1, 0, "CursorHelper", [](QQmlEngine* engine, QJSEngine* scriptEngine) -> QObject* {
        Q_UNUSED(engine);
        Q_UNUSED(scriptEngine);
        return new CursorHelper();
    });

    const QUrl url(QStringLiteral("qrc:/qt/qml/refrakt-qt-qml/qml/main.qml"));
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection
    );

    engine.load(url);

    return app.exec();
}

#include "main.moc"