#pragma once

#include <QObject>
#include <QStringList>
#include <QQmlEngine>

#include <librefrakt/flame_info.hpp>

#include <qqmlintegration.h>

class VariationDatabase : public QObject {
    Q_OBJECT
    QML_SINGLETON

    Q_PROPERTY(int variationCount READ variationCount CONSTANT)
    Q_PROPERTY(QStringList variationNames READ variationNames CONSTANT)

public:
    rfkt::flamedb& db() { return m_db; }
    const rfkt::flamedb& db() const { return m_db; }

    int variationCount() const { return static_cast<int>(m_db.variations().size()); }
    QStringList variationNames() const { return m_variationNames; }

    Q_INVOKABLE bool hasVariation(const QString& name) const { 
        return m_db.is_variation(name.toStdString()); 
    }

    static VariationDatabase* create(QQmlEngine*, QJSEngine*) { return instance(); }

    static VariationDatabase* instance() {
        static VariationDatabase* instance = nullptr;
        if (!instance) {
            instance = new VariationDatabase();
        }
        return instance;
    }

    static void initialize(const QString& configPath) {
        rfkt::initialize(instance()->m_db, configPath.toStdString());
    }

private:
    explicit VariationDatabase(QObject* parent = nullptr) : QObject(parent) {}

    rfkt::flamedb m_db;
    QStringList m_variationNames;
};
