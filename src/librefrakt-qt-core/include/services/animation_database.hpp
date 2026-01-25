#pragma once

#include <QObject>
#include <QQmlEngine>

#include <librefrakt/anima.hpp>

#include <qqmlintegration.h>

class AnimationDatabase : public QObject {
    Q_OBJECT
    QML_SINGLETON

public:
    rfkt::function_table& table() { return m_table; }

    static AnimationDatabase* instance() {
        static AnimationDatabase* instance = nullptr;
        if (!instance) {
            instance = new AnimationDatabase();
        }
        return instance;
    }

private:
    explicit AnimationDatabase(QObject* parent = nullptr) : QObject(parent) {}

    rfkt::function_table m_table;
};
