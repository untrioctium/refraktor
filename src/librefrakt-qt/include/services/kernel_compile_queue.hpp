#pragma once

#include <QObject>
#include <QThreadPool>
#include <QFuture>
#include <QtConcurrent/QtConcurrent>
#include <QQmlEngine>

#include <librefrakt/flame_compiler.hpp>

#include <qqmlintegration.h>

class KernelCompileQueue : public QObject {
    Q_OBJECT

public:
    explicit KernelCompileQueue(QObject* parent = nullptr);

    static KernelCompileQueue* instance();

    QFuture<ezrtc::compiler::result> requestCompile(ezrtc::spec&& spec);

    QFuture<rfkt::flame_compiler::result> requestCompile(
        const rfkt::flamedb& fdb,
        const rfkt::flame& f,
        rfkt::precision prec);

    static ezrtc::compiler* kernelManagerInstance();

private:
    ezrtc::compiler m_kernelManager;
    QThreadPool m_compilePool;
    rfkt::flame_compiler m_flameCompiler;
};
