#include "kernel_compile_queue.hpp"

#include <librefrakt/util/filesystem.hpp>

#include <roccu.hpp>

KernelCompileQueue::KernelCompileQueue(QObject* parent)
    : QObject(parent)
    , m_kernelManager(ezrtc::compiler(
          std::make_shared<ezrtc::cache_adaptors::zlib>(
              std::make_shared<ezrtc::cache_adaptors::guarded>(
                  std::make_shared<ezrtc::sqlite_cache>(
                      (rfkt::fs::user_local_directory() / "kernel.sqlite3").string().c_str())))))
    , m_flameCompiler(&m_kernelManager)
{
    m_compilePool.setMaxThreadCount(1);
    m_compilePool.setExpiryTimeout(-1);

    auto ctx = roccu::context_view::current();
    m_compilePool.start([ctx]() {
        ctx.make_current();
    });
}

KernelCompileQueue* KernelCompileQueue::instance() {
    static KernelCompileQueue* instance = nullptr;
    if (!instance) {
        instance = new KernelCompileQueue();
    }
    return instance;
}

QFuture<ezrtc::compiler::result> KernelCompileQueue::requestCompile(ezrtc::spec&& spec) {
    return QtConcurrent::run(&m_compilePool, [this, spec = std::move(spec)]() mutable {
        return m_kernelManager.compile(std::move(spec));
    });
}

QFuture<rfkt::flame_compiler::result> KernelCompileQueue::requestCompile(
    const rfkt::flamedb& fdb,
    const rfkt::flame& f,
    rfkt::precision prec)
{
    auto thunk = m_flameCompiler.prepare_flame_kernel(fdb, prec, f, {}, 2);
    return QtConcurrent::run(&m_compilePool, [thunk = std::move(thunk)]() mutable {
        return thunk();
    });
}

ezrtc::compiler* KernelCompileQueue::kernelManagerInstance() {
    return &instance()->m_kernelManager;
}
