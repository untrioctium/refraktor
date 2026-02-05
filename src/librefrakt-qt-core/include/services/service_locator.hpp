#include <unordered_map>

#include <QObject>

class ServiceLocator : public QObject {
    Q_OBJECT

public:

    template<typename Service, typename Impl>
    void provide(Impl* impl) requires std::is_base_of_v<Service, Impl> {
        m_services[qMetaTypeId<Service>()] = impl;
    }

    template<typename T>
    T* get() {
        auto it = m_services.find(qMetaTypeId<T>());
        if (it == m_services.end()) {
            return nullptr;
        }
        return static_cast<T*>(it->second);
    }

    explicit ServiceLocator(QObject* parent = nullptr) : QObject(parent) {}

    static ServiceLocator* instance() {
        static ServiceLocator* s_instance = nullptr;
        if (!s_instance) {
            s_instance = new ServiceLocator();
        }
        return s_instance;
    }

private:

    std::unordered_map<int, QObject*> m_services;
};