#ifndef CPUJIT_HPP
#define CPUJIT_HPP

#include "cpujit.h"

#include <stdexcept>
#include <utility>

#include <librefrakt/flame_types.hpp>

namespace cpujit {

class module {
public:
    explicit module(const char* llvm_ir) {
        const char* error = nullptr;
        handle_ = cpujit_compile(llvm_ir, &error);
        if (!handle_) {
            std::string msg = error ? error : "unknown cpujit error";
            cpujit_error_free(error);
            throw std::runtime_error(std::move(msg));
        }
    }

    ~module() {
        if (handle_) cpujit_destroy(handle_);
    }

    module(const module&) = delete;
    module& operator=(const module&) = delete;

    module(module&& o) noexcept : handle_(std::exchange(o.handle_, nullptr)) {}

    module& operator=(module&& o) noexcept {
        std::swap(handle_, o.handle_);
        return *this;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

    template<typename F>
    F* lookup(const char* name) const {
        return reinterpret_cast<F*>(cpujit_lookup(handle_, name));
    }

    const char* assembly() const noexcept {
        return cpujit_asm(handle_);
    }

    static const char* preamble() noexcept {
        return cpujit_preamble();
    }

private:
    cpujit_module_t handle_ = nullptr;
};

} // namespace cpujit

#endif /* CPUJIT_HPP */
