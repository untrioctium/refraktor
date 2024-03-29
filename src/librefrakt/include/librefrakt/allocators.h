#include <cstdint>
#include <concepts>
#include <roccu_cpp_types.h>
#include <atomic>

namespace rfkt {

    template<typename T>
    concept AllocatorTraits = requires(typename T::PtrType ptr, std::size_t sz) {
        { T::alloc(sz) } -> std::same_as<typename T::PtrType>;
        { T::free(ptr) } -> std::same_as<void>;
    };

    struct device_allocator_traits {
        using PtrType = RUdeviceptr;

        static PtrType alloc(std::size_t sz) {
            PtrType ptr;
            ruMemAlloc(&ptr, sz);
            return ptr;
        }

        static void free(PtrType ptr) {
            ruMemFree(ptr);
        }

        template<typename StoredT>
        static auto to_span(PtrType ptr, std::size_t sz) {
            return roccu::gpu_span<StoredT>{ ptr, sz };
        }
    };

    struct pinned_allocator_traits {
        using PtrType = std::byte*;

        static PtrType alloc(std::size_t sz) {
            std::byte* memory = nullptr;
            auto ret = ruMemAllocHost((void**)&memory, sz);
            return memory;
        }

        static void free(PtrType ptr) {
            ruMemFreeHost(ptr);
        }

        template<typename StoredT>
        static auto to_span(PtrType ptr, std::size_t sz) {
            return std::span<StoredT>{ reinterpret_cast<StoredT*>(ptr), sz};
        }
    };

    template<AllocatorTraits Traits>
    class ring_allocator {
    public:

        ring_allocator(std::size_t size) : size(size), memory(Traits::alloc(size)) {}

        ~ring_allocator() {
            Traits::free(memory);
        }

        template<typename T>
        auto reserve(std::size_t amount) {
            if (amount * sizeof(T) > size)
                throw std::bad_alloc{};

            return Traits::template to_span<T>(reserve(amount * sizeof(T)), amount);
        }

        ring_allocator(const ring_allocator&) = delete;
        ring_allocator& operator=(const ring_allocator&) = delete;

        ring_allocator(ring_allocator&&) = default;
        ring_allocator& operator=(ring_allocator&&) = default;

    private:

        class atomic_wrap_counter {
        public:
            explicit atomic_wrap_counter() noexcept : offset(new std::atomic_size_t{ 0 }) {}

            std::size_t increment(std::size_t amount, std::size_t max) noexcept {
                auto old_offset = offset->load();

                do {
                    auto new_offset = (old_offset + amount >= max) ? amount : old_offset + amount;

                    if (offset->compare_exchange_strong(old_offset, new_offset)) {
                        return new_offset - amount;
                    }
                } while (true);
            }

        private:
            std::unique_ptr<std::atomic_size_t> offset = nullptr;
        };

        typename Traits::PtrType reserve(std::size_t amount) {
            return memory + counter.increment(amount + (16 - amount % 16), size);
        }

        std::size_t size;
        typename Traits::PtrType memory;
        atomic_wrap_counter counter;
    };

    using pinned_ring_allocator = ring_allocator<pinned_allocator_traits>;
    using device_ring_allocator = ring_allocator<device_allocator_traits>;

}