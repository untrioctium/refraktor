#pragma once

#include <cuda.h>
#include <type_traits>

namespace rfkt {

	using buffer_ptr = std::size_t;

	namespace detail {
		template<class Buffer, class Contained, class PtrType>
		class buffer_base {
		public:
			constexpr static std::size_t element_size = sizeof(Contained);
			using ptr_type = PtrType;

			buffer_base() : ptr_(0), size_(0), count_(0) {}

			buffer_base(std::size_t count) : 
				size_(element_size * count),
				count_(count) 
			{
				//assert(count > 0);

				ptr_ = static_cast<Buffer*>(this)->allocate();

				//SPDLOG_DEBUG("[{}:{}:{}] Created buffer at {} of type '{}' with {} '{}' elements", filename_from_path(loc.file_name()), loc.function_name(), loc.line(), (void*)ptr_, demangle_buffer(typeid(Buffer).name()), count, demangle_type(typeid(Contained).name()));
			}

			buffer_base(PtrType ptr, std::size_t count): 
				size_(element_size * count),
				count_(count) {
				//assert(count > 0);

				ptr_ = ptr;

				//SPDLOG_DEBUG("[{}:{}:{}] Adopted buffer at {} of type '{}' with {} '{}' elements", filename_from_path(loc.file_name()), loc.function_name(), loc.line(), (void*)ptr_, demangle_buffer(typeid(Buffer).name()), count, demangle_type(typeid(Contained).name()));

			}

			~buffer_base() {
				if (ptr_)
				{
					static_cast<Buffer*>(this)->deallocate();
					//SPDLOG_DEBUG("Deleted buffer {}", (void*)ptr_);
				}
					
			}

			buffer_base(const buffer_base&) = delete;
			buffer_base& operator=(const Buffer) = delete;

			auto ptr() const { return ptr_; }
			auto size() const { return size_; }

			auto count() const { return count_; }

			operator bool() { return ptr_ != 0; }

			buffer_base& operator=(buffer_base&& o) {
				std::swap(ptr_, o.ptr_);
				std::swap(count_, o.count_);
				std::swap(size_, o.size_);
				return *this;
			}

			buffer_base(buffer_base&& o): ptr_(0), size_(0), count_(0) {
				std::swap(ptr_, o.ptr_);
				std::swap(count_, o.count_);
				std::swap(size_, o.size_);
			}
		private:

			/*static std::string demangle_buffer(const std::string& id) {
				if (!id.starts_with("class") && !id.starts_with("struct")) return id;

				auto start_pos = id.find_first_of(" ") + 1;
				auto end_pos = id.find_first_of("<");

				return id.substr(start_pos, end_pos - start_pos);

			}

			static std::string demangle_type(const std::string& t) {
				if (!t.starts_with("class") && !t.starts_with("struct")) return t;

				return t.substr(t.find_first_of(" ") + 1);
			}

			static std::string filename_from_path(const std::string& path) {
				auto start = path.find_last_of("\\/");

				if (start == std::string::npos) return path;
				else return path.substr(start + 1);
			}*/

			PtrType ptr_ = 0;
			std::size_t count_;
			std::size_t size_;
		};
	}

	class cuda_buffer_impl {
	public:
		using stream_type = CUstream;
		using ptr_type = CUdeviceptr;
		using default_type = unsigned char;

		static auto allocate(std::size_t size) {
			ptr_type new_ptr;
			cuMemAlloc(&new_ptr, size);
			return new_ptr;
		}

		static void deallocate(ptr_type ptr) {
			cuMemFree(ptr);
		}

		template<typename Contained>
		static void clear(ptr_type ptr, std::size_t size, Contained value) {
			constexpr auto element_size = sizeof(Contained);

			if constexpr (element_size == 1) {
				cuMemsetD8(ptr, *reinterpret_cast<unsigned char*>(&value), size);
			}
			else if constexpr (element_size == 2) {
				cuMemsetD16(ptr, *reinterpret_cast<unsigned short*>(&value), size / 2);
			}
			else if constexpr (element_size == 4) {
				cuMemsetD32(ptr, *reinterpret_cast<unsigned int*>(&value), size / 4);
			}
			else
			{
				// Memset only supports 1, 2, and 4 byte clear values
				cuMemsetD8(ptr, 0, size);
			}
		}

		template<typename Contained>
		static void to_host(Contained* host, ptr_type device, std::size_t size) {
			cuMemcpyDtoH(host, device, size);
		}

		template<typename Contained>
		static void to_host_async(Contained* host, ptr_type device, std::size_t size, stream_type stream) {
			cuMemcpyDtoHAsync(host, device, size, stream);
		}

		template<typename Contained>
		static void to_device(ptr_type device, Contained* host, std::size_t size) {
			cuMemcpyHtoD(device, host, size);
		}

		template<typename Contained>
		static void to_device_async(ptr_type device, Contained* host, std::size_t size, stream_type stream) {
			cuMemcpyHtoDAsync(device, host, size, stream);
		}
	};

	template<class Impl, class Contained = typename Impl::default_type>
	class gpu_buffer : public detail::buffer_base<gpu_buffer<Impl, Contained>, Contained, typename Impl::ptr_type> {
	private:
		using parent = detail::buffer_base<gpu_buffer<Impl, Contained>, Contained, typename Impl::ptr_type>;
	public:
		using parent::parent;

		void to_host(Contained* host) {
			Impl::to_host(host, this->ptr(), this->size());
		}

		void to_host_async(Contained* host, typename Impl::stream_type stream) {
			Impl::to_host_async(host, this->ptr(), this->size(), stream);
		}

		void clear() {
			Impl::clear(this->ptr(), this->size(), Contained{});
		}

		void clear_value(Contained v) {
			Impl::clear(this->ptr(), this->size(), v);
		}

	private:
		friend class parent;

		auto allocate() {
			return Impl::allocate(this->size());
		}

		void deallocate() {
			Impl::deallocate(this->ptr());
		}
	};

	template<typename Contained = unsigned char>
	using cuda_buffer = gpu_buffer<cuda_buffer_impl, Contained>;

	namespace cuda {
		template<typename T>
		auto make_buffer_async(std::size_t size, CUstream stream) -> cuda_buffer<T> {
			CUdeviceptr buf{};
			cuMemAllocAsync(&buf, size * sizeof(T), stream);
			return cuda_buffer<T>{ buf, size };
		}
	}
}