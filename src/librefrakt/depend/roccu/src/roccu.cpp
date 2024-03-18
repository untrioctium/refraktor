#include <dylib.hpp>
#include <optional>
#include <string_view>
#include <cstdio>
#include <array>
#include <atomic>
#include <mutex>

#include <stacktrace>

#define ROCCU_IMPL
#include "roccu.h"

struct roccu_impl {
	roccu_api api;
	dylib driver_lib;
	dylib rtc_lib;
};

static inline std::optional<roccu_impl> api;
constinit inline static std::atomic_size_t mem_alloc_size = 0;

inline static std::unordered_map<RUdeviceptr, std::pair<size_t, std::stacktrace>> mem_alloc_map = {};
inline static std::mutex mem_alloc_mutex = {};

bool load_symbols() {
	if (!api) return false;

	for(auto& [name, traits] : ru_map) {
		const auto& lib = traits.source == RU_DRIVER? api->driver_lib : api->rtc_lib;
		try {
			*traits.func = lib.get_function<void*>(api->api == ROCCU_API_CUDA ? traits.cuda_name : traits.rocm_name);
		} catch(const dylib::symbol_error& se) {
			printf("Failed to load symbol %s: %s\n", traits.cuda_name, se.what());
			return false;
		}
	}

	return true;
}

struct roccu_load_info {
	roccu_api api;
	const char* driver;
	const char* rtc;
};

constexpr static auto roccu_load_info_list = std::array{
	roccu_load_info{ROCCU_API_CUDA, "nvcuda", "nvrtc64_120_0"}, // CUDA Windows
	roccu_load_info{ROCCU_API_CUDA, "libcuda", "libnvrtc"} // CUDA Linux
};

void hook_allocation() {
	const static auto old_mem_alloc = ruMemAlloc;
	ruMemAlloc = [](RUdeviceptr* ptr, size_t size) {
		std::scoped_lock lock(mem_alloc_mutex);
		mem_alloc_size += size;
		auto ret = old_mem_alloc(ptr, size);
		mem_alloc_map[*ptr] = { size, std::stacktrace::current() };
		return ret;
		};

	const static auto old_mem_free = ruMemFree;
	ruMemFree = [](RUdeviceptr ptr) {
		std::scoped_lock lock(mem_alloc_mutex);
		mem_alloc_size -= mem_alloc_map[ptr].first;
		mem_alloc_map.erase(ptr);
		return old_mem_free(ptr);
	};

	const static auto old_mem_alloc_async = ruMemAllocAsync;
	ruMemAllocAsync = [](RUdeviceptr* ptr, size_t size, RUstream stream) {
		std::scoped_lock lock(mem_alloc_mutex);

		auto ret = old_mem_alloc_async(ptr, size, stream);
		mem_alloc_map[*ptr] = { size, std::stacktrace::current() };

		ruLaunchHostFunc(stream, [](void* ptr) {
			mem_alloc_size += mem_alloc_map[reinterpret_cast<RUdeviceptr>(ptr)].first;
		}, reinterpret_cast<void*>(*ptr));

		return ret;
	};

	const static auto old_mem_free_async = ruMemFreeAsync;
	ruMemFreeAsync = [](RUdeviceptr ptr, RUstream stream) {
		auto ret = old_mem_free_async(ptr, stream);

		ruLaunchHostFunc(stream, [](void* ptr) {
			std::scoped_lock lock(mem_alloc_mutex);
			RUdeviceptr dptr = reinterpret_cast<RUdeviceptr>(ptr);
			mem_alloc_size -= mem_alloc_map[dptr].first;
			mem_alloc_map.erase(dptr);

		}, reinterpret_cast<void*>(ptr));

		return ret;
	};
}

roccu_api roccuInit() {

	if(api) return api->api;

	api = []() -> std::optional<roccu_impl> {
		for(const auto& info : roccu_load_info_list) {
			try{ return roccu_impl{info.api, dylib{info.driver}, dylib{info.rtc}}; }
			catch (const dylib::load_error&) { continue; }
		}
		return std::nullopt;
	}();

	if(api) {
		if (load_symbols()) {
			hook_allocation();
			return api->api;
		}
		else {
			api.reset();
		}
	}

	return ROCCU_API_NONE;
}

roccu_api roccuGetApi()
{
	return api? api->api : ROCCU_API_NONE;
}

size_t roccuGetMemoryUsage() {
	return mem_alloc_size;
}

void roccuPrintAllocations() {
	std::scoped_lock lock(mem_alloc_mutex);
	for(const auto& [ptr, alloc] : mem_alloc_map) {
		printf("Allocation at %u of size %zu (%.1f MB)\n", ptr, alloc.first, alloc.first / (1024.0 * 1024.0));
		for(int i = 1; i < alloc.second.size(); i++) {
			auto& entry = alloc.second[i];
			printf("%s %d %s\n", entry.source_file().c_str(), entry.source_line(), entry.description().c_str());
		}
	}
}