#include <dylib.hpp>
#include <optional>
#include <string_view>
#include <cstdio>

#define ROCCU_IMPL
#include "roccu.h"

struct roccu_impl {
	dylib driver_lib;
	dylib rtc_lib;
	roccu_api api;
};

static inline std::optional<roccu_impl> api;

void load_symbols() {
	if (!api) return;

	for(auto& [name, traits] : ru_map) {
		auto& lib = traits.source == RU_DRIVER? api->driver_lib : api->rtc_lib;
		try {
			auto sym = lib.get_function<void*>(api->api == ROCCU_API_CUDA ? traits.cuda_name : traits.rocm_name);
			if (sym) {
				*traits.func = sym;
			}
		} catch(const dylib::symbol_error& se) {
			printf("Failed to load symbol %s: %s\n", traits.cuda_name, se.what());
		}
	}
}

roccu_api roccuInit() {

	if(api) return api->api;

	auto driver_lib = []() -> std::optional<roccu_impl> {
		try { return { roccu_impl{dylib{ "nvcuda" }, dylib{"nvrtc64_120_0"}, ROCCU_API_CUDA } }; }
		catch (...) {}
		try { return { roccu_impl{dylib{ "libcuda"}, dylib{"libnvrtc"}, ROCCU_API_CUDA } }; }
		catch (...) {}
		return {};
	}();

	if(driver_lib) {
		api = std::move(driver_lib);

		load_symbols();

		return api->api;
	}

	return ROCCU_API_NONE;
}

roccu_api roccuGetApi()
{
	return api? api->api : ROCCU_API_NONE;
}
