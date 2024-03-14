#include <spdlog/spdlog.h>

#include <librefrakt/util/cuda.h>
#include <librefrakt/util/http.h>
#include <librefrakt/util/zlib.h>

std::optional<rfkt::fs::path> rfkt::cuda::check_and_download_cudart()
{
    const auto& base_path = rfkt::fs::user_local_directory();

    static constexpr auto etag_name = "etag.txt";
    const auto runtime_path = base_path / "cudart-main";

    bool has_include_dir = fs::exists(runtime_path);
    bool has_etag_file = fs::exists(base_path / etag_name);
    bool already_has_working = has_include_dir && has_etag_file;

    static constexpr auto cudart_url = "https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/archive/main/cudart-main.zip";
    const auto cudart_zip_path = base_path / "cudart.zip";

    auto headers = rfkt::http::head(cudart_url);

    if (!headers) {
        if (already_has_working) {
			return runtime_path;
        }
        else {
			return std::nullopt;
		}
    }

    if (fs::exists(base_path / "etag.txt") && has_include_dir) {
        if((*headers)["Etag"] == rfkt::fs::read_string(base_path / etag_name)) {
			return runtime_path;
		}
    }

    SPDLOG_INFO("Downloading CUDA runtime headers");
    SPDLOG_INFO(
        "The following command will download NVIDIA proprietary software. "
        "By using the software you agree to comply with the terms of the license agreement that accompanies the software. "
        "If you do not agree to the terms of the license agreement, do not use the software.");

    if (!rfkt::http::download(cudart_url, cudart_zip_path)) {
		SPDLOG_ERROR("Failed to download CUDA runtime headers");
		return std::nullopt;
	}

    SPDLOG_INFO("Extracting CUDA runtime headers");
    if (!rfkt::zlib::extract_zip(cudart_zip_path, base_path)) {
		SPDLOG_ERROR("Failed to extract CUDA runtime headers");
		return std::nullopt;
	}

	rfkt::fs::write(base_path / etag_name, (*headers)["Etag"], false);

	return runtime_path;

}

auto rfkt::cuda::init() -> context
{
    CUdevice dev;
    CUcontext ctx;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);

    auto devobj = rfkt::cuda::device_t{ dev };

    std::size_t max_persist_l2 = devobj.max_persist_l2_cache_size();
    CUDA_SAFE_CALL(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_persist_l2));

    return { ctx, dev };
}
