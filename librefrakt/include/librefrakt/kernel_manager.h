#pragma once

#include <map>
#include <set>
#include <memory>

#include <cuda.h>
#include <vector_types.h>
#include <nvrtc.h>
#include <fmt/format.h>

#include <librefrakt/util/cuda.h>
#include <librefrakt/traits/hashable.h>

namespace rfkt {

	class kernel_manager;
	class cuda_module;

	enum class compile_flag: char {
		use_fast_math,
		relocatable_device_code,
		device_debug,
		line_info,
		extra_vectorization,
		extensible_whole_program
	};

	class ptx {
	public:
		ptx(nvrtcProgram prog_state);
		~ptx();

		ptx(const ptx&) = delete;
		ptx& operator=(const ptx&) = delete;

		ptx(ptx&& o) noexcept {
			std::swap(ptx_, o.ptx_);
			std::swap(ptx_size, o.ptx_size);
			std::swap(prog, o.prog);
		}
		ptx& operator=(ptx&& o) noexcept {
			std::swap(ptx_, o.ptx_);
			std::swap(ptx_size, o.ptx_size);
			std::swap(prog, o.prog);
			return *this;
		}

		ptx() = default;

		const char* get_ptx() const { return (prog != nullptr)? ptx_.c_str(): nullptr; }
		const char* get_name(const std::string& name) const;

		std::size_t size() const { return (prog != nullptr)? ptx_size: 0;  }

	private:
		std::string ptx_ = {};
		std::size_t ptx_size = 0;

		nvrtcProgram prog = nullptr;
	};

	class kernel_t {
	public:
		friend class cuda_module;

		auto launch(dim3 grid, dim3 block, CUstream stream = 0, bool cooperative = false) const {
			return[f = this->f, grid, block, stream, cooperative](auto&&... args) {
				return launch_impl(f, grid, block, stream, cooperative, std::array<void*, sizeof...(args)>{&args...}.data());
			};
		}

		auto launch(std::uint32_t grid, std::uint32_t block, CUstream stream = 0, bool cooperative = false) const {
			return this->launch({ grid, 1, 1 }, { block, 1, 1 }, stream, cooperative);
		}

		int max_blocks_per_mp(int block_size) const {
			int num_blocks;
			cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, f, block_size, 0);
			return num_blocks;
		}

		int max_blocks_per_mp(const dim3& block_dim) const {
			return max_blocks_per_mp(block_dim.x * block_dim.y * block_dim.z);
		}

		std::pair<int, int> suggested_dims() const {
			std::pair<int, int> result;
			cuOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, 0, 0, 0);
			return result;
		}

		auto shared_bytes() const { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>(); }
		auto const_bytes() const { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>(); }
		auto local_bytes() const { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>(); }
		auto register_count() const { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS>(); }

		void set_smem_carveout(int carveout) const {
			cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout);
		}

	private:

		static CUresult launch_impl(CUfunction, dim3, dim3, CUstream, bool, void**);

		template<CUfunction_attribute a>
		auto attribute() const {
			int ret;
			cuFuncGetAttribute(&ret, a, f);
			return static_cast<std::size_t>(ret);
		}

		CUfunction f;
		kernel_t(CUfunction function) : f(function) {}
	};

	class cuda_module {
	public:
		friend class kernel_manager;

		auto operator()() const {
			return kernel();
		};
	
		auto operator()(std::string_view name) const {
			return kernel(name);
		}

		auto kernel(std::string_view name) const -> kernel_t {
			return { functions_.find(name)->second };
		}

		auto kernel() const -> kernel_t {
			return { functions_.begin()->second };
		}

		~cuda_module() {
			if(module_handle) cuModuleUnload(module_handle);
		}

		operator bool() const { return module_handle != nullptr; }

		cuda_module(const cuda_module&) = delete;
		cuda_module& operator=(const cuda_module&) = delete;

		cuda_module(cuda_module&& o) noexcept
		{
			std::swap(module_handle, o.module_handle);
			std::swap(functions_, o.functions_);
			std::swap(globals_, o.globals_);
		}

		cuda_module& operator=(cuda_module&& o) noexcept {
			std::swap(module_handle, o.module_handle);
			std::swap(functions_, o.functions_);
			std::swap(globals_, o.globals_);
			return *this;
		}

		cuda_module() = default;

	private:
		CUmodule module_handle = nullptr;
		std::map<std::string, CUfunction, std::less<>> functions_ = {};
		std::map<std::string, std::pair<std::size_t, CUdeviceptr>, std::less<>> globals_ = {};
	};

	class compile_opts: public traits::hashable<compile_opts> {
	public:
		friend class kernel_manager;

		compile_opts(std::string prog_name) : name(prog_name) {}
		
		auto& flag(compile_flag flag) { flags.insert(flag); return *this; }
		auto& define(std::string macro, std::string value = "") { defines[macro] = value; return *this; }

		template<typename Stringable>
		auto& define(std::string macro, const Stringable& value) { defines[macro] = fmt::format("{}", value); return *this; }

		auto& function(std::string name) { functions.insert(name); return *this; }
		auto& global(std::string name) { globals.insert(name); return *this; }
		auto& header(std::string name, std::string src) { headers[name] = src; return *this; }
		auto& depend(const std::string& path) { dependencies.insert(path); return *this; }
		auto& link(const ptx* lib) { links.push_back(lib); return *this; }

		std::string get_header(std::string name) { return headers[name]; }

		void add_to_hash(hash::state_t& hs) const;

	private:
		std::set<compile_flag> flags;
		std::map<std::string, std::string> defines;
		std::set<std::string> functions;
		std::set<std::string> globals;
		std::map<std::string, std::string> headers;
		std::set<std::string> dependencies;
		std::vector<const ptx*> links;

		std::string name;
	};

	class kernel_manager {
	public:
		struct compile_result {
			bool success;
			std::string log;
		};

		std::pair<compile_result, cuda_module> 
			compile_file(const std::string& filename, const compile_opts& opts);

		std::pair<compile_result, cuda_module>
			compile_string(const std::string& source, const compile_opts& opts);

		std::pair<compile_result, ptx>
			ptx_from_file(const std::string& filename, const compile_opts& opts);

		std::pair<compile_result, ptx>
			ptx_from_string(const std::string& filename, const compile_opts& opts);

		bool is_cached(const compile_opts& opts) const;

		kernel_manager();
		~kernel_manager();

	private:

		class kernel_cache;
		std::unique_ptr<kernel_cache> k_cache;

		cuda_module load_cache(const compile_opts& opts);
	};

}

