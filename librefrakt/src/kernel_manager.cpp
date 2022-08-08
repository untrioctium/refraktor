#include <SQLiteCpp/SQLiteCpp.h>

#include <librefrakt/kernel_manager.h>
#include <librefrakt/util.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/string.h>
#include <librefrakt/util/zlib.h>
#include <librefrakt/util/hash.h>

#include <spdlog/spdlog.h>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      SPDLOG_ERROR("\nNVRTC error: {} failed with error: {}", #x, nvrtcGetErrorString(result));           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const std::string& compute_version() {
	static const std::string version = []() {
		CUdevice dev;
		cuCtxGetDevice(&dev);

		int major = 0, minor = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

		return fmt::format("sm_{}{}", major, minor);
	}();

	return version;
}

class rfkt::kernel_manager::kernel_cache {
public:

	struct row {
		std::string id;
		long long create_ts;
		double compile_ms;
		std::vector<std::string> dependencies;
		std::map<std::string, std::string> functions;
		std::map<std::string, std::string> globals;
		std::vector<char> cubin;
		std::string user_hash;
		unsigned int uncompressed_size;
	};

	kernel_cache() : db("kernel.sqlite", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE) {
		db.exec(
			"CREATE TABLE IF NOT EXISTS cache("
			"id TEXT PRIMARY KEY,"
			"opts_hash TEXT,"
			"create_ts INTEGER,"
			"dependencies TEXT,"
			"functions TEXT,"
			"globals TEXT,"
			"compile_ms REAL,"
			"uncompressed_size INTEGER,"
			"cubin BLOB);"
		);
	}
	~kernel_cache() {}

	bool is_available(const std::string& id, const rfkt::hash_t& opts_hash) {
		//std::lock_guard<std::mutex> lock(mtx);

		auto query = SQLite::Statement(db, "SELECT dependencies,create_ts,opts_hash FROM cache WHERE id = ?");
		query.bind(1, id);
		if (!query.executeStep()) return false;

		auto depends = str_util::split(query.getColumn(0), ';');
		long long create_ts = query.getColumn(1);
		std::string stored_hash = query.getColumn(2);

		if (stored_hash != opts_hash.str64()) {
			SPDLOG_INFO("Invalidating cache for {}: options hash mismatch", id);
			remove(id);
			return false;
		}

		for (auto& dep : depends) {
			if (fs::last_modified(dep) > create_ts) {
				SPDLOG_INFO("Invalidating cache for {}: {} has been modified", id, dep);
				remove(id);
				return false;
			}
		}

		return true;
	}

	row get(const std::string& id) {
		//std::lock_guard<std::mutex> lock(mtx);

		auto query = SQLite::Statement(db, "SELECT id, create_ts, dependencies, functions, globals, cubin FROM cache WHERE id = ?");
		query.bind(1, id);
		if (!query.executeStep()) exit(1);

		auto get_name_map = [](const std::string& names) {
			auto ret = std::map<std::string, std::string>{};
			for (auto& p : str_util::split(names, ';')) {
				auto row = str_util::split(p, '=');
				ret[row[0]] = row[1];
			}
			return ret;
		};

		row ret{};
		ret.id = query.getColumn(0).getString();
		ret.create_ts = query.getColumn(1);
		ret.dependencies = str_util::split(query.getColumn(2));
		ret.functions = get_name_map(query.getColumn(3));
		ret.globals = get_name_map(query.getColumn(4));


		auto blob = query.getColumn(5);
		ret.cubin = zlib::uncompress(blob.getBlob(), blob.getBytes());
		return ret;
	}

	void insert(const row& r, const rfkt::hash_t& hash) {
		//std::lock_guard<std::mutex> lock(mtx);

		auto query = SQLite::Statement(db, "INSERT OR IGNORE INTO cache VALUES(?,?,?,?,?,?,?,?,?)");

		query.bind(1, r.id);
		query.bind(2, hash.str64());
		query.bind(3, fs::now());
		query.bind(4, str_util::join(r.dependencies, ';'));
		query.bind(5, str_util::join(r.functions, '=', ';'));
		query.bind(6, str_util::join(r.globals, '=', ';'));
		query.bind(7, r.compile_ms);
		query.bind(8, (unsigned int)r.cubin.size());
		auto comp = zlib::compress(r.cubin);
		query.bind(9, comp.data(), comp.size());
		query.exec();
	}

private:

	void remove(const std::string& id) {
		auto query = SQLite::Statement(db, "DELETE FROM cache WHERE id = ?");
		query.bind(1, id);
		query.exec();
	}

	SQLite::Database db;
	//std::mutex mtx;
};

rfkt::kernel_manager::kernel_manager() : k_cache(new kernel_cache()){}

rfkt::kernel_manager::~kernel_manager() = default;

auto rfkt::kernel_manager::load_cache(const compile_opts & opts) -> cuda_module
{
	auto handle = cuda_module{};
	auto row = k_cache->get(opts.name);
	cuModuleLoadData(&handle.module_handle, row.cubin.data());

	for (auto& [name, mangled] : row.functions) {
		CUDA_SAFE_CALL(cuModuleGetFunction(&handle.functions_[name], handle.module_handle, mangled.c_str()));
	}

	for (auto& [name, mangled] : row.globals) {
		CUDA_SAFE_CALL(cuModuleGetGlobal(&handle.globals_[name].second, &handle.globals_[name].first, handle.module_handle, mangled.c_str()));
	}

	return handle;
}

auto find_global_includes(const std::string& src) {
	return rfkt::str_util::find_unique(src, R"regex(#include\s+<([^>]+)>)regex");
}

auto find_provided_includes(const std::string& src) {
	return rfkt::str_util::find_unique(src, R"regex(#include\s+"([^"]+)")regex");
}

auto rfkt::kernel_manager::compile_file(const std::string& filename, const compile_opts& opts) -> std::pair<compile_result, cuda_module>
{
	compile_opts new_opts = opts;
	new_opts.depend(filename);
	return compile_string(fs::read_string(filename), new_opts);
}

auto rfkt::kernel_manager::compile_string(const std::string& source, const compile_opts& opts) -> std::pair<compile_result, cuda_module>
{	
	auto opt_hash = opts.hash();
	if (k_cache->is_available(opts.name, opt_hash)) return { compile_result{true, ""}, std::move(load_cache(opts)) };

	kernel_cache::row cache;

	auto start = std::chrono::high_resolution_clock::now();
	auto [cr, ptx] = ptx_from_string(source, opts);
	auto end = std::chrono::high_resolution_clock::now();

	if (!cr.success) {
		return { cr,  cuda_module{} };
	}
	cache.compile_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1'000'000.0f;

	CUmodule mod;
	CUlinkState link_state;
	cache.id = opts.name;
	for (auto& dep : opts.dependencies) cache.dependencies.push_back(dep);

	// TODO: move this out, this prevents this API from being entirely reusable
	for (auto& dep : find_global_includes(source)) 
		if(dep.starts_with("refrakt")) 
			cache.dependencies.push_back("assets/kernels/include/" + dep);

	char info_log[1024 * 4];
	char err_log[1024 * 4];

	CUjit_option option_names[] = {
		CU_JIT_INFO_LOG_BUFFER,
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_ERROR_LOG_BUFFER,
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
	};

	void* options[] = { info_log, reinterpret_cast<void*>(uintptr_t(sizeof(info_log))), err_log, reinterpret_cast<void*>(uintptr_t(sizeof(err_log))) };

	CUDA_SAFE_CALL(cuLinkCreate(4,option_names,options, &link_state));
	CUDA_SAFE_CALL(cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void*) ptx.get_ptx(), ptx.size(), opts.name.c_str(), 0, 0, 0));

	/*for (auto link : opts.links) {
		CUDA_SAFE_CALL(cuLinkAddData(link_state, CU_JIT_INPUT_PTX, (void*)link->get_ptx(), link->size(), 0, 0, 0, 0));
	}*/

	std::size_t cubin_size;
	void* cubin;
	if (cuLinkComplete(link_state, &cubin, &cubin_size) != CUDA_SUCCESS) {
		//std::cout << info_log << std::endl;
		//std::cout << err_log << std::endl;
		exit(1);
	}

	cache.cubin.resize(cubin_size);
	memcpy(cache.cubin.data(), cubin, cubin_size);

	CUDA_SAFE_CALL(cuModuleLoadData(&mod, cubin));
	CUDA_SAFE_CALL(cuLinkDestroy(link_state));

	auto handle = cuda_module{};
	handle.module_handle = mod;
	for (auto& kernel : opts.functions) {
		auto mangled = ptx.get_name(kernel);
		cache.functions[kernel] = mangled;
		CUDA_SAFE_CALL(cuModuleGetFunction(&handle.functions_[kernel], mod, mangled));
	}
	for (auto& global : opts.globals) {
		auto mangled = ptx.get_name(global);
		cache.globals[global] = mangled;
		CUDA_SAFE_CALL(cuModuleGetGlobal(&handle.globals_[global].second, &handle.globals_[global].first, mod, mangled));
	}

	k_cache->insert(cache, opt_hash);

	cr.success = true;
	return { cr, std::move(handle) };
}

auto rfkt::kernel_manager::ptx_from_file(const std::string& filename, const compile_opts& opts) -> std::pair<compile_result, ptx>
{
	return ptx_from_string(fs::read_string(filename), opts);
}

auto rfkt::kernel_manager::ptx_from_string(const std::string& source, const compile_opts& opts) -> std::pair<compile_result, ptx>
{
	auto provided_includes = find_provided_includes(source);

	std::vector<const char*> header_names;
	std::vector<const char*> header_contents;
	// headers
	{
		for (const auto& [name, src] : opts.headers) {
			header_names.push_back(name.c_str());
			header_contents.push_back(src.c_str());
			if (provided_includes.contains(name)) {
				provided_includes.erase(name);
			}
		}
	}

	if (provided_includes.size() > 0) {
		return { compile_result{false, fmt::format("Cannot find ({})", rfkt::str_util::join(provided_includes, ','))}, ptx{} };
	}

	nvrtcProgram prog;
	NVRTC_SAFE_CALL(nvrtcCreateProgram(
		&prog, source.c_str(), opts.name.c_str(),
		header_names.size(), header_contents.data(), header_names.data()
	));

	// names
	{
		for (const auto& kernel : opts.functions) {
			NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel.c_str()));
		}

		for (const auto& global : opts.globals) {
			NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, global.c_str()));
		}
	}

	// prepare all flags for the compiler
	std::vector<const char*> compile_options;
	auto arch = fmt::format("--gpu-architecture={}", compute_version());
	compile_options.push_back(arch.c_str());
	compile_options.push_back("--std=c++17");
	compile_options.push_back("--include-path=assets/extern/include/");
	compile_options.push_back("--include-path=assets/kernels/include/");
	compile_options.push_back("-default-device");

	if (opts.flags.count(compile_flag::use_fast_math)) compile_options.push_back("--use_fast_math");
	if (opts.flags.count(compile_flag::line_info)) compile_options.push_back("-lineinfo");
	if (opts.flags.count(compile_flag::relocatable_device_code)) compile_options.push_back("-dc");
	if (opts.flags.count(compile_flag::device_debug)) compile_options.push_back("-G");
	if (opts.flags.count(compile_flag::extensible_whole_program)) compile_options.push_back("--extensible-whole-program");
	if (opts.flags.count(compile_flag::extra_vectorization)) compile_options.push_back("--extra-device-vectorization");

	std::vector<std::string> define_strings;
	for (const auto& [name, value] : opts.defines) {
		define_strings.push_back(
			(value.size() > 0)
			? fmt::format("--define-macro={}={}", name, value)
			: fmt::format("--define-macro={}", name)
		);
		compile_options.push_back(define_strings.back().c_str());
	}

	auto [c_time, c_result] = 
		time_it([&]() 
		{ 
				SPDLOG_INFO("Starting compile for {}", opts.name);
				return nvrtcCompileProgram(prog, compile_options.size(), compile_options.data()); 
		});

	size_t logSize;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	char* log = new char[logSize];
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

	compile_result cr;
	cr.log = log;
	delete[] log;

	if (c_result != NVRTC_SUCCESS) {
		cr.success = false;
		NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
		return { cr, nullptr };
	}
	cr.success = true;
	SPDLOG_INFO("Compiled '{}' in {}ms.", opts.name, c_time);

	return { cr, ptx{prog} };
}

bool rfkt::kernel_manager::is_cached(const compile_opts& opts) const
{
	return k_cache->is_available(opts.name, opts.hash());
}

rfkt::ptx::ptx(nvrtcProgram prog_state) : prog(prog_state) {
	nvrtcGetPTXSize(prog, &ptx_size);
	ptx_.reserve(ptx_size);
	nvrtcGetPTX(prog, ptx_.data());
}

rfkt::ptx::~ptx() {
	if (prog != nullptr) nvrtcDestroyProgram(&prog);
}

const char* rfkt::ptx::get_name(const std::string& name) const {
	if (prog == nullptr) return nullptr;

	const char* lname;
	nvrtcGetLoweredName(prog, name.c_str(), &lname);
	return lname;
}

CUresult rfkt::kernel_t::launch_impl(CUfunction f, dim3 grid, dim3 block, CUstream stream, bool cooperative, void** data)
{
	if (cooperative) {
		return cuLaunchCooperativeKernel(
			f,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0, stream,
			data
		);
	}
	else {
		return cuLaunchKernel(
			f,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0, stream,
			data, 0
		);
	}
}

void rfkt::compile_opts::add_to_hash(hash::state_t& hs) const {
	hs.update("device");
	hs.update(cuda::context::current().device().name());
	hs.update("compile_opts");
	for (auto& flag : flags) hs.update(&flag, sizeof(flag));

	hs.update("defines");
	for (auto& [n, v] : defines) {
		hs.update(fmt::format("{}={}", n, v));
	}

	hs.update("functions");
	for (auto& f : functions) hs.update(f);

	hs.update("globals");
	for (auto& g : globals) hs.update(g);

	hs.update("headers");
	for (auto& [n, v] : headers) {
		hs.update(n);
		hs.update(v);
	}

	hs.update("links");
	for (auto& link : links) {
		hs.update(link->get_ptx(), link->size());
	}

	hs.update(compute_version());
}
