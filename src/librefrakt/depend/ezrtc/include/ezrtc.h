#ifndef EZRTC_HEADER_GUARD
#define EZRTC_HEADER_GUARD

#ifdef EZRTC_USE_ZLIB
#include <zlib.h>
#endif

#ifdef EZRTC_USE_SQLITE
#include <sqlite3.h>
#endif

#include <array>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <span>
#include <fstream>
#include <mutex> 
#include <filesystem>

#include <roccu.h>

#include <limits>

// public api
namespace ezrtc {

	class kernel {
	public:

		explicit kernel(RUfunction f) noexcept : f(f) {}

		auto launch(dim3 grid, dim3 block, RUstream stream = nullptr, bool cooperative = false) const noexcept {
			return[f = this->f, grid, block, stream, cooperative](auto... args) noexcept {
				auto packed_args = std::array<void*, sizeof...(args)>{ &args... };
				return launch_impl(f, grid, block, stream, cooperative, packed_args.data());
			};
		}

		auto launch(std::uint32_t grid, std::uint32_t block, RUstream stream = nullptr, bool cooperative = false) const noexcept {
			return this->launch({ grid, 1, 1 }, { block, 1, 1 }, stream, cooperative);
		}

		int max_blocks_per_mp(int block_size) const noexcept {
			int num_blocks;
			ruOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, f, block_size, 0);

			return (roccuGetApi() == ROCCU_API_CUDA)? num_blocks : num_blocks * 2;
		}

		int max_blocks_per_mp(const dim3& block_dim) const noexcept {
			return max_blocks_per_mp(block_dim.x * block_dim.y * block_dim.z);
		}

		std::pair<int, int> suggested_dims() const noexcept {
			std::pair<int, int> result;
			ruOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, nullptr, 0, 0);
			return result;
		}

		auto shared_bytes() const noexcept { return attribute<RU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>(); }
		auto const_bytes() const noexcept { return attribute<RU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>(); }
		auto local_bytes() const noexcept { return attribute<RU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>(); }
		auto register_count() const noexcept { return attribute<RU_FUNC_ATTRIBUTE_NUM_REGS>(); }

	private:

		template<RUfunction_attribute a>
		auto attribute() const noexcept {
			int ret;
			ruFuncGetAttribute(&ret, a, f);
			return static_cast<std::size_t>(ret);
		}


		static RUresult launch_impl(RUfunction f, dim3 grid, dim3 block, RUstream stream, bool cooperative, void** args) noexcept;
		RUfunction f;
	};

	class variable {
	public:

		variable(RUdeviceptr ptr, std::size_t size) noexcept : ptr_(ptr), size_(size) {}

		explicit(false) operator RUdeviceptr() const noexcept {
			return ptr_;
		}

		auto ptr() const noexcept {
			return ptr_;
		}

		auto size() const noexcept { return size_; }

	private:

		RUdeviceptr ptr_;
		std::size_t size_;
	};

	class compiler;

	class cuda_module {
	public:
		auto operator()() const {
			return kernel();
		};

		auto operator()(const std::string& name) const {
			return kernel(name);
		}

		auto operator[](const std::string& name) const {
			return variables.at(name);
		}

		auto kernel(const std::string& name) const -> ezrtc::kernel {
			return ezrtc::kernel{ kernels.at(name) };
		}

		auto kernel() const -> ezrtc::kernel {
			return ezrtc::kernel{ kernels.begin()->second };
		}

		~cuda_module() {
			if (handle) ruModuleUnload(handle);
		}

		explicit operator bool() const noexcept { return handle != nullptr; }

		cuda_module(const cuda_module&) = delete;
		cuda_module& operator=(const cuda_module&) = delete;

		cuda_module(cuda_module&& o) noexcept {
			(*this) = std::move(o);
		}

		cuda_module& operator=(cuda_module&& o) noexcept {
			std::swap(handle, o.handle);
			std::swap(kernels, o.kernels);
			std::swap(variables, o.variables);
			return *this;
		}

		cuda_module() = default;

	private:
		static std::optional<cuda_module> from_cubin(std::span<const char> cubin);

		bool load_kernel(std::string_view pretty, std::string_view mangled);
		bool load_variable(std::string_view pretty, std::string_view mangled);

		friend class compiler;

		RUmodule handle = nullptr;
		std::unordered_map<std::string, RUfunction> kernels = {};
		std::unordered_map<std::string, variable> variables = {};
	};

	enum class compile_flag {
		relocatable_code,
		extensible_whole_program,
		device_debug,
		generate_line_info,
		enable_optimization,
		use_fast_math,
		extra_device_vectorization,
		disable_warnings,
		restricted_kernel_pointers,
		default_device,
		device_int128,
		warn_double_usage
	};

	class spec {      
	public:

		static spec source_string(std::string name, std::string source) {
			auto ret = spec{};
			ret.name = std::move(name);
			ret.source = std::move(source);

			return ret;
		}

		static spec source_file(std::string name, const std::filesystem::path& path) {
			namespace fs = std::filesystem;

			auto ret = spec{};
			ret.name = std::move(name);

			const auto size = fs::file_size(path);
			ret.source = std::string(size, '\0');
			std::ifstream(path, std::ios::in).read(ret.source.data(), size);

			ret.dependency(path);
			return ret;
		}

		spec& flag(compile_flag flag) { return emplace_and_chain(compile_flags, flag); }
		spec& define(std::string_view dname, std::string_view value = {});
		spec& define(std::string_view dname, std::integral auto value) {
			return define(dname, std::to_string(value));
		}
		spec& kernel(std::string_view kname) { return emplace_and_chain(kernels, kname); }
		spec& variable(std::string_view vname) { return emplace_and_chain(variables, vname, std::format("&{}", vname)); }
		spec& header(std::string_view hname, const std::string& hsource) {
			return emplace_and_chain(headers, hname, std::string_view{ hsource.c_str(), hsource.size() + 1 });
		}
		spec& dependency(const std::filesystem::path& path) { return emplace_and_chain(dependencies, std::filesystem::absolute(path).u8string()); }

	private:
		std::string_view signature() const;

		spec() noexcept = default;

		friend class compiler;

		template<typename Container, typename... Args>
		spec& emplace_and_chain(Container& container, Args&&... args) {
			cached_signature.reset();
			container.emplace(std::forward<Args>(args)...);
			return *this;
		}

		std::set<compile_flag> compile_flags;
		std::set<std::string, std::less<>> kernels;
		std::set<std::u8string, std::less<>> dependencies;

		std::map<std::string, std::string, std::less<>> defines;
		std::map<std::string, std::string, std::less<>> variables;
		std::map<std::string, std::string_view, std::less<>> headers;

		std::string name;
		std::string source;
		mutable std::optional<std::string> cached_signature;
	};

	class cache {
	public:
		using handle = std::shared_ptr<cache>;

		class row {
		public:
			using destructor_t = std::move_only_function<void(void) noexcept>;

			std::string_view signature{};
			std::span<const char> meta{};
			std::span<const char> data{};

			row(const row&) = delete;
			row& operator=(const row&) = delete;

			row(row&& o) noexcept {
				(*this) = std::move(o);
			}

			row& operator=(row&& o) noexcept {
				std::swap(destructor, o.destructor);
				std::swap(signature, o.signature);
				std::swap(meta, o.meta);
				std::swap(data, o.data);
				return *this;
			}

			explicit row(destructor_t destructor) : destructor(std::move(destructor)) {}

			virtual ~row() {
				if (destructor) destructor();
			}

		private:
			std::move_only_function<void(void)> destructor{};
		};

		virtual std::optional<row> get(std::string_view id) = 0;
		virtual void put(std::string_view id, row&& r) = 0;
		virtual void remove(std::string_view id) = 0;
	};

	class disk_cache {

	};

#ifdef EZRTC_USE_SQLITE
	class sqlite_cache : public cache {
	public:
		explicit sqlite_cache(std::string_view db_path) {
			sqlite3_open(db_path.data(), &db);
			sqlite3_exec(db, R"sql(
				CREATE TABLE IF NOT EXISTS cache(
					id TEXT PRIMARY KEY,
					signature TEXT,
					meta BLOB,
					data BLOB
				);
			)sql", nullptr, nullptr, nullptr);

			remove_stmt = statement{ db, "DELETE FROM cache WHERE id = ?" };
			put_stmt = statement{ db, "INSERT OR REPLACE INTO cache (id, signature, meta, data) VALUES (?,?,?,?)" };
		}

		~sqlite_cache() {
			sqlite3_close(db);
		}

		std::optional<row> get(std::string_view id) override {
			auto get_stmt = statement{ db, "SELECT signature, meta, data FROM cache WHERE id = ?" };

			get_stmt.bind_text(1, id);

			if (get_stmt.step() != SQLITE_ROW) {
				return std::nullopt;
			}

			auto sig = std::string_view{ (const char*) sqlite3_column_text(get_stmt, 0), static_cast<std::size_t>(sqlite3_column_bytes(get_stmt, 0)) };
			auto meta = std::span<const char>{ (const char*) sqlite3_column_blob(get_stmt, 1), static_cast<std::size_t>(sqlite3_column_bytes(get_stmt, 1)) };
			auto data = std::span<const char>{ (const char*) sqlite3_column_blob(get_stmt, 2), static_cast<std::size_t>(sqlite3_column_bytes(get_stmt, 2)) };

			auto row = cache::row{[stmt = std::move(get_stmt)]() noexcept {}};
			row.signature = sig;
			row.meta = std::move(meta);
			row.data = std::move(data);

			return row;
		}

		void put(std::string_view id, row&& r) override {
			put_stmt.reset();
			put_stmt.bind_text(1, id);
			put_stmt.bind_text(2, r.signature);
			put_stmt.bind_blob(3, r.meta);
			put_stmt.bind_blob(4, r.data);
			put_stmt.step();
		}

		void remove(std::string_view id) override {
			remove_stmt.reset();
			remove_stmt.bind_text(1, id);
			remove_stmt.step();
		}
	private:

		struct statement {
		public:
			statement() = default;

			statement(sqlite3* db, std::string_view sql) {
				sqlite3_prepare_v2(db, sql.data(), sql.size(), &stmt, nullptr);
			}

			statement(const statement&) = delete;
			statement& operator=(const statement&) = delete;

			statement(statement&& o) noexcept {
				*this = std::move(o);
			}

			statement& operator=(statement&& o) noexcept {
				std::swap(stmt, o.stmt);
				return *this;
			}

			~statement() {
				sqlite3_finalize(stmt);
			}

			operator sqlite3_stmt* () {
				return stmt;
			}

			int bind_text(int index, std::string_view text) {
				return sqlite3_bind_text(stmt, index, text.data(), text.size(), SQLITE_STATIC);
			}

			int bind_blob(int index, std::span<const char> data) {
				return sqlite3_bind_blob(stmt, index, data.data(), data.size(), SQLITE_STATIC);
			}

			int reset() {
				return sqlite3_reset(stmt);
			}

			int step() {
				return sqlite3_step(stmt);
			}

		private:
			sqlite3_stmt* stmt = nullptr;
		};

		sqlite3* db = nullptr;
		statement put_stmt;
		statement remove_stmt;
	};
#endif 

	namespace cache_adaptors {

		class adaptor_base : public cache {
		public:
			adaptor_base(cache::handle wrapped) : wrapped(std::move(wrapped)) {}

		protected:
			auto next() -> cache* {
				return wrapped.get();
			}

		private:
			cache::handle wrapped;
		};

#ifdef EZRTC_USE_ZLIB
		class zlib: public adaptor_base {
		public:
			using adaptor_base::adaptor_base;

			std::optional<row> get(std::string_view id) {
				auto row = next()->get(id);
				if (not row) { return std::nullopt; }
				if (row->data.size() < sizeof(size_type)) { return std::nullopt; }

				size_type uncompressed_size{};
				auto compressed_size = static_cast<size_type>(row->data.size() - sizeof(size_type));
				std::memcpy(&uncompressed_size, row->data.data(), sizeof(size_type));

				std::vector<char> uncompressed{};
				uncompressed.resize(uncompressed_size);
				auto result = ::uncompress2(
					(unsigned char*)uncompressed.data(), &uncompressed_size, 
					(const unsigned char*)row->data.data() + sizeof(size_type), &compressed_size);

				if (result != Z_OK) {
					return std::nullopt;
				}

				auto signature = row->signature;
				auto meta = row->meta;
				auto data = std::span<const char>{ uncompressed };

 				auto new_row = cache::row{ [next = std::move(row), data = std::move(uncompressed)]() noexcept {} };
				new_row.meta = meta;
				new_row.data = data;
				new_row.signature = signature;

				return new_row;
			}

			void put(std::string_view id, row&& r) {
				
				size_type uncompressed_size = static_cast<size_type>(r.data.size());
				auto compressed_size = compressBound(uncompressed_size);
				auto compressed = std::vector<char>{};
				compressed.resize(compressed_size + sizeof(size_type));

				auto status = compress2((unsigned char*)compressed.data() + sizeof(size_type), &compressed_size, (const unsigned char*)r.data.data(), r.data.size(), 9);
				compressed.resize(compressed_size + sizeof(size_type));

				*((size_type*)(compressed.data())) = uncompressed_size;

				if (status != Z_OK) {
					return;
				}

				auto signature = r.signature;
				auto meta = r.meta;
				auto data = std::span<const char>{ compressed };

				auto new_row = cache::row{ [next = std::move(r), data = std::move(compressed)]() noexcept {} };
				new_row.signature = signature;
				new_row.meta = meta;
				new_row.data = data;
				next()->put(id, std::move(new_row));
			
			}
			void remove(std::string_view id) {
				return next()->remove(id);
			}
		private:
			using size_type = unsigned long;
		};
#endif
		class guarded : public adaptor_base {
		public:
			using adaptor_base::adaptor_base;

			std::optional<cache::row> get(std::string_view id) override {
				std::scoped_lock lock{ guard };
				return next()->get(id);
			}

			void put(std::string_view id, row&& r) override {
				std::scoped_lock lock{ guard };
				return next()->put(id, std::move(r));
			}

			void remove(std::string_view id) override {
				std::scoped_lock lock{ guard };
				return next()->remove(id);
			}

		private:
			cache::handle wrapped;
			std::mutex guard;
		};
	}

	enum class cache_result {
		no_cache,
		not_in_cache,
		invalidated
	};

	class compiler {
	public:

		struct result {
			std::optional<cuda_module> module;
			std::string log;
			bool loaded_from_cache = false;
		};

		compiler(cache::handle kernel_cache = nullptr);

		bool find_system_cuda(const std::filesystem::path& hint = {});
		result compile(const spec& s);

		void use_cache(cache::handle new_cache) {
			k_cache.swap(new_cache);
		}

		bool add_include_path(const std::filesystem::path& p) {
			if (not std::filesystem::is_directory(p)) {
				return false;
			}

			auto absolute = std::filesystem::absolute(p);

			if (std::ranges::find(include_paths, absolute) != include_paths.end()) {
				return false;
			}

			include_paths.emplace_back(absolute);
			return true;
		}

	private:

		std::optional<cuda_module> load_cache(const spec& s);
		bool cache_header(std::string_view name, const std::filesystem::path&);

		cache::handle k_cache;

		std::vector<std::filesystem::path> include_paths{};
		std::vector<std::filesystem::path> library_paths{};

		struct hcache_info {
			std::string source;
			std::filesystem::path real_path;
			std::u8string path_string;
			std::size_t cached_ts;
		};

		std::map<std::string, hcache_info, std::less<>> hcache{};

		const std::string arch_flag;
		std::optional<std::vector<std::string>> cuda_include_flags;
	};

}

#endif