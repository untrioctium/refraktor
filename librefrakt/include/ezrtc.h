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

#include <filesystem>

#include <cuda.h>
#include <vector_types.h>

#include <limits>

// public api
namespace ezrtc {

	class kernel {
	public:

		explicit kernel(CUfunction f) noexcept : f(f) {}

		auto launch(dim3 grid, dim3 block, CUstream stream = nullptr, bool cooperative = false) const noexcept {
			return[f = this->f, grid, block, stream, cooperative](auto... args) noexcept {
				auto packed_args = std::array<void*, sizeof...(args)>{ &args... };
				return launch_impl(f, grid, block, stream, cooperative, packed_args.data());
			};
		}

		auto launch(std::uint32_t grid, std::uint32_t block, CUstream stream = nullptr, bool cooperative = false) const noexcept {
			return this->launch({ grid, 1, 1 }, { block, 1, 1 }, stream, cooperative);
		}

		int max_blocks_per_mp(int block_size) const noexcept {
			int num_blocks;
			cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, f, block_size, 0);
			return num_blocks;
		}

		int max_blocks_per_mp(const dim3& block_dim) const noexcept {
			return max_blocks_per_mp(block_dim.x * block_dim.y * block_dim.z);
		}

		std::pair<int, int> suggested_dims() const noexcept {
			std::pair<int, int> result;
			cuOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, nullptr, 0, 0);
			return result;
		}

		auto shared_bytes() const noexcept { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>(); }
		auto const_bytes() const noexcept { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>(); }
		auto local_bytes() const noexcept { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>(); }
		auto register_count() const noexcept { return attribute<CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS>(); }

	private:

		template<CUfunction_attribute a>
		auto attribute() const noexcept {
			int ret;
			cuFuncGetAttribute(&ret, a, f);
			return static_cast<std::size_t>(ret);
		}


		static CUresult launch_impl(CUfunction f, dim3 grid, dim3 block, CUstream stream, bool cooperative, void** args) noexcept {
			if (cooperative) {
				return cuLaunchCooperativeKernel(
					f,
					grid.x, grid.y, grid.z,
					block.x, block.y, block.z,
					0, stream,
					args
				);
			}
			else {
				return cuLaunchKernel(
					f,
					grid.x, grid.y, grid.z,
					block.x, block.y, block.z,
					0, stream,
					args, nullptr
				);
			}
		}
		CUfunction f;
	};

	class variable {
	public:

		variable(CUdeviceptr ptr, std::size_t size) noexcept : ptr_(ptr), size_(size) {}

		explicit(false) operator CUdeviceptr() const noexcept {
			return ptr_;
		}

		auto ptr() const noexcept {
			return ptr_;
		}

		auto size() const noexcept { return size_; }

	private:

		CUdeviceptr ptr_;
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
			if (handle) cuModuleUnload(handle);
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

		CUmodule handle = nullptr;
		std::unordered_map<std::string, CUfunction> kernels = {};
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
		spec& define(std::string_view name, std::string_view value = {});
		spec& define(std::string_view name, std::integral auto value) {
			return define(name, std::to_string(value));
		}
		spec& kernel(std::string_view name) { return emplace_and_chain(kernels, name); }
		spec& variable(std::string_view name) { return emplace_and_chain(variables, name, std::format("&{}", name)); }
		spec& header(std::string_view name, const std::string& source) {
			return emplace_and_chain(headers, name, std::string_view{ source.c_str(), source.size() + 1 });
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
			row.signature = std::move(sig);
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
				
				size_type uncompressed_size = r.data.size();
				auto compressed_size = compressBound(uncompressed_size);
				auto compressed = std::vector<char>{};
				compressed.resize(compressed_size + sizeof(size_type));

				auto status = compress2((unsigned char*)compressed.data() + sizeof(size_type), &compressed_size, (const unsigned char*)r.data.data(), r.data.size(), 9);
				compressed.resize(compressed_size + sizeof(size_type));

				*((size_type*)(compressed.data())) = static_cast<size_type>(uncompressed_size);

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

		bool find_system_cuda(std::filesystem::path hint = {});
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

#ifdef EZRTC_IMPLEMENTATION_UNIT

#ifdef EZRTC_USE_FMTLIB
#include <fmt/fmt.h>
#define EZRTC_FMT_IMPL fmt::format
#else
#include <format>
#define EZRTC_FMT_IMPL std::format
#endif

#ifdef EZRTC_ENABLE_RTTI
#define NVRTC_GET_TYPE_NAME 1
#endif
#include <nvrtc.h>


#define EZRTC_CHECK_NVRTC(expression)                              \
{                                                                  \
	if (const auto result = expression; result != NVRTC_SUCCESS) { \
		throw std::runtime_error(                                  \
			std::format(                                           \
				"NVRTC error: {} caused by '{}'",                  \
				nvrtcGetErrorString(result),                       \
				#expression)                                       \
		);                                                         \
	}                                                              \
}                                                                  \

#define EZRTC_CHECK_CUDA(expression)                               \
{                                                                  \
	if (const auto result = expression; result != CUDA_SUCCESS) {  \
		const char* error_str = nullptr;                           \
		cuGetErrorString(result, &error_str);                      \
		throw std::runtime_error(                                  \
			std::format(                                           \
				"CUDA error: {} caused by '{}'",                   \
				error_str, #expression)                            \
		);                                                         \
	}                                                              \
} 

namespace ezrtc::detail::hash {

	// A reimplementation of the TinySHA1 library. The original license follows:

	/*
	 *
	 * TinySHA1 - a header only implementation of the SHA1 algorithm in C++. Based
	 * on the implementation in boost::uuid::details.
	 *
	 * SHA1 Wikipedia Page: http://en.wikipedia.org/wiki/SHA-1
	 *
	 * Copyright (c) 2012-22 SAURAV MOHAPATRA <mohaps@gmail.com>
	 *
	 * Permission to use, copy, modify, and distribute this software for any
	 * purpose with or without fee is hereby granted, provided that the above
	 * copyright notice and this permission notice appear in all copies.
	 *
	 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
	 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
	 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
	 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
	 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
	 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
	 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
	*/

	class sha1 {
	public:
		using byte = std::uint8_t;
		using digest_t = std::array<std::uint32_t, 5>;

		constexpr sha1& reset() noexcept {
			digest = reset_bytes;
			block_index = 0;
			byte_count = 0;
			dirty = true;
			return *this;
		}

		constexpr sha1& process_byte(byte octet) noexcept {
			dirty = true;

			block[block_index++] = octet;
			byte_count++;
			process_if_needed();
			return *this;
		}

		template<typename T>
		requires std::same_as<char, T> || std::same_as<unsigned char, T>
		constexpr sha1& process_bytes(const T* start, std::size_t length) noexcept {
			if (length == 0) return *this;
			
			dirty = true;
			byte_count += length;

			for (std::size_t offset = 0; offset < length;) {
				const auto bytes_left = block_size - block_index;
				const auto to_copy = std::min(length - offset, bytes_left);

				
				std::transform(start + offset, start + offset + to_copy, block.data() + block_index, [](char v) {return std::bit_cast<char, T>(v); });

				block_index += to_copy;
				process_if_needed();
				offset += to_copy;
			}

			return *this;
		}

		constexpr sha1& process_nulls(std::size_t count) noexcept {
			if (count == 0) return *this;

			dirty = true;
			byte_count += count;

			for (std::size_t i = 0; i < count;) {
				const auto bytes_left = block_size - block_index;
				const auto to_fill = std::min(count - i, bytes_left);

				std::fill(block.data() + block_index, block.data() + block_index + to_fill, 0);

				block_index += to_fill;
				process_if_needed();
				i += to_fill;
			}

			return *this;
		}

		constexpr sha1& process_bytes(const byte* start, const byte* end) noexcept {
			return process_bytes(start, end - start);
		}

		constexpr sha1& process_bytes(std::span<const byte> sp) noexcept {
			return process_bytes(sp.data(), sp.size());
		}

		constexpr sha1& process_bytes(std::string_view sv) noexcept {
			return process_bytes(sv.data(), sv.length());
		}

		constexpr sha1& process_bytes(const char* str) noexcept {
			return process_bytes(std::string_view{ str });
		}

		template<std::size_t Size>
		constexpr sha1& process_bytes(const std::array<byte, Size>& bytes) noexcept {
			return process_bytes(bytes.data(), bytes.size());
		}

		template<std::size_t Size>
		constexpr sha1& process_bytes(const char (&str)[Size]) {
			return process_bytes(str, (str[Size - 1] == '\0')? Size - 1: Size);
		}

		template<typename T>
		requires (std::is_trivially_copyable_v<T>)
		constexpr sha1& process_bytes(const T& type) {
			return process_bytes(std::bit_cast<std::array<byte, sizeof(T)>>(type));
		}

		constexpr digest_t get_digest() noexcept {
			make_digest();
			return digest;
		}

	private:

		constexpr void make_digest() noexcept {
			if (!dirty) return;

			auto bit_count_big_endian = [](auto count) {
				if constexpr (std::endian::native == std::endian::little) {
					return std::byteswap(count);
				}
				else return count;
			}(byte_count * 8);

			process_byte(0x80);

			if (constexpr auto target_index = block_size - sizeof(std::uint64_t); block_index < target_index) {
				process_nulls(target_index - block_index);
			}
			else {
				process_nulls(block_size - block_index + target_index);
			}

			process_bytes(bit_count_big_endian);

			dirty = false;
		}

		constexpr static std::size_t block_size = 64;

		constexpr static std::uint32_t left_rotate(std::uint32_t value, std::size_t count) noexcept {
			return (value << count) ^ (value >> (32 - count));
		}

		constexpr static auto reset_bytes = digest_t{ 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };

		std::array<std::uint8_t, block_size> block{};
		digest_t digest = reset_bytes;
		std::size_t block_index = 0;
		std::size_t byte_count = 0;
		bool dirty = true;

		constexpr void process_if_needed() noexcept {
			if (block_index == block_size) {
				block_index = 0;
				process_block();
			}
		}

		constexpr void process_block() noexcept {
			auto w = std::array<std::uint32_t, 80>{};

			for (std::size_t i = 0; i < 16; i++) {
				w[i] = static_cast<std::uint32_t>(block[i * 4 + 0]) << 24;
				w[i] |= static_cast<std::uint32_t>(block[i * 4 + 1]) << 16;
				w[i] |= static_cast<std::uint32_t>(block[i * 4 + 2]) << 8;
				w[i] |= static_cast<std::uint32_t>(block[i * 4 + 3]);
			}

			for (std::size_t i = 16; i <= 31; i++) {
				w[i] = left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
			}

			for (std::size_t i = 32; i < 80; i++) {
				w[i] = left_rotate(w[i - 6] ^ w[i - 16] ^ w[i - 28] ^ w[i - 32], 2);
			}

			std::uint32_t a = digest[0];
			std::uint32_t b = digest[1];
			std::uint32_t c = digest[2];
			std::uint32_t d = digest[3];
			std::uint32_t e = digest[4];

			for (std::size_t i = 0; i < 80; i++) {
				std::uint32_t f = 0;
				std::uint32_t k = 0;

				if (i < 20) {
					f = (b & c) | (~b & d);
					k = 0x5A827999;
				}
				else if (i < 40) {
					f = b ^ c ^ d;
					k = 0x6ED9EBA1;
				}
				else if (i < 60) {
					f = (b & c) | (b & d) | (c & d);
					k = 0x8F1BBCDC;
				}
				else {
					f = b ^ c ^ d;
					k = 0xCA62C1D6;
				}
				std::uint32_t temp = left_rotate(a, 5) + f + e + k + w[i];
				e = d;
				d = c;
				c = left_rotate(b, 30);
				b = a;
				a = temp;
			}

			digest[0] += a;
			digest[1] += b;
			digest[2] += c;
			digest[3] += d;
			digest[4] += e;
		}
	};
}

namespace ezrtc::detail {

	auto last_mod(const std::filesystem::path& p) {
		return static_cast<std::size_t>(std::filesystem::last_write_time(p).time_since_epoch().count());
	}

	std::string get_arch() {
		CUdevice dev;
		EZRTC_CHECK_CUDA(cuCtxGetDevice(&dev));

		int major = 0;
		int minor = 0;
		EZRTC_CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
		EZRTC_CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

		return EZRTC_FMT_IMPL("sm_{}{}", major, minor);
	}

	std::string get_device_name() {
		CUdevice dev;
		EZRTC_CHECK_CUDA(cuCtxGetDevice(&dev));

		std::string buf;
		buf.resize(128);

		cuDeviceGetName(buf.data(), buf.size(), dev);
		return buf;
	}

	constexpr static bool is_windows = []() {
#ifdef WIN32
		return true;
#else
		return false;
#endif
	}();

	constexpr const char* flag_to_option(compile_flag flag) {
		using enum compile_flag;
		using namespace std::literals;

		switch (flag) {
		case relocatable_code: return "-dc";
		case extensible_whole_program: return "-ewp";
		case device_debug: return "-G";
		case generate_line_info: return "-lineinfo";
		case enable_optimization: return "--dopt=on";
		case use_fast_math: return "-use_fast_math";
		case extra_device_vectorization: return "-extra-device-vectorization";
		case disable_warnings: return "-w";
		case restricted_kernel_pointers: return "-restrict";
		case default_device: return "-default-device";
		case device_int128: return "-device-int128";
		case warn_double_usage: return "--ptxas-options=--warn-on-double-precision-use";
		default: std::unreachable();
		}
	}

	class metadata_iterator {
	public:
		explicit metadata_iterator(std::span<const char> bytes) : bytes(bytes) {}

		template<typename T>
		bool has_a() const noexcept {
			return sizeof(T) <= bytes.size();
		}

		template<typename T>
		T pop_a() noexcept {
			auto ptr = bytes.data();
			bytes = bytes.subspan(sizeof(T));
			return *reinterpret_cast<const T*>(ptr);
		}

		bool has_long() const { return has_a<std::size_t>(); }
		auto pop_long() { return pop_a<std::size_t>(); }

		bool has_len() const { return has_a<std::uint16_t>(); }
		auto pop_len() { return pop_a<std::uint16_t>(); }

		bool has_str(std::size_t size) const {
			return size <= bytes.size();
		}

		template<typename CharT = char>
		std::basic_string_view<CharT> pop_str(std::size_t size) noexcept {
			auto ptr = bytes.data();
			bytes = bytes.subspan(size);
			return std::basic_string_view<CharT>{ (CharT*) ptr, size };
		}

		explicit operator bool() const noexcept {
			return !bytes.empty();
		}

	private:
		std::span<const char> bytes;
	};

	struct metadata {
		constexpr static std::size_t serial = 1;

		std::size_t create_ts;

		struct dependency {
			std::u8string_view path;
			std::size_t last_mod;
		};

		struct name {
			std::string_view pretty;
			std::string_view mangled;
		};

		std::vector<dependency> dependencies;
		std::vector<name> kernels;
		std::vector<name> variables;

		static std::optional<metadata> from_bytes(std::span<const char> bytes) {

			if (bytes.size() < sizeof(std::size_t) * 2 + 3 * sizeof(length_t)) {
				return std::nullopt;
			}

			#define MUST_LONG(variable) \
				if (not iter.has_long()) { return std::nullopt; } \
			    auto variable = iter.pop_long()

			#define MUST_LEN(variable) \
				if (not iter.has_len()) { return std::nullopt; } \
			    auto variable = iter.pop_len()

			#define MUST_STR(variable, t)           \
			    auto variable = std## t ## string_view{}; \
				do {                                   \
					MUST_LEN(s_len);				\
                                                    \
					if (not iter.has_str(s_len))    \
						{ return std::nullopt; }    \
					variable = iter.pop_str<decltype(variable)::value_type>(s_len); \
                } while(0)                               
				

			auto iter = metadata_iterator{ bytes };
			auto md = metadata{};

			// check serial
			if (auto version = iter.pop_long(); version != serial) { return std::nullopt; }

			// get creation timestamp
			if (!iter.has_long()) { return std::nullopt; }
			md.create_ts = iter.pop_long();

			// get count of dependencies
			MUST_LEN(dep_length);

			// loop through dependencies
			for (int i = 0; i < dep_length; i++) {
				// get path string
				MUST_STR(path, ::u8);

				// get last modified timestamp
				MUST_LONG(last_mod);

				// emplace
				md.dependencies.emplace_back(path, last_mod);
			}

			// get count of kernels
			MUST_LEN(k_length);

			// no kernels is a bit useless, must be an error
			if (k_length == 0) { return std::nullopt; }

			// loop through kernels
			for (int i = 0; i < k_length; i++) {
				MUST_STR(pretty, ::);
				MUST_STR(mangled, ::);

				if (mangled.back() != '\0') {
					return std::nullopt;
				}

				md.kernels.emplace_back(pretty, mangled);
			}

			MUST_LEN(v_length);
			for (int i = 0; i < v_length; i++) {
				MUST_STR(pretty, ::);
				MUST_STR(mangled, ::);

				if (mangled.back() != '\0') {
					return std::nullopt;
				}

				md.variables.emplace_back(pretty, mangled);
			}

			// if we still have data left, it is an error
			if (iter) return std::nullopt;

			return md;

			#undef MUST_LEN
			#undef MUST_LONG
			#undef MUST_STR
		}

		std::vector<char> to_bytes() const {

			// determine how big the vector should be
			std::size_t n_bytes = sizeof(std::size_t) * 2 + sizeof(std::uint16_t) * 3;

			for (const auto& [path, _] : dependencies) {
				n_bytes += path.size() + sizeof(std::uint16_t) + sizeof(std::size_t);
			}

			for (const auto& [pretty, mangled] : kernels) {
				n_bytes += pretty.size() + mangled.size() + sizeof(std::uint16_t) * 2;
			}

			for (const auto& [pretty, mangled] : variables) {
				n_bytes += pretty.size() + mangled.size() + sizeof(std::uint16_t) * 2;
			}

			auto bytes = std::vector<char>{};
			bytes.resize(n_bytes);
			char* ptr = bytes.data();

			auto put_long = [&ptr](std::size_t l) noexcept {
				*reinterpret_cast<std::size_t*>(ptr) = l;
				ptr += sizeof(std::size_t);
			};

			auto put_len = [&ptr](std::uint16_t l) noexcept {
				*reinterpret_cast<std::uint16_t*>(ptr) = l;
				ptr += sizeof(std::uint16_t);
			};

			auto put_str = [&ptr, &put_len](std::string_view sv) {
				put_len(sv.size());
				sv.copy(ptr, sv.size());
				ptr += sv.size();
			};

			auto put_str_u8 = [&ptr, &put_len](std::u8string_view sv) {
				put_len(sv.size());
				sv.copy((char8_t*) ptr, sv.size());
				ptr += sv.size();
			};

			put_long(serial);
			put_long(create_ts);

			put_len(dependencies.size());
			for (const auto& [path, ts] : dependencies) {
				put_str_u8(path);
				put_long(ts);
			}

			put_len(kernels.size());
			for (const auto& [pretty, mangled] : kernels) {
				put_str(pretty);
				put_str(mangled);
			}

			put_len(variables.size());
			for (const auto& [pretty, mangled] : variables) {
				put_str(pretty);
				put_str(mangled);
			}

			return bytes;
		}

	private:
		using length_t = std::uint16_t;
	};

	template<typename T, auto Destructor, T Default = nullptr>
	class scoped {
	public:
		~scoped() {
			Destructor(val);
		}

		explicit scoped(T val) : val(std::move(val)) {}

		scoped(const scoped&) = delete;
		scoped& operator=(const scoped&) = delete;

		scoped(scoped&& o) noexcept {
			std::swap(val, o.val);
		}

		scoped& operator = (scoped && o) noexcept{
			std::swap(val, o.val);
			return *this;
		}

		explicit(false) operator T() {
			return val;
		}
	private:
		T val = Default;
	};

}

std::optional<ezrtc::cuda_module> ezrtc::cuda_module::from_cubin(std::span<const char> cubin) {

	if (cubin.size() < 4
		|| cubin[0] != 0x7F
		|| cubin[1] != 'E'
		|| cubin[2] != 'L'
		|| cubin[3] != 'F'
	) {
		return std::nullopt;
	}

	auto mod = cuda_module{};
	if (const auto status = cuModuleLoadDataEx(&mod.handle, cubin.data(), 0, nullptr, nullptr); 
		status != CUDA_SUCCESS) {
		return std::nullopt;
	}
	return mod;
}

bool ezrtc::cuda_module::load_kernel(std::string_view pretty, std::string_view mangled) {
	if (mangled.back() != '\0') return false;

	CUfunction f;

	if (const auto status = cuModuleGetFunction(&f, handle, mangled.data()); 
		status != CUDA_SUCCESS) {
		return false;
	}

	const auto [iter, inserted] = kernels.try_emplace(std::string{ pretty }, f);
	return inserted;
}

bool ezrtc::cuda_module::load_variable(std::string_view pretty, std::string_view mangled) {
	if (mangled.back() != '\0') return false;

	CUdeviceptr ptr;
	std::size_t size;

	if (const auto status = cuModuleGetGlobal(&ptr, &size, handle, mangled.data()); 
		status != CUDA_SUCCESS) {
		return false;
	}

	const auto [iter, inserted] = variables.try_emplace(std::string{ pretty }, variable{ ptr, size });
	return inserted;
}

ezrtc::spec& ezrtc::spec::define(std::string_view name, std::string_view value) {
	return emplace_and_chain(defines, name,
		(value.size() > 0)
		? EZRTC_FMT_IMPL("--define-macro={}={}", name, value)
		: EZRTC_FMT_IMPL("--define-macro={}", name)
	);
}

std::string_view ezrtc::spec::signature() const {

	if (cached_signature.has_value()) {
		return cached_signature.value();
	}

	auto sha1 = detail::hash::sha1{};

	sha1.process_bytes(detail::get_arch())
		.process_bytes(detail::get_device_name())
		.process_bytes(name)
		.process_bytes(source);

	for (auto flag : compile_flags) {
		sha1.process_bytes(detail::flag_to_option(flag));
	}

	for (const auto& [_, v] : defines) {
		sha1.process_bytes(v);
	}

	for (const auto& kernel : kernels) {
		sha1.process_bytes(kernel);
	}

	for (const auto& [_, v] : variables) {
		sha1.process_bytes(v);
	}

	for (const auto& path : dependencies) {
		sha1.process_bytes((std::uint8_t*) path.data(), path.size());
	}

	for (const auto& [name, source] : headers) {
		sha1.process_bytes(name);
		sha1.process_bytes(source);
	}

	const auto digest = sha1.get_digest();

	cached_signature = EZRTC_FMT_IMPL(
		"{:08x}{:08x}{:08x}{:08x}{:08x}", 
		digest.at(0), digest.at(1), digest.at(2), digest.at(3), digest.at(4));

	return cached_signature.value();
}

ezrtc::compiler::compiler(cache::handle kernel_cache) :
	k_cache(std::move(kernel_cache)),
	arch_flag("--gpu-architecture=" + detail::get_arch()) 
{
		CUcontext ctx;
		const auto status = cuCtxGetCurrent(&ctx);
		if (status != CUDA_SUCCESS or not ctx) throw std::runtime_error("cannot create ezrtc::compiler without a CUDA context");
}

bool ezrtc::compiler::find_system_cuda(std::filesystem::path hint) {
	if (cuda_include_flags.has_value()) return true;
	namespace fs = std::filesystem;

	constexpr static auto major = CUDA_VERSION / 1000;
	constexpr static auto minor = (CUDA_VERSION % 1000) / 10;
	constexpr static auto patch = CUDA_VERSION % 10;

	const auto target_version = EZRTC_FMT_IMPL("{}.{}.{}", major, minor, patch);

	auto check_and_add = [this](fs::path dir) -> bool {
		auto ec = std::error_code{};
		bool exists = fs::exists(dir / "version.json", ec);
		if (ec or not exists) return false;

		cuda_include_flags = std::vector <std::string>{};
		cuda_include_flags->emplace_back(std::format("--include-path={}", (dir / "include").string()));

		for (auto& flag : cuda_include_flags.value()) {
			for (auto& c : flag) {
				if (c == '\\') c = '/';
			}
		}

		return true;
	};

	// check our user's hint
	if (!hint.empty() && check_and_add(hint)) return true;

	// check env
	auto val = std::getenv("CUDA_PATH");
	if (val && check_and_add(val)) return true;

	return false;
}

std::optional<ezrtc::cuda_module> ezrtc::compiler::load_cache(const ezrtc::spec& s) {
	if (!k_cache) {
		return std::nullopt;
	}

	auto row = k_cache->get(s.name);
	if (!row.has_value()) {
		return std::nullopt;
	}

	auto md = detail::metadata::from_bytes(row->meta);
	if (!md.has_value()) {
		return std::nullopt;
	}

	if (row->signature != s.signature()) {
		return std::nullopt;
	}

	const bool meta_is_valid = [](const detail::metadata& md, const spec& s) -> bool {
		const bool all_kernels_are_equal = std::equal(
			md.kernels.begin(), md.kernels.end(), s.kernels.begin(),
			[](const auto& md_kernel, const auto& spec_kernel) {
				return md_kernel.pretty == spec_kernel;
			});

		if (!all_kernels_are_equal) {
			return false;
		}

		const bool all_variables_are_equal = std::equal(
			md.variables.begin(), md.variables.end(), s.variables.begin(),
			[](const auto& md_variable, const auto& spec_variable) {
				return md_variable.pretty == spec_variable.first;
			});

		if (!all_variables_are_equal) {
			return false;
		}

		const bool dependencies_not_modified = std::all_of(
			md.dependencies.begin(), md.dependencies.end(),
			[&md](const auto& dep) {
				auto ec = std::error_code{};
				if (bool exists = std::filesystem::exists(dep.path, ec);
					ec or not exists
				) {
					return false;
				}

				return dep.last_mod == detail::last_mod(dep.path);
			}
		);

		return dependencies_not_modified;

	}(md.value(), s);

	if (!meta_is_valid) {
		return std::nullopt;
	}

	// everything is valid now. do the actual loading
	auto mod = cuda_module::from_cubin( row->data );

	for (const auto& kernel : md->kernels) {
		if (!mod->load_kernel(kernel.pretty, kernel.mangled)) {
			return std::nullopt;
		}
	}

	for (const auto& variable : md->variables) {
		if (!mod->load_variable(variable.pretty, variable.mangled)) {
			return std::nullopt;
		}
	}

	return mod;
}

std::vector<std::pair<std::string_view, std::string_view>> find_missing_headers(std::string_view src) {

	auto ret = std::vector<std::pair<std::string_view, std::string_view>>{};

	constexpr static std::string_view search_string = "t open source file \"";

	for (auto newline_pos = src.find('\n'); newline_pos != std::string_view::npos; newline_pos = src.find(newline_pos)) {
		std::string_view line = src.substr(0, newline_pos);
		src = src.substr(newline_pos);
		auto filename_end = line.find('(');
		auto included_start = line.find(search_string);
		auto included_end = line.find("\"", included_start + search_string.size());

		if (filename_end == std::string_view::npos or included_start == std::string_view::npos) continue;

		std::string_view filename = line.substr(0, filename_end);

		auto start = included_start + search_string.size();
		auto len = included_end - start;
		std::string_view included = line.substr(start, len);

		ret.emplace_back(filename, included);
	}

	return ret;

}

std::optional<std::filesystem::path> find_header(std::string_view name, const std::span<const std::filesystem::path> include_dirs, const std::filesystem::path cwd) {
	{
		auto candidate = cwd / name;
		auto ec = std::error_code{};
		if (auto exists = std::filesystem::exists(candidate, ec); !ec && exists) {
			return candidate;
		}
	}
	
	for (const auto& path : include_dirs) {
		auto candidate = path / name;
		auto ec = std::error_code{};
		if (auto exists = std::filesystem::exists(candidate, ec); !ec && exists) {
			return candidate;
		}
	}

	return std::nullopt;
}

bool ezrtc::compiler::cache_header(std::string_view name, const std::filesystem::path& path) {

	auto ec = std::error_code{};
	auto size = std::filesystem::file_size(path, ec);
	auto is = std::ifstream{ path, std::ios::in };
	if (ec or not is.good()) return false;

	std::string src{};
	src.resize(size);
	is.read(src.data(), size);
	hcache.emplace(std::string{ name }, hcache_info{ std::move(src), path, path.u8string(), static_cast<std::size_t>(std::chrono::file_clock::now().time_since_epoch().count())});
	return true;
}

ezrtc::compiler::result ezrtc::compiler::compile(const ezrtc::spec& s) {

	auto ret = result{};

	ret.module = load_cache(s);
	if (ret.module) {
		ret.loaded_from_cache = true;
		return ret;
	}

	auto md = detail::metadata{};
	md.create_ts = std::chrono::file_clock::now().time_since_epoch().count();

	for (const auto& dep : s.dependencies) {
		md.dependencies.emplace_back(dep, detail::last_mod(dep));
	}



	std::vector<const char*> header_names {"cstdio", "iostream"};
	std::vector<const char*> header_contents {"", ""};

	for (const auto& [name, contents] : s.headers) {
		header_names.push_back(name.c_str());
		header_contents.push_back(contents.data());
	}

	std::vector<const char*> compile_options{ "--std=c++20" };
	compile_options.push_back(arch_flag.c_str());
	if (cuda_include_flags.has_value()) 
		for(const auto& flag: cuda_include_flags.value())
			compile_options.push_back(flag.c_str());

	for (const auto flag : s.compile_flags) {
		compile_options.push_back(detail::flag_to_option(flag));
	}

	for (const auto& [_, value] : s.defines) {
		compile_options.push_back(value.c_str());
	}

	using prog_scope = detail::scoped<nvrtcProgram, [](nvrtcProgram p) { nvrtcDestroyProgram(&p); } > ;

	auto make_program = [&]() {
		while (true) {
			nvrtcProgram prog_handle;
			EZRTC_CHECK_NVRTC(
				nvrtcCreateProgram(
					&prog_handle, s.source.c_str(), s.name.c_str(),
					header_names.size(), header_contents.data(), header_names.data()
				)
			);

			auto prog = prog_scope{ prog_handle };

			for (const auto& kernel : s.kernels) {
				EZRTC_CHECK_NVRTC(nvrtcAddNameExpression(prog, kernel.c_str()));
			}

			for (const auto& [_, expr] : s.variables) {
				EZRTC_CHECK_NVRTC(nvrtcAddNameExpression(prog, expr.c_str()));
			}

			auto status = nvrtcCompileProgram(prog, compile_options.size(), compile_options.data());

			ret.log.clear();
			std::size_t log_size;
			EZRTC_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
			if (log_size > 1) {
				ret.log.resize(log_size);
				EZRTC_CHECK_NVRTC(nvrtcGetProgramLog(prog, ret.log.data()));
			}

			if (status == NVRTC_SUCCESS) {
				return std::make_pair(std::move(prog), status );
			}
			auto missing = find_missing_headers(ret.log);

			if (missing.empty()) {
				return std::make_pair(std::move(prog), status);
			}

			for (const auto& [name,header] : missing) {

				auto cwd = [&]() {
					if (name == s.name) { return std::filesystem::current_path(); }
					else {
						auto iter = hcache.find(std::string{ name });
						if(iter == hcache.end()) return std::filesystem::current_path();
						return iter->second.real_path.parent_path();
					}
				}();

				auto path = find_header(header, this->include_paths, cwd);
				if (not path or not this->cache_header(header, path.value())) {
					return std::make_pair(std::move(prog), status);
				}

				auto iter = hcache.find(std::string{ header });
				md.dependencies.push_back({ iter->second.path_string, detail::last_mod(iter->second.real_path) });
				header_names.push_back(iter->first.c_str());
				header_contents.push_back(iter->second.source.c_str());
			}
		}
	};

	auto&& [prog, status] = make_program();

	if (status != NVRTC_SUCCESS) {
		return ret;
	}

	std::size_t ptx_size;
	EZRTC_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
	auto ptx = std::string(ptx_size, '\0');
	EZRTC_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));

	CUlinkState ls_handle;
	EZRTC_CHECK_CUDA(cuLinkCreate(0, 0, 0, &ls_handle));
	using link_scope = detail::scoped<CUlinkState, cuLinkDestroy>;
	auto ls = link_scope{ ls_handle };

	EZRTC_CHECK_CUDA(cuLinkAddData(ls, CU_JIT_INPUT_PTX,
		(void*)ptx.data(), ptx_size, s.name.c_str(),
		0, 0, 0));

	std::size_t cubin_size;
	char* cubin;
	EZRTC_CHECK_CUDA(cuLinkComplete(ls, (void**)&cubin, &cubin_size));

	auto handle = cuda_module::from_cubin({cubin, cubin_size});
	if (not handle) {
		ret.log = "invalid cubin";
		return ret;
	}

	auto get_lowered_name = [&prog](std::string_view expression) {
		const char* name;
		EZRTC_CHECK_NVRTC(nvrtcGetLoweredName(prog, expression.data(), &name));
		return std::string_view{name, std::strlen(name) + 1};
	};

	for (const auto& kernel : s.kernels) {
		auto lname = get_lowered_name(kernel);
		handle->load_kernel(kernel, lname);
		md.kernels.emplace_back(kernel, lname);
	}

	for (const auto& [name, expression] : s.variables) {
		auto lname = get_lowered_name(expression);
		handle->load_variable(name, lname);
		md.variables.emplace_back(name, lname);
	}

	ret.module = std::move(handle);

	if (k_cache) {
		cache::row row{ {} };
		auto meta_bytes = md.to_bytes();
		row.signature = s.signature();
		row.meta = { meta_bytes.data(), meta_bytes.size() };
		row.data = { cubin, cubin_size };
		k_cache->put(s.name, std::move(row));
	}

	return ret;
}

// static assertions

#define EZRTC_ASSERT_COPYABLE(type) \
	static_assert(std::is_copy_assignable_v<type>, #type " should be copy assignable"); \
	static_assert(std::is_copy_constructible_v<type>,  #type " should be copy constructible")

#define EZRTC_ASSERT_NONCOPYABLE(type) \
	static_assert(not std::is_copy_assignable_v<type>, #type " should not be copy assignable"); \
	static_assert(not std::is_copy_constructible_v<type>,  #type " should not be copy constructible")

#define EZRTC_ASSERT_MOVABLE(type) \
	static_assert(std::is_move_assignable_v<type>, #type " should be move assignable"); \
	static_assert(std::is_move_constructible_v<type>,  #type " should be move constructible")

#define EZRTC_ASSERT_NONMOVABLE(type) \
	static_assert(not std::is_move_assignable_v<type>, #type " should not be move assignable"); \
	static_assert(not std::is_move_constructible_v<type>,  #type " should not be move constructible")

EZRTC_ASSERT_COPYABLE(ezrtc::kernel);
EZRTC_ASSERT_MOVABLE(ezrtc::kernel);

EZRTC_ASSERT_NONCOPYABLE(ezrtc::cuda_module);
EZRTC_ASSERT_MOVABLE(ezrtc::cuda_module);

#define EZRTC_CHECK_HASH(ct, ...) static_assert(ezrtc::detail::hash::sha1().process_bytes(ct).get_digest() == ezrtc::detail::hash::sha1::digest_t{__VA_ARGS__});

EZRTC_CHECK_HASH("", 0xda39a3ee, 0x5e6b4b0d, 0x3255bfef, 0x95601890, 0xafd80709 );
EZRTC_CHECK_HASH("a million bright ambassadors of morning", 0x0014eed1, 0x2309f51f, 0x476e60b0, 0xcf096fcd, 0xfe68d486);
EZRTC_CHECK_HASH("overhead the albatross hangs motionless upon the air", 0x013efc4b, 0x6d93e766, 0x056df3ca, 0x4b36530b, 0xf1413ea7);

#undef EZRTC_ASSERT_COPYABLE
#undef EZRTC_ASSERT_NONCOPYABLE
#undef EZRTC_ASSERT_MOVABLE
#undef EZRTC_ASSERT_NONMOVABLE

#endif

#endif