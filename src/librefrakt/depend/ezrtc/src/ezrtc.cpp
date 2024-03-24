#include "ezrtc.h"

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

#include <cstdio>

#define EZRTC_CHECK_RURTC(expression)                              \
do {                                                               \
	if (const auto result = expression; result != RURTC_SUCCESS) { \
		printf("RTC error %d\n", result);						   \
		throw std::runtime_error(                                  \
			std::format(                                           \
				"NVRTC error: {} caused by '{}'",                  \
				rurtcGetErrorString(result),                       \
				#expression)                                       \
		);                                                         \
	}                                                              \
} while(0)

#define EZRTC_CHECK_ROCCU(expression)                              \
do {                                                               \
	if (const auto result = expression; result != RU_SUCCESS) {    \
		const char* error_str = nullptr;                           \
		ruGetErrorString(result, &error_str);                      \
		throw std::runtime_error(                                  \
			std::format(                                           \
				"CUDA error: {} caused by '{}'",                   \
				error_str, #expression)                            \
		);                                                         \
	}                                                              \
} while(0)

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

			block[block_index] = octet;
			block_index++;
			byte_count++;
			process_if_needed();
			return *this;
		}

		template<typename T>
			requires std::same_as<char, T> || std::same_as<unsigned char, T>
		constexpr sha1 & process_bytes(const T * start, std::size_t length) noexcept {
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
		constexpr sha1& process_bytes(const char(&str)[Size]) {
			return process_bytes(str, (str[Size - 1] == '\0') ? Size - 1 : Size);
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

RUresult ezrtc::kernel::launch_impl(RUfunction f, dim3 grid, dim3 block, RUstream stream, bool cooperative, void** args) noexcept {
	if (cooperative) {
		return ruLaunchCooperativeKernel(
			f,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0, stream,
			args
		);
	}
	else {
		return ruLaunchKernel(
			f,
			grid.x, grid.y, grid.z,
			block.x, block.y, block.z,
			0, stream,
			args, nullptr
		);
	}
}

namespace ezrtc::detail {

	auto last_mod(const std::filesystem::path& p) {
		return static_cast<std::size_t>(std::filesystem::last_write_time(p).time_since_epoch().count());
	}

	std::string get_arch() {
		RUdevice dev;
		EZRTC_CHECK_ROCCU(ruCtxGetDevice(&dev));

		int major = 0;
		int minor = 0;
		EZRTC_CHECK_ROCCU(ruDeviceGetAttribute(&major, RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
		EZRTC_CHECK_ROCCU(ruDeviceGetAttribute(&minor, RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

		return EZRTC_FMT_IMPL("sm_{}{}", major, minor);
	}

	std::string get_device_name() {
		RUdevice dev;
		EZRTC_CHECK_ROCCU(ruCtxGetDevice(&dev));

		std::string buf;
		buf.resize(128);

		ruDeviceGetName(buf.data(), buf.size(), dev);
		return buf;
	}

	constexpr static bool is_windows = []() {
#ifdef WIN32
		return true;
#else
		return false;
#endif
	}();

	constexpr const char* flag_to_option_cuda(compile_flag flag) {
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
		default: return "";
		}
	}

	constexpr const char* flag_to_option_rocm(compile_flag flag) {
		using enum compile_flag;
		using namespace std::literals;

		switch (flag) {
		case use_fast_math: return "-ffast-math";
		default: return "";
		}
	}

	constexpr const char* flag_to_option(compile_flag flag, roccu_api api) {
		return (api == ROCCU_API_CUDA) ? flag_to_option_cuda(flag) : flag_to_option_rocm(flag);
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
			return std::basic_string_view<CharT>{ (const CharT*)ptr, size };
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
				sv.copy((char8_t*)ptr, sv.size());
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

		scoped& operator = (scoped&& o) noexcept {
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
	if (const auto status = ruModuleLoadDataEx(&mod.handle, cubin.data(), 0, nullptr, nullptr);
		status != RU_SUCCESS) {
		return std::nullopt;
	}
	return mod;
}

bool ezrtc::cuda_module::load_kernel(std::string_view pretty, std::string_view mangled) {
	if (mangled.back() != '\0') return false;

	RUfunction f;

	if (const auto status = ruModuleGetFunction(&f, handle, mangled.data());
		status != RU_SUCCESS) {
		return false;
	}

	const auto [iter, inserted] = kernels.try_emplace(std::string{ pretty }, f);
	return inserted;
}

bool ezrtc::cuda_module::load_variable(std::string_view pretty, std::string_view mangled) {
	if (mangled.back() != '\0') return false;

	RUdeviceptr ptr;
	std::size_t size;

	if (const auto status = ruModuleGetGlobal(&ptr, &size, handle, mangled.data());
		status != RU_SUCCESS) {
		return false;
	}

	const auto [iter, inserted] = variables.try_emplace(std::string{ pretty }, variable{ ptr, size });
	return inserted;
}

ezrtc::spec& ezrtc::spec::define(std::string_view dname, std::string_view value) {
	return emplace_and_chain(defines, dname,
		(value.size() > 0)
		? EZRTC_FMT_IMPL("--define-macro={}={}", dname, value)
		: EZRTC_FMT_IMPL("--define-macro={}", dname)
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

	auto api = roccuGetApi();
	for (auto flag : compile_flags) {
		sha1.process_bytes(detail::flag_to_option(flag, api));
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
		sha1.process_bytes((const std::uint8_t*)path.data(), path.size());
	}

	for (const auto& [hname, hsource] : headers) {
		sha1.process_bytes(hname);
		sha1.process_bytes(hsource);
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
	RUcontext ctx;
	const auto status = ruCtxGetCurrent(&ctx);
	if (status != RU_SUCCESS or not ctx) throw std::runtime_error("cannot create ezrtc::compiler without a CUDA context");
}

bool ezrtc::compiler::find_system_cuda(const std::filesystem::path& hint) {
	if (cuda_include_flags.has_value()) return true;
	namespace fs = std::filesystem;

	//constexpr static auto major = CUDA_VERSION / 1000;
	//constexpr static auto minor = (CUDA_VERSION % 1000) / 10;
	//constexpr static auto patch = CUDA_VERSION % 10;

	//const auto target_version = EZRTC_FMT_IMPL("{}.{}.{}", major, minor, patch);

	auto check_and_add = [this](fs::path dir) -> bool {
		auto ec = std::error_code{};
		bool exists = fs::exists(dir / "cuda.h", ec);
		if (ec or not exists) return false;

		cuda_include_flags = std::vector <std::string>{};
		cuda_include_flags->emplace_back(std::format("--include-path={}", dir.string()));

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
	if (val && check_and_add(std::format("{}/include", val))) return true;

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
	auto mod = cuda_module::from_cubin(row->data);

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

	for (auto newline_pos = src.find('\n'); newline_pos != std::string_view::npos; newline_pos = src.find('\n')) {
		std::string_view line = src.substr(0, newline_pos);
		src = src.substr(newline_pos + 1);
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
	hcache.emplace(std::string{ name }, hcache_info{ std::move(src), path, path.u8string(), static_cast<std::size_t>(std::chrono::file_clock::now().time_since_epoch().count()) });
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

	std::vector<const char*> header_names{};
	std::vector<const char*> header_contents{};

	for (const auto& [name, contents] : s.headers) {
		header_names.push_back(name.c_str());
		header_contents.push_back(contents.data());
	}

	std::vector<const char*> compile_options{};

	const auto rapi = roccuGetApi();
	compile_options.push_back("--std=c++20");

	if (rapi == ROCCU_API_CUDA) {

		compile_options.push_back("-DROCCU_CUDA");

		compile_options.push_back(arch_flag.c_str());

		if (cuda_include_flags.has_value())
			for (const auto& flag : cuda_include_flags.value())
				compile_options.push_back(flag.c_str());

		for (const auto flag : s.compile_flags) {
			compile_options.push_back(detail::flag_to_option(flag, rapi));
		}
	}
	else if (rapi == ROCCU_API_ROCM) {
		compile_options.push_back("-ffast-math");
		compile_options.push_back("-O3");
		compile_options.push_back("-mno-cumode");
		compile_options.push_back("-DROCCU_ROCM");
	}

	for (const auto& [_, value] : s.defines) {
		compile_options.push_back(value.c_str());
	}

	using prog_scope = detail::scoped < rurtcProgram, [](rurtcProgram p) { rurtcDestroyProgram(&p); } > ;

	auto attempts = 0;
	auto make_program = [&]() {
		while (true) {
			attempts++;
			rurtcProgram prog_handle;
			EZRTC_CHECK_RURTC(
				rurtcCreateProgram(
					&prog_handle, s.source.c_str(), s.name.c_str(),
					header_names.size(), header_contents.data(), header_names.data()
				)
			);

			auto prog = prog_scope{ prog_handle };

			for (const auto& kernel : s.kernels) {
				EZRTC_CHECK_RURTC(rurtcAddNameExpression(prog, kernel.c_str()));
			}

			for (const auto& [_, expr] : s.variables) {
				EZRTC_CHECK_RURTC(rurtcAddNameExpression(prog, expr.c_str()));
			}

			auto status = rurtcCompileProgram(prog, compile_options.size(), compile_options.data());

			ret.log.clear();
			std::size_t log_size;
			EZRTC_CHECK_RURTC(rurtcGetProgramLogSize(prog, &log_size));
			if (log_size > 1) {
				ret.log.resize(log_size);
				EZRTC_CHECK_RURTC(rurtcGetProgramLog(prog, ret.log.data()));
			}

			if (status == RURTC_SUCCESS) {
				return std::make_pair(std::move(prog), status);
			}
			auto missing = find_missing_headers(ret.log);

			if (missing.empty()) {
				return std::make_pair(std::move(prog), status);
			}

			for (const auto& [name, header] : missing) {

				auto cwd = [&]() {
					if (name == s.name) { return std::filesystem::current_path(); }
					else {
						auto iter = hcache.find(std::string{ name });
						if (iter == hcache.end()) return std::filesystem::current_path();
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

	if (status != RURTC_SUCCESS) {
		return ret;
	}

	std::size_t ptx_size;
	EZRTC_CHECK_RURTC(rurtcGetAssemblySize(prog, &ptx_size));
	auto ptx = std::string(ptx_size, '\0');
	EZRTC_CHECK_RURTC(rurtcGetAssembly(prog, ptx.data()));

	/*
	RUlinkState ls_handle;
	EZRTC_CHECK_ROCCU(ruLinkCreate(0, 0, 0, &ls_handle));
	using link_scope = detail::scoped<RUlinkState, [](RUlinkState p) { ruLinkDestroy(p); } >;
	auto ls = link_scope{ ls_handle };

	EZRTC_CHECK_ROCCU(ruLinkAddData(ls, RU_JIT_INPUT_PTX,
		(void*)ptx.data(), ptx_size, s.name.c_str(),
		0, 0, 0));

	std::size_t cubin_size;
	char* cubin;
	EZRTC_CHECK_ROCCU(ruLinkComplete(ls, (void**)&cubin, &cubin_size));*/

	auto handle = cuda_module::from_cubin({ ptx.data(), ptx_size });
	if (not handle) {
		ret.log = "invalid cubin";
		return ret;
	}

	auto get_lowered_name = [&prog](std::string_view expression) {
		const char* name;
		EZRTC_CHECK_RURTC(rurtcGetLoweredName(prog, expression.data(), &name));
		return std::string_view{ name, std::strlen(name) + 1 }; // include null terminator
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
		row.data = { ptx.data(), ptx_size};
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

EZRTC_CHECK_HASH("", 0xda39a3ee, 0x5e6b4b0d, 0x3255bfef, 0x95601890, 0xafd80709);
EZRTC_CHECK_HASH("a million bright ambassadors of morning", 0x0014eed1, 0x2309f51f, 0x476e60b0, 0xcf096fcd, 0xfe68d486);
EZRTC_CHECK_HASH("overhead the albatross hangs motionless upon the air", 0x013efc4b, 0x6d93e766, 0x056df3ca, 0x4b36530b, 0xf1413ea7);