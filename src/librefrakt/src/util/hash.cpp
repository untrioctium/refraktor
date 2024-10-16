#include <xxhash.h>
#include <format>

#include <base32_crockford.hpp>
#include <base64_url_unpadded.hpp>

#include <librefrakt/util/hash.h>

using b64_codec = cppcodec::base64_url_unpadded;

rfkt::hash::state_t::state_t() noexcept: state_(XXH3_createState())
{
	XXH3_128bits_reset((XXH3_state_t*)state_);
}

rfkt::hash::state_t::~state_t() noexcept
{
	XXH3_freeState((XXH3_state_t*)state_);
}

rfkt::hash_t rfkt::hash::state_t::digest() const noexcept {
	auto xxh_hash = XXH3_128bits_digest((XXH3_state_t*)state_);
	return hash_t{ xxh_hash.low64, xxh_hash.high64 };
}

void rfkt::hash::state_t::update(const void* data, std::size_t len) noexcept {
	XXH3_128bits_update((XXH3_state_t*)state_, data, len);
}

auto rfkt::hash_t::str16() const noexcept -> std::string 
{
	return std::format("{:016X}{:016X}", bytes.second, bytes.first);
}

auto rfkt::hash_t::str32() const noexcept -> std::string
{
	return cppcodec::base32_crockford::encode((const unsigned char*)&bytes, sizeof(bytes));
}

auto rfkt::hash_t::str64() const noexcept -> std::string
{
	return cppcodec::base64_url_unpadded::encode((const unsigned char*)&bytes, sizeof(bytes));
}