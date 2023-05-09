#include <zlib.h>
#include <zip.h>

#include <librefrakt/util/zlib.h>

std::vector<char> rfkt::zlib::compress(const std::vector<char>& data, unsigned int level)
{
    return compress(data.data(), data.size(), level);
}

std::vector<char> rfkt::zlib::compress(const void* data, std::size_t len, unsigned int level)
{
    std::vector<char> ret;
    ret.resize(len);
    auto comp_size = static_cast<unsigned long>(len);
    ::compress2((Bytef*) ret.data(), &comp_size, (const Bytef*) data, len, level);
    ret.resize(comp_size);

    char* len_bytes = (char*)&len;
    ret.insert(ret.begin(), len_bytes, len_bytes + sizeof(len));
    return ret;
}

std::vector<char> rfkt::zlib::uncompress(const std::vector<char>& data)
{
    return uncompress(data.data(), data.size());
}

std::vector<char> rfkt::zlib::uncompress(const void* data, std::size_t len)
{
    auto inflated_size = static_cast<unsigned long>(((std::size_t*)data)[0]);
    std::vector<char> ret;
    ret.resize(inflated_size);
    ::uncompress((Bytef*) ret.data(), &inflated_size, (Bytef*) data + sizeof(len), len - sizeof(len));
    return ret;
}

bool rfkt::zlib::extract_zip(const rfkt::fs::path& zip_path, const rfkt::fs::path& out_path)
{
    if (!fs::exists(zip_path) || !fs::exists(out_path) || !fs::is_directory(out_path)) return false;

    int error = zip_extract(zip_path.string().c_str(), out_path.string().c_str(), nullptr, nullptr);

    return error == 0;
}
