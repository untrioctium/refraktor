#include <curl/curl.h>

#include <librefrakt/util/http.h>

void check_curl_init() {
	struct curl_init_wrapper {
		curl_init_wrapper() {
			curl_global_init(CURL_GLOBAL_ALL);
		}
		~curl_init_wrapper() {
			curl_global_cleanup();
		}
	};

	static curl_init_wrapper init;
}

bool rfkt::http::download(std::string_view url, const rfkt::fs::path& path)
{
	check_curl_init();

	std::unique_ptr<FILE, decltype(&fclose)> of(fopen(path.string().c_str(), "wb"), &fclose);

	if (!of) {
		return false;
	}

	CURL* curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_URL, url.data());
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, of.get());

	return curl_easy_perform(curl) == CURLE_OK;
}

auto rfkt::http::download(std::string_view url) -> std::optional<std::vector<char>> 
{
	check_curl_init();

	std::vector<char> data;

	CURL* curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_URL, url.data());
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char* ptr, size_t size, size_t nmemb, void* userdata) {
		auto& data = *static_cast<std::vector<char>*>(userdata);
		data.insert(data.end(), ptr, ptr + size * nmemb);
		return size * nmemb;
	});
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

	if (curl_easy_perform(curl) == CURLE_OK) {
		return data;
	}

	return std::nullopt;
}

auto rfkt::http::head(std::string_view url) -> std::optional<std::map<std::string, std::string>>
{
	check_curl_init();

	std::map<std::string, std::string> headers;

	CURL* curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_URL, url.data());
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, +[](char* ptr, size_t size, size_t nmemb, void* userdata) {
		auto& headers = *static_cast<std::map<std::string, std::string>*>(userdata);
		std::string_view line(ptr, size * nmemb - 2);
		if (auto colon = line.find(':'); colon != std::string_view::npos) {
			headers.emplace(line.substr(0, colon), line.substr(colon + 2));
		}
		return size * nmemb;
	});
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);
	curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

	if (curl_easy_perform(curl) == CURLE_OK) {
		return headers;
	}

	return std::nullopt;
}
