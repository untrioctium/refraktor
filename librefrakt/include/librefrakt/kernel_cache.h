#include <SQLiteCpp/SQLiteCpp.h>
#include <mutex>

#include <refrakt/util/string.h>
#include <refrakt/util/filesystem.h>
#include <refrakt/util/zlib.h>
#include <refrakt/traits/hashable.h>

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

	kernel_cache(): db("cache/kernel.sqlite", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE) {
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

	static SQLite::Database& thread_local_handle() {
		thread_local auto db = 
		[]() 
		{
			auto db = SQLite::Database{ "cache/kernel.sqlite", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE };

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
			return db;
		}();

		return db;
	}

	bool is_available(const std::string& id, const rfkt::hash_t& opts_hash) {
		//std::lock_guard<std::mutex> lock(mtx);

		auto query = SQLite::Statement(db, "SELECT dependencies,create_ts,opts_hash FROM cache WHERE id = ?");
		query.bind(1, id);
		if (!query.executeStep()) return false;

		auto depends = str_util::split(query.getColumn(0), ';');
		long long create_ts = query.getColumn(1);
		std::string stored_hash = query.getColumn(2);

		if (stored_hash != hash::to_string(opts_hash)) {
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
		query.bind(2, hash::to_string(hash));
		query.bind(3, fs::now());
		query.bind(4, str_util::join(r.dependencies, ';'));
		query.bind(5, str_util::join(r.functions, '=', ';'));
		query.bind(6, str_util::join(r.globals, '=', ';'));
		query.bind(7, r.compile_ms);
		query.bind(8, (unsigned int) r.cubin.size());
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