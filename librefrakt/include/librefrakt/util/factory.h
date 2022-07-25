// Adapted from "Unforgettable Factory Registration" by Nir Friedman
// http://www.nirfriedman.com/2018/04/29/unforgettable-factory/

#pragma once
#include <unordered_map>
#include <memory>
#include <spdlog/spdlog.h>

template <class Base, class... Args> class factory {
public:
	template <class ... T>
	static std::unique_ptr<Base> make(const std::string& s, T&&... args) {
		return data().at(s)(std::forward<T>(args)...);
	}

	static bool exists(const std::string& s) {
		return data().contains(s);
	}

	static auto types() -> std::vector<std::string> {
		auto ret = std::vector<std::string>{};
		for (auto& [name, factory] : data()) ret.emplace_back(name);
		return ret;
	}

	template <class T> struct registrar : Base {
		friend T;

		static std::string demangle(const std::string& name) {
			auto namespace_pos = name.find_last_of("::");
			if (namespace_pos != std::string::npos) return name.substr(namespace_pos + 1);

			// TODO: Compiler independence, only works for MSVC right now
			for (const char* name_ptr = name.c_str(); *name_ptr != '\0'; name_ptr++)
				if (*name_ptr == ' ') return std::string(name_ptr + 1);

			return name;
		}

		static bool registerT() {
			auto name = demangle(typeid(T).name());
			auto b_name = demangle(typeid(Base).name());
			SPDLOG_INFO("Registering subclass '{}' for base '{}'", name, b_name);
			factory::data()[name] = [](Args... args) -> std::unique_ptr<Base> {
				return std::make_unique<T>(std::forward<Args>(args)...);
			};
			return true;
		}
		static bool registered;

	private:
		registrar() : Base(Key{}) { (void)registered; }
	};

	friend Base;

private:
	class Key {
		Key() {};
		template <class T> friend struct registrar;
	};
	using FuncType = std::unique_ptr<Base>(*)(Args...);
	factory() = default;

	static auto& data() {
		static std::unordered_map<std::string, FuncType> s;
		return s;
	}
};

template <class Base, class... Args>
template <class T>
bool factory<Base, Args...>::template registrar<T>::registered =
factory<Base, Args...>::template registrar<T>::registerT();