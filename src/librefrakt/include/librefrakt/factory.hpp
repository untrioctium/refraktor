#pragma once

#include <source_location>
#include <string_view>
#include <unordered_map>
#include <ranges>
#include <memory>

namespace rfkt {

	namespace detail {

		struct demangler {
		private:
			template<typename T>
			struct $ {};

			template<typename T>
			static consteval auto demangle_impl() {
				auto sv = std::string_view{ std::source_location::current().function_name() };
				auto start = sv.find_last_of('$') + 2;
				auto end = sv.find_last_of(">") - 1;

				sv = sv.substr(start, end - start);
				if (sv.starts_with("class ")) sv = sv.substr(6);
				else if (sv.starts_with("struct ")) sv = sv.substr(7);
				else if (sv.starts_with("enum ")) sv = sv.substr(5);
				else if (sv.starts_with("enum class")) sv = sv.substr(11);

				return sv;
			}

		public:
			template<typename T>
			static consteval auto demangle() {
				return demangle_impl<$<T>>();
			}
		};

		template<typename T, typename... Args>
		concept has_static_make = requires(Args ...args) { { T::make(args...) } -> std::same_as<std::unique_ptr<T>>; };

		template<typename T>
		concept has_meta_type = requires { typename T::meta_type; };

	}

	template<class Base, class... Args>
	class factory {
	public:

		static std::unique_ptr<Base> make(std::string_view name, Args... args) {
			auto it = factories().find(name);
			if (it == factories().end()) return nullptr;

			auto ptr = [name, &it](auto&&... args_inner) {
				if constexpr (detail::has_meta_type<Base>) {
					auto ptr = it->second.first(std::forward<decltype(args_inner)>(args_inner)...);
					ptr->meta_ptr = (const void*) it->second.second;
					return ptr;
				}
				else
					return it->second(std::forward<decltype(args_inner)>(args_inner)...);
			}(std::forward<Args>(args)...);

			ptr->name_ = name;

			return ptr;
		}

		template<typename T>
		struct registrar : Base {
			friend T;

			static bool register_factory() {
				constexpr static auto name = detail::demangler::demangle<T>();
				constexpr static auto func =
					[](Args... args) -> std::unique_ptr<Base> {
					if constexpr (detail::has_static_make<T, Args...>)
						return T::make(std::forward<Args>(args)...);
					else
						return std::make_unique<T>(std::forward<Args>(args)...);
					};

				if constexpr (detail::has_meta_type<Base>) {
					factories()[name] = { func, &T::meta };
				}
				else {
					factories()[name] = func;
				}

				return true;
			}

		private:
			const inline static bool registered = register_factory();
			registrar() : Base(key{}) { (void)registered; }
		};

		static auto names() {
			return std::views::keys(factories());
		}

		static const auto* meta_for(std::string_view name)
			requires detail::has_meta_type<Base>
		{
			auto it = factories().find(name);
			if (it == factories().end()) return (typename Base::meta_type*) nullptr;
			return it->second.second;
		}

		std::string_view name() const {
			return name_;
		}

		const auto& meta() const
			requires detail::has_meta_type<Base>
		{
			return *static_cast<const typename Base::meta_type*>(meta_ptr);
		}

		friend Base;

	private:

		std::string_view name_;
		const void* meta_ptr;

		class key {
			key() {};
			template <class T> friend struct registrar;
		};

		using factory_function = std::add_pointer_t<std::unique_ptr<Base>(Args...)>;
		factory() = default;

		static auto& factories() {
			if constexpr (detail::has_meta_type<Base>) {
				static std::unordered_map<std::string_view, std::pair<factory_function, const typename Base::meta_type*>> factories_{};
				return factories_;
			}
			else {
				static std::unordered_map<std::string_view, factory_function> factories_;
				return factories_;
			}
		}
	};
}