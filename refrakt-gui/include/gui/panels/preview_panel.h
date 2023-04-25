#include <future>

#include <librefrakt/flame_compiler.h>

#include "gui.h"
#include "gl.h"

class preview_panel {
public:

	using executor_t = std::function<void(std::move_only_function<void(void)&&>)>;

	bool show(rfkt::flame& flame);

private:

	static rfkt::hash_t get_value_hash(const rfkt::flame& flame);
	bool render_is_ready() {
		return rendering_texture.has_value() && rendering_texture->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	std::shared_ptr<rfkt::flame_kernel::saved_state> current_state;
	std::shared_ptr<rfkt::flame_kernel> kernel;

	std::optional<rfkt::gl::texture> displayed_texture;
	std::optional<std::future<rfkt::gl::texture>> rendering_texture;

	executor_t submitter;
	rfkt::cuda_stream& stream;

	rfkt::hash_t flame_structure_hash;
	rfkt::hash_t flame_value_hash;

	uint2 render_dims = { 0, 0 };
	double target_quality = 2000;
};