#include <future>

#include <librefrakt/flame_compiler.h>

#include "gui.h"
#include "gl.h"

class preview_panel {
public:

	using executor_t = std::function<void(std::move_only_function<void(void)>&&)>;
	using renderer_t = std::function<rfkt::cuda_buffer<uchar4>(rfkt::cuda_stream&, const rfkt::flame_kernel&, rfkt::flame_kernel::saved_state&, rfkt::flame_kernel::bailout_args, double3)>;

	preview_panel() = delete;
	preview_panel(rfkt::cuda_stream& stream, rfkt::flame_compiler& compiler, executor_t& submitter, renderer_t& renderer) :
		submitter(submitter),
		renderer(renderer),
		stream(stream),
		compiler(compiler) {}

	~preview_panel();

	bool show(const rfkt::flamedb& fdb, rfkt::flame& flame);

private:

	static rfkt::hash_t get_value_hash(const rfkt::flame& flame);
	bool render_is_ready() {
		return rendering_texture.has_value() && rendering_texture->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	uint2 gui_logic(rfkt::flame& flame);

	std::shared_ptr<rfkt::flame_kernel::saved_state> current_state = std::make_shared<rfkt::flame_kernel::saved_state>();
	std::shared_ptr<rfkt::flame_kernel> kernel = std::make_shared<rfkt::flame_kernel>();

	std::optional<rfkt::gl::texture> displayed_texture = std::nullopt;
	std::optional<std::future<rfkt::gl::texture>> rendering_texture = std::nullopt;
	std::optional<rfkt::gl::texture::cuda_map> cuda_map = std::nullopt;

	executor_t& submitter;
	renderer_t& renderer;
	rfkt::cuda_stream& stream;
	rfkt::flame_compiler& compiler;

	rfkt::hash_t flame_structure_hash = {};
	rfkt::hash_t flame_value_hash = {};
	rfkt::hash_t flamedb_hash = {};

	uint2 render_dims = { 0, 0 };
	double target_quality = 128;

	bool dragging = false;
	ImVec2 last_delta = { 0, 0 };
};