#include <future>

#include <librefrakt/flame_compiler.h>
#include <librefrakt/anima.h>

#include "gl.h"
#include "command_executor.h"

class preview_panel {
public:

	using executor_t = std::function<void(std::move_only_function<void(void)>&&)>;
	using texture_format = rfkt::gl::texture<rfkt::gl::texture_format::rgba8>;
	using pixel_type = texture_format::traits::pixel_type;

	using renderer_t = std::function<rfkt::gpu_image<pixel_type>(roccu::gpu_stream&, const rfkt::flame_kernel&, rfkt::flame_kernel::saved_state&, rfkt::flame_kernel::bailout_args, double3, bool, bool)>;

	preview_panel() = delete;
	preview_panel(rfkt::flame_compiler& compiler, executor_t&& submitter, renderer_t&& renderer, command_executor& cmd_exec) :
		submitter(std::move(submitter)),
		renderer(std::move(renderer)),
		compiler(compiler),
		cmd_exec(cmd_exec) {}

	~preview_panel();

	bool show(const rfkt::flamedb& fdb, rfkt::flame& flame, rfkt::function_table& ft);

private:

	using texture_t = texture_format::handle;

	bool render_is_ready() {
		return rendering_texture.has_value() && rendering_texture->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	uint2 gui_logic(rfkt::flame& flame, rfkt::function_table& ft);

	std::shared_ptr<rfkt::flame_kernel::saved_state> current_state = std::make_shared<rfkt::flame_kernel::saved_state>();
	std::shared_ptr<rfkt::flame_kernel> kernel = std::make_shared<rfkt::flame_kernel>();

	std::optional<texture_t> displayed_texture = std::nullopt;
	std::optional<std::future<texture_t>> rendering_texture = std::nullopt;
	//std::optional<rfkt::gl::texture::cuda_map> cuda_map = std::nullopt;

	executor_t submitter;
	renderer_t renderer;
	roccu::gpu_stream stream;
	rfkt::flame_compiler& compiler;
	command_executor& cmd_exec;

	rfkt::hash_t flame_structure_hash = {};
	rfkt::hash_t flame_value_hash = {};
	rfkt::hash_t flamedb_hash = {};

	uint2 render_dims = { 0, 0 };
	double target_quality = 128;

	double aspect_ratio = 0.0;
	bool render_options_changed = false;
	bool upscale = true;
	bool denoise = true;
	bool animate = true;
	bool dragging = false;
	bool playing = false;
	double current_time = 0.0;
	ImVec2 last_delta = { 0, 0 };
	double2 drag_start = { 0, 0 };

	std::optional<double> scroll_scale_start;
	std::chrono::steady_clock::time_point scroll_start_time = std::chrono::steady_clock::now();
	const std::chrono::milliseconds scroll_timeout = std::chrono::milliseconds(100);
};