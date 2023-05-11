#include <future>

#include <librefrakt/flame_compiler.h>
#include <librefrakt/anima.h>

#include "gl.h"

class preview_panel {
public:

	using command_executor_t = std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)>;
	using executor_t = std::function<void(std::move_only_function<void(void)>&&)>;
	using renderer_t = std::function<rfkt::cuda_buffer<uchar4>(rfkt::cuda_stream&, const rfkt::flame_kernel&, rfkt::flame_kernel::saved_state&, rfkt::flame_kernel::bailout_args, double3, bool)>;

	preview_panel() = delete;
	preview_panel(rfkt::cuda_stream& stream, rfkt::flame_compiler& compiler, executor_t& submitter, renderer_t& renderer, command_executor_t& cmd_exec) :
		submitter(submitter),
		renderer(renderer),
		stream(stream),
		compiler(compiler),
		cmd_exec(cmd_exec) {}

	~preview_panel();

	bool show(const rfkt::flamedb& fdb, rfkt::flame& flame, rfkt::function_table& ft);

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
	command_executor_t& cmd_exec;

	rfkt::hash_t flame_structure_hash = {};
	rfkt::hash_t flame_value_hash = {};
	rfkt::hash_t flamedb_hash = {};

	uint2 render_dims = { 0, 0 };
	double target_quality = 128;

	bool render_options_changed = false;
	bool upscale = false;
	bool dragging = false;
	double current_time = 0.0;
	ImVec2 last_delta = { 0, 0 };
	double2 drag_start = { 0, 0 };

	std::optional<double> scroll_scale_start;
	std::chrono::steady_clock::time_point scroll_start_time = std::chrono::steady_clock::now();
	const std::chrono::milliseconds scroll_timeout = std::chrono::milliseconds(100);
};