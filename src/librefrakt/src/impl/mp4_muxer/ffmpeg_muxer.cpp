#include <reproc++/reproc.hpp>

#include <librefrakt/interface/mp4_muxer.h>
#include <librefrakt/util/filesystem.h>

namespace rfkt {

	struct ffmpeg_muxer : mp4_muxer::registrar<ffmpeg_muxer> {
		explicit ffmpeg_muxer(std::string path, int fps)
			: path(std::move(path)), fps(fps) {
		
			std::vector<std::string> args = {
				std::format("{}\\bin\\ffmpeg.exe", rfkt::fs::working_directory().string()),
				"-hide_banner", "-loglevel", "error",
				"-y","-i", "-",
				"-c", "copy",
				"-r", std::to_string(fps),
				this->path
			};

			process.start(args);

		}

		void write_chunk(std::span<const char> chunk) override {
			process.write((const uint8_t*) chunk.data(), chunk.size());
		}

		void finish() override {
			process.close(reproc::stream::in);
			process.wait(reproc::milliseconds(1000));
		}

		std::string path;
		int fps;
		reproc::process process;
	};

}