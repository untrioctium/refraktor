#include <App.h>
#include <fmt/format.h>

#include <librefrakt/flame_compiler.h>

int main(int argc, char** argv) {

	rfkt::flame_info::initialize("config/variations.yml");
	rfkt::cuda::init();

	auto app = uWS::SSLApp();

#define WS_HANDLER() (uWS::HttpResponse<true>* res, uWS::HttpRequest* req)


	app.get("/", [] WS_HANDLER() {
		res->writeStatus("200 OK");
		res->end();
	});

	app.listen(3000,
		[](auto* listen_socket) {
			if (listen_socket) {
				fmt::print("Listening on port 3000");
			}
		});

	app.run();

	return 1;
}