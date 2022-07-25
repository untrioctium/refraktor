#include <cooperative_groups.h>
using namespace cooperative_groups;

#include <refrakt/flamelib.h>

__device__ void flamelib::sync_grid() {
	this_grid().sync();
}