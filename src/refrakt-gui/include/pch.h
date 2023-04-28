#pragma once

// real basic stuff
#include <memory>
#include <string>
#include <source_location>

// containers
#include <vector>
#include <set>
#include <map>
#include <array>

#include <filesystem>
#include <chrono>
#include <optional>
#include <algorithm>

#include <format>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <librefrakt/util.h>
#include <nlohmann/json.hpp>