#pragma once

#include <string>
#include <string_view>
#include <fstream>
#include <filesystem>
#include <cctype>
// debug
#include <iostream>
#include <cstring>
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

namespace detail {
	namespace {
		constexpr int base = __COUNTER__+1;
	};

	inline std::string formGlslArg(const std::string& path, const int counter, const int version, const std::string& given_source) {
		auto index = std::to_string(counter - base), versionString = std::to_string(version), file = std::filesystem::path(path).relative_path().filename().string();
		return path + "," + index + "," + versionString + "\n#version " + versionString + "\n" + given_source;
	}
}

#define GLSL(version, ...) detail::formGlslArg(__FILENAME__, __COUNTER__, version, #__VA_ARGS__)
