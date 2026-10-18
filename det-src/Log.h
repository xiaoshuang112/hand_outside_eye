#pragma once

#include <string>

namespace util::Log {

	void InitSpdLog(std::string path, int maxfilenum) noexcept;

	void trace(char const* const format, ...) noexcept;
	inline void trace(std::string const& content) noexcept {
		trace(content.c_str());
	}

	void debug(char const* const format, ...) noexcept;
	inline void debug(std::string const& content) noexcept {
		debug(content.c_str());
	}

	void info(char const* const format, ...) noexcept;
	inline void info(std::string const& content) noexcept {
		info(content.c_str());
	}

	void log(char const* const format, ...) noexcept;
	inline void log(std::string const& content) noexcept {
		log(content.c_str());
	}

	void log_cmd(char const* const format, ...) noexcept;
	inline void log_cmd(std::string const& content) noexcept {
		log_cmd(content.c_str());
	}

	void warn(char const* const format, ...) noexcept;
	inline void warn(std::string const& content) noexcept {
		warn(content.c_str());
	}

	void error(char const* const format, ...) noexcept;
	inline void error(std::string const& content) noexcept {
		error(content.c_str());
	}

	void critical(char const* const format, ...) noexcept;
	inline void critical(std::string const& content) noexcept {
		critical(content.c_str());
	}

	void CloseLog() noexcept;

} // namespace util::Log
