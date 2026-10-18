#include "Log.h"

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include <cstdarg>
#include <iostream>
#include <memory>

namespace util::Log {
    std::shared_ptr<spdlog::logger> logger = nullptr;
    std::shared_ptr<spdlog::logger> logger_cmd = nullptr;

    void InitSpdLog(std::string path, int maxfilenum) noexcept {
        std::vector<spdlog::sink_ptr> sinks;

        // cmd输出
#if WIN32
        auto sink1 = std::make_shared<spdlog::sinks::wincolor_stdout_sink_mt>();
#else
        auto sink1 = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
#endif

        /*daily日志输出
         * param0: Log路径
         * param1: 时
         * param2: 分 param1 and param2 代表 几时几分更新文件
         * param3: 是否trunc打开文件
         * param4: MaxFilesNum
         */
        auto sink2 = std::make_shared<spdlog::sinks::daily_file_sink_mt>(path, 0, 0, false, maxfilenum);

        /*rotating日志输出
         * param0: Log路径
         * param1: MaxSize
         * param2: MaxFilesNum
         */
        // auto sink3 = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("Logs/RobotLog.txt", 4 * 1024 * 1024 * 1024, 3);

        sinks.push_back(sink1);
        sinks.push_back(sink2);

        logger = std::make_shared<spdlog::logger>("MyLog", sinks.begin(), sinks.end());
        logger_cmd = std::make_shared<spdlog::logger>("CmdLog", sink1);

        // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e][thd %t][%^%L%$]:%v");
        logger_cmd->set_pattern("[%Y-%m-%d %H:%M:%S.%e][thd %t][%^%L%$]:%v");

        // logger->set_level(spdlog::level::trace);
        
        logger->flush_on(spdlog::level::info);
        logger_cmd->flush_on(spdlog::level::info);
    }

    void InitTaskLog(std::shared_ptr<spdlog::logger>& logger_task, std::string task_id, std::string path, int maxfilenum) noexcept {
        std::vector<spdlog::sink_ptr> sinks;


        /*rotating日志输出
         * param0: Log路径
         * param1: MaxSize
         * param2: MaxFilesNum
         */
         auto sink3 = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path, 5 * 1024 * 1024, maxfilenum);

        sinks.push_back(sink3);

        logger_task = std::make_shared<spdlog::logger>(task_id, sinks.begin(), sinks.end());

        // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
        logger_task->set_pattern("[%Y-%m-%d %H:%M:%S.%e][thd %t][%^%L%$]:%v");

        // logger_task->set_level(spdlog::level::trace);

        logger_task->flush_on(spdlog::level::info);
    }


    void trace(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        if (nullptr==logger)
        {
            std::cout << buffer << std::endl;
            return;
        }
        

        logger->trace("{}", buffer);
    }

    void debug(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);
        if (nullptr==logger)
        {
            std::cout << buffer << std::endl;
            return;
        }
        logger->debug("{}", buffer);
    }

    void info(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);
        if (nullptr==logger)
        {
            std::cout << buffer << std::endl;
            return;
        }        
        logger->info("{}", buffer);
    }

    void log(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        if (nullptr==logger)
        {
            std::cout << buffer << std::endl;
            return;
        }        
        logger->info("{}", buffer);
    }

    void log_cmd(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);
        if (nullptr==logger_cmd)
        {
            std::cout << buffer << std::endl;
            return;
        }              
        logger_cmd->info("{}", buffer);
    }

    void log_task(std::shared_ptr<spdlog::logger>& logger_task, char const* const format, ...) noexcept {
        if (nullptr == logger_task) return;

        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{ 0 };
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        try
        {
            logger_task->info("{}", buffer);
        }
        catch (...) {}
        
    }
    void error_task(std::shared_ptr<spdlog::logger>& logger_task, char const* const format, ...) noexcept {
        if (nullptr == logger_task) return;

        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        try
        {
            logger_task->error("{}", buffer);
        }
        catch (...) {}
        
    }

    void warn(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);
        if (nullptr==logger_cmd)
        {
            std::cerr << buffer << std::endl;
            return;
        }             
        logger->warn("{}", buffer);
    }

    void error(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        if (nullptr==logger_cmd)
        {
            std::cerr << buffer << std::endl;
            return;
        }     

        logger->error("{}", buffer);
    }

    void critical(char const* const format, ...) noexcept {
        static constexpr const size_t BUFFER_LENGTH = 65536;
        static char buffer[BUFFER_LENGTH]{0};
        std::va_list args;
        va_start(args, format);
        auto contentLength = std::vsprintf(buffer, format, args);
        va_end(args);

        if (nullptr==logger_cmd)
        {
            std::cerr << buffer << std::endl;
            return;
        }     

        logger->critical("{}", buffer);
    }

    void CloseLog() noexcept {
        // 不要 spdlog::drop_all()：GNU unique 符号会把这边的 registry
        // 和宿主进程里的 libspdlog.so 并成一份，drop_all 会把宿主日志一起拆掉。
        logger.reset();
        logger_cmd.reset();
    }

} // namespace util::Log
