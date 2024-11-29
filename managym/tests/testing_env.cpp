// tests/global_test_setup.cpp

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// Define a global environment class
class GlobalTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // Create a console logger if it doesn't exist
        if (!spdlog::get("console")) {
            auto console = spdlog::stdout_color_mt("console");
            console->set_level(spdlog::level::debug); // Set log level to debug
            spdlog::set_default_logger(console);
            spdlog::set_level(spdlog::level::debug); // Global log level
            spdlog::flush_on(spdlog::level::debug); // Flush on debug
        }
    }
    
    void TearDown() override {
        spdlog::shutdown();
    }
};

// Register the environment
::testing::Environment* const global_env = 
    ::testing::AddGlobalTestEnvironment(new GlobalTestEnvironment());