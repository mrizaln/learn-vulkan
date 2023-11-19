#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>

#define VULKAN_HPP_NO_EXCEPTIONS      // enable return value traansformations
#define VULKAN_HPP_NO_CONSTRUCTORS    // enable designated initializers
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

std::atomic<bool> g_terminate = false;

void signal_handler(int signal)
{
    auto signal_name = [&]() {
        switch (signal) {
        case SIGINT: return "SIGINT";
        case SIGTERM: return "SIGTERM";
        default: return "UNKNOWN";
        }
    }();

    std::cout << "Signal: " << signal_name << " received\n";
    g_terminate = true;
}

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnVulkan", nullptr, nullptr);

    uint32_t     glfwExtensionCount{ 0 };
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto instance = vk::createInstanceUnique(
        vk::InstanceCreateInfo{
            .enabledExtensionCount   = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions,
        },
        nullptr
    );

    auto [result, extensionCount] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        std::cerr << "failed to enumerate extensions\n";
        return 1;
    }
    std::cout << extensionCount.size() << " extensions supported\n";

    auto matrix = glm::mat4{ 1.0f };
    auto vec    = glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f };
    auto test   = matrix * vec;

    using namespace std::chrono_literals;
    const auto wait_duration = 3s;
    const auto start_time    = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window) && !g_terminate) {
        if (std::chrono::steady_clock::now() - start_time > wait_duration) {
            std::cout << "Waited for " << wait_duration << '\n';
            break;
        }
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "All OK!\n";
}
