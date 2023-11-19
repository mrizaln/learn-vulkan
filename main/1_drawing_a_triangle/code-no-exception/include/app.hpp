#ifndef APP_HPP_1BYX9BLT
#define APP_HPP_1BYX9BLT

#include <atomic>
#include <csignal>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "vulkan.hpp"
#include "scene.hpp"

class FpsCounter
{
private:
    double                      m_updateInterval;
    double                      m_currentTime{ 0.0 };
    double                      m_sumTime{ 0.0 };
    double                      m_avgTime{ 0.0 };
    int                         m_frames{ 0 };
    std::function<void(double)> m_callback;

public:
    // updateInterval: how often the average time is updated (in seconds)
    // callback: called when the average time is updated, nullptr if no callback
    // (check update() return value to see if the average time was updated)
    FpsCounter(double updateInterval = 1.0, std::function<void(double)>&& callback = nullptr)
        : m_updateInterval{ updateInterval }
        , m_callback{ std::move(callback) }
    {
    }

    // returns true if the average time was updated. average time is updated every updateInterval seconds
    bool update(double newTime)
    {
        double deltaTime  = newTime - m_currentTime;
        m_currentTime     = newTime;
        m_sumTime        += deltaTime;
        m_frames++;

        if (m_sumTime >= m_updateInterval) {
            m_avgTime = m_sumTime / m_frames;
            m_sumTime = 0.0;
            m_frames  = 0;
            if (m_callback) {
                m_callback(m_avgTime);
            }
            return true;
        }
        return false;
    }

    double getAvgTime() const { return m_avgTime; }
};

class App
{
private:
    static constexpr std::size_t s_windowHeight = 600;
    static constexpr std::size_t s_windowWidth  = 800;
    static constexpr std::string s_windowName   = "LearnVulkan";

    using unique_GLFWwindow = std::unique_ptr<GLFWwindow, decltype([](GLFWwindow* window) { glfwDestroyWindow(window); })>;

    inline static std::unique_ptr<App> s_instance{ nullptr };

    unique_GLFWwindow m_window;

    Vulkan m_vulkan;

    Scene             m_scene;
    std::atomic<bool> m_running{ true };

    FpsCounter m_fpsCounter;

public:
    static void init() noexcept(false)
    {
        glfwSetErrorCallback([](int error_code, const char* description) {
            std::cerr << std::format("ERROR: [GLFW] ({}) {}\n", error_code, description);
        });

        if (glfwInit() == 0) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        unique_GLFWwindow window{ glfwCreateWindow(s_windowWidth, s_windowHeight, s_windowName.c_str(), nullptr, nullptr) };
        if (!window) {
            throw std::runtime_error("Failed to create window");
        }

        Vulkan vulkan{ window.get(), s_windowName };    // might throw

        auto* appPtr = new App{ std::move(window), std::move(vulkan) };
        s_instance.reset(appPtr);
    }

    static void run() noexcept(false)
    {
        if (!s_instance) {
            throw std::runtime_error("App not initialized");
        }
        s_instance->run_impl();
    }

    static void deinit()
    {
        s_instance.reset();

        glfwTerminate();
    }

    App(const App&)            = delete;
    App(App&&)                 = delete;
    App& operator=(const App&) = delete;
    App& operator=(App&&)      = delete;

    ~App() = default;

private:
    App(unique_GLFWwindow&& window, Vulkan&& vulkan)
        : m_window{ std::move(window) }
        , m_vulkan{ std::move(vulkan) }
    {
        // signal
        std::signal(SIGINT, [](int) {
            std::cout << "Interrupt signal received\n";
            s_instance->m_running = false;
        });
        std::signal(SIGTERM, [](int) {
            std::cout << "Terminate signal received\n";
            s_instance->m_running = false;
        });

        // fps counter
        // (i'm doing it here, just to make sure the class is fully constructed before using it
        // and it's just looks nicer here than in initializer list :>)
        m_fpsCounter = {
            1.0,
            [&window = m_window](double time) {
                glfwSetWindowTitle(
                    window.get(),
                    std::format("{} - FPS: {:.2f}", s_windowName, 1.0 / time).c_str()
                );
            }
        };
    }

    void draw()
    {
        m_vulkan.drawFrame();
    }

    void run_impl()
    {
        while ((glfwWindowShouldClose(m_window.get()) == 0) && m_running) {
            glfwPollEvents();
            draw();
            m_fpsCounter.update(glfwGetTime());
        }
    }
};

#endif /* end of include guard: APP_HPP_1BYX9BLT */
