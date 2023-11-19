#ifndef VULKAN_HPP_GVYQUUQ0
#define VULKAN_HPP_GVYQUUQ0

#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(NDEBUG)
#    define ENABLE_VULKAN_VALIDATION_LAYERS 1
#else
#    define ENABLE_VULKAN_VALIDATION_LAYERS 0
#endif

class Vulkan
{
private:
    static constexpr bool        s_enableValidation  = ENABLE_VULKAN_VALIDATION_LAYERS;
    static constexpr std::array  s_validationLayers  = { "VK_LAYER_KHRONOS_validation" };
    static constexpr std::array  s_deviceExtensions  = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    static constexpr std::size_t s_maxFramesInFlight = 2;    // should not be zero, of course

    // NOTE: the number of members might change in the future
    struct QueueFamilyIndices
    {
    public:
        uint32_t m_graphicsFamily;
        uint32_t m_presentFamily;
        /* ... */

    private:
        struct OptionalQueueFamilyIndices
        {
            std::optional<uint32_t> m_graphicsFamily;
            std::optional<uint32_t> m_presentFamily;

            bool isComplete() const
            {
                return (
                    m_graphicsFamily.has_value()
                    && m_presentFamily.has_value()
                    /* ... */
                );
            }
        };

    public:
        static std::optional<QueueFamilyIndices> getCompleteQueueFamilies(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
        {
            OptionalQueueFamilyIndices indices;

            std::vector queueFamilies{ device.getQueueFamilyProperties() };
            for (uint32_t index{ 0 }; const auto& queueFamiliy : queueFamilies) {
                if (queueFamiliy.queueFlags & vk::QueueFlagBits::eGraphics) {
                    indices.m_graphicsFamily = index;
                }
                auto [result, value] = device.getSurfaceSupportKHR(index, surface);
                if (result == vk::Result::eSuccess && value == VK_TRUE) {
                    indices.m_presentFamily = index;
                } else {
                    return std::nullopt;
                }

                if (indices.isComplete()) {
                    break;
                }
                ++index;
            }
            if (!indices.isComplete()) {
                return std::nullopt;
            }

            return QueueFamilyIndices{
                .m_graphicsFamily = indices.m_graphicsFamily.value(),
                .m_presentFamily  = indices.m_presentFamily.value(),
                /* ... */
            };
        }

        static bool checkCompleteness(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
        {
            OptionalQueueFamilyIndices indices;

            std::vector queueFamilies{ device.getQueueFamilyProperties() };
            for (uint32_t index{ 0 }; const auto& queueFamiliy : queueFamilies) {
                if (queueFamiliy.queueFlags & vk::QueueFlagBits::eGraphics) {
                    indices.m_graphicsFamily = index;
                }
                // TODO: don't ignore the unsuccesful result
                auto [result, value] = device.getSurfaceSupportKHR(index, surface);
                if (result == vk::Result::eSuccess && value == VK_TRUE) {
                    indices.m_presentFamily = index;
                } else {
                    return false;
                }

                if (indices.isComplete()) {
                    break;
                }
                ++index;
            }
            return indices.isComplete();
        }

        std::set<uint32_t> getUniqueIndices() const
        {
            return {
                m_graphicsFamily,
                m_presentFamily,
                /* ... */
            };
        }

        auto asArray() const
        {
            return std::array{
                m_graphicsFamily,
                m_presentFamily,
                /* ... */
            };
        }
    };

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR        m_capabilities;
        std::vector<vk::SurfaceFormatKHR> m_formats;
        std::vector<vk::PresentModeKHR>   m_presentModes;

        bool isAdequate() const
        {
            return !m_formats.empty() && !m_presentModes.empty();
        }

        vk::SurfaceFormatKHR chooseSurfaceFormat() const
        {
            // prefer srgb color space with 8-bit rgba
            for (const auto& format : m_formats) {
                if (
                    format.format == vk::Format::eB8G8R8A8Srgb
                    && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
                ) {
                    return format;
                }
            }

            // otherwise, just choose the first one
            return m_formats.front();
        }

        vk::PresentModeKHR choosePresentMode() const
        {
            // possible modes:
            // - vk::PresentModeKHR::eImmediate     : image presented right away
            // - vk::PresentModeKHR::eFifo          : image presented when screen refreshes
            // - vk::PresentModeKHR::eFifoRelaxed   : same as above, but if queue is empty, image is presented right away
            // - vk::PresentModeKHR::eMailbox       : same as above, but if queue is full, previous image is replaced
            // only fifo is guaranteed to be available

            // prefer mailbox mode (aka triple buffering)
            for (const auto& mode : m_presentModes) {
                if (mode == vk::PresentModeKHR::eMailbox) {
                    return mode;
                }
            }

            // otherwise, just choose fifo
            return vk::PresentModeKHR::eFifo;
        }

        vk::Extent2D chooseExtent(GLFWwindow* const window) const
        {
            // if the extent is max uint32_t, then we need to set it by ourselves
            // NOTE: the extent is the resolution of the swap chain images
            // NOTE: the extent is in pixels, not in screen coordinates
            // NOTE: the extent must be in the range [minExtent, maxExtent]

            // if the extent is already set, then just return it
            if (m_capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return m_capabilities.currentExtent;
            }

            // otherwise, set it manually
            int width{};
            int height{};
            glfwGetFramebufferSize(window, &width, &height);

            return {
                .width  = std::clamp(static_cast<uint32_t>(width), m_capabilities.minImageExtent.width, m_capabilities.maxImageExtent.width),
                .height = std::clamp(static_cast<uint32_t>(height), m_capabilities.minImageExtent.height, m_capabilities.maxImageExtent.height),
            };
        }
    };

    class DebugMessenger
    {
    public:
        vk::Instance               m_instance{ nullptr };
        vk::DebugUtilsMessengerEXT m_debugMessenger;

    public:
        // NOTE: I probably should not use vk::ResultValue here, but I'm too lazy to write my own
        static vk::ResultValue<DebugMessenger> create(vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
        {
            DebugMessenger debugMessenger;
            auto           dynamicDispatch = vk::DispatchLoaderDynamic{ instance, vkGetInstanceProcAddr };

            // load the function pointer for debug messenger creation and destruction using the dynamic dispatch
            auto [result, debugMessengerUnderlying] = instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dynamicDispatch);
            if (result == vk::Result::eSuccess) {
                debugMessenger.m_instance       = instance;
                debugMessenger.m_debugMessenger = debugMessengerUnderlying;
            }

            return { result, std::move(debugMessenger) };
        }

    public:
        DebugMessenger()                                 = default;
        DebugMessenger(const DebugMessenger&)            = delete;
        DebugMessenger& operator=(const DebugMessenger&) = delete;

        DebugMessenger& operator=(DebugMessenger&& other) noexcept
        {
            std::swap(m_instance, other.m_instance);
            std::swap(m_debugMessenger, other.m_debugMessenger);
            return *this;
        }

        DebugMessenger(DebugMessenger&& other) noexcept
            : DebugMessenger()
        {
            std::swap(m_instance, other.m_instance);
            std::swap(m_debugMessenger, other.m_debugMessenger);
        }

        ~DebugMessenger()
        {
            if (!m_instance) {
                return;
            }

            auto dynamicDispatch = vk::DispatchLoaderDynamic{ m_instance, vkGetInstanceProcAddr };
            m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger, nullptr, dynamicDispatch);
        }
    };

    struct SyncObject
    {
        vk::UniqueSemaphore m_imageAvailableSemaphore;
        vk::UniqueSemaphore m_renderFinishedSemaphore;
        vk::UniqueFence     m_inFlightFence;
    };

public:
    vk::UniqueInstance   m_instance;
    vk::UniqueSurfaceKHR m_surface;
    DebugMessenger       m_debugMessenger;

    vk::PhysicalDevice m_physicalDevice;    // implicitly destroyed when instance is destroyed
    vk::UniqueDevice   m_device;

    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;

    vk::UniqueSwapchainKHR             m_swapChain;
    std::vector<vk::Image>             m_swapChainImages;
    vk::Format                         m_swapChainImageFormat;
    vk::Extent2D                       m_swapChainExtent;
    std::vector<vk::UniqueImageView>   m_swapChainImageViews;
    std::vector<vk::UniqueFramebuffer> m_swapChainFramebuffers;

    vk::UniqueRenderPass     m_renderPass;
    vk::UniquePipelineLayout m_pipelineLayout;
    vk::UniquePipeline       m_graphicsPipeline;

    vk::UniqueCommandPool                m_commandPool;
    std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
    std::vector<SyncObject>              m_syncs;

    uint32_t m_currentFrameIndex{ 0 };

public:
    Vulkan(Vulkan&&)                       = default;
    Vulkan& operator=(Vulkan&&)            = default;
    Vulkan(const Vulkan&)                  = delete;
    Vulkan& operator=(const Vulkan& other) = delete;

    Vulkan(GLFWwindow* const window, const std::string& name)
    {
        std::vector extensions{ getExtensions() };
        if constexpr (s_enableValidation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // setup vulkan instance
        m_instance = createVulkanInstance(name, extensions);

        // setup debug messenger if validation layers are enabled
        if constexpr (s_enableValidation) {
            auto debugCreateInfo          = getDebugMessengerCreateInfo();
            auto [result, debugMessenger] = DebugMessenger::create(*m_instance, debugCreateInfo);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error(std::format("Failed to set up debug messenger: {}", vk::to_string(result)));
            }
            m_debugMessenger = std::move(debugMessenger);
        }

        // setup window surface
        m_surface = createSurface(window, *m_instance);

        // configure vulkan physical device
        m_physicalDevice = pickPhysicalDevice(*m_instance, *m_surface);
        std::cout << "INFO: [Vulkan] Using physical device: " << m_physicalDevice.getProperties().deviceName << '\n';

        // configure vulkan logical device and queues
        auto maybeQueueIndices{ QueueFamilyIndices::getCompleteQueueFamilies(m_physicalDevice, *m_surface) };
        if (!maybeQueueIndices) {
            throw std::runtime_error("Queue family not complete for selected device (This should not happen!)");
        }
        QueueFamilyIndices queueFamilies{ maybeQueueIndices.value() };
        m_device = createLogicalDevice(m_physicalDevice, queueFamilies);

        m_graphicsQueue = m_device->getQueue(queueFamilies.m_graphicsFamily, 0);
        m_presentQueue  = m_device->getQueue(queueFamilies.m_presentFamily, 0);

        // setup swap chain
        std::tie(m_swapChain, m_swapChainImageFormat, m_swapChainExtent) = createSwapChain(
            m_physicalDevice, *m_device, *m_surface, queueFamilies, window
        );
        auto maybeSwapchainImages = m_device->getSwapchainImagesKHR(*m_swapChain);
        if (maybeSwapchainImages.result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to get swap chain images: {}", vk::to_string(maybeSwapchainImages.result)));
        }
        m_swapChainImages     = maybeSwapchainImages.value;
        m_swapChainImageViews = createImageViews(*m_device, m_swapChainImages, m_swapChainImageFormat);

        // create graphics pipeline
        m_renderPass                                   = createRenderPass(*m_device, m_swapChainImageFormat);
        std::tie(m_pipelineLayout, m_graphicsPipeline) = createPipelineLayout(*m_device, *m_renderPass);

        // create framebuffers
        m_swapChainFramebuffers = createFramebuffers(*m_device, *m_renderPass, m_swapChainImageViews, m_swapChainExtent);

        m_commandPool    = createCommandPool(*m_device, queueFamilies);
        m_commandBuffers = createCommandBuffers(*m_device, *m_commandPool);

        // create sync objects
        m_syncs = createSyncObjects(*m_device);
    }

    ~Vulkan()
    {
        if (!m_instance) {
            // class instance is moved
            return;
        }

        // NOLINTNEXTLINE: the only failure that may happen is when a catastrophic failure occurs
        auto _ = m_device->waitIdle();
    }

private:
    Vulkan() = default;

    static std::vector<const char*> getExtensions()
    {
        uint32_t     glfwExtensionCount{ 0 };
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        return { glfwExtensions, glfwExtensions + glfwExtensionCount };
    }

    // NOTE: for some reason, I can't use the c++ bindings for this callback
    // (can't be assigned to vk::DebugUtilsMessengerCreateInfoEXT)
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
        VkDebugUtilsMessageTypeFlagsEXT             type [[maybe_unused]],
        const VkDebugUtilsMessengerCallbackDataEXT* pData,
        void*                                       pUserData [[maybe_unused]]
    )
    {
        using Severity               = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        constexpr auto to_underlying = [](Severity flag) { return static_cast<std::underlying_type_t<Severity>>(flag); };
        if (severity >= to_underlying(Severity::eVerbose)) {
            std::cerr << "[VKDEBUG] >> " << pData->pMessage << '\n';
        }
        return VK_FALSE;
    }

    static vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo()
    {
        using s = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using t = vk::DebugUtilsMessageTypeFlagBitsEXT;
        return {
            .messageSeverity = s::eVerbose | s::eWarning | s::eError,    // | s::eInfo,
            .messageType     = t::eGeneral | t::eValidation | t::ePerformance,
            .pfnUserCallback = debugCallback,
            .pUserData       = nullptr,
        };
    }

    static vk::UniqueInstance createVulkanInstance(const std::string& name, std::vector<const char*> extensions) noexcept(false)
    {
        vk::ApplicationInfo appInfo{
            .pApplicationName   = name.c_str(),
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = 0,    // disable validation layer for now
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;

        // enable debug utils extension
        if constexpr (s_enableValidation) {
            if (!checkValidationLayerSupport()) {
                throw std::runtime_error("Validation layers requested, but not available");
            }
            createInfo.enabledLayerCount   = s_validationLayers.size();
            createInfo.ppEnabledLayerNames = s_validationLayers.data();

            debugCreateInfo  = getDebugMessengerCreateInfo();
            createInfo.pNext = &debugCreateInfo;
        }

        auto [result, instance] = vk::createInstanceUnique(createInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create vulkan instance: {}", vk::to_string(result)));
        }

        return std::move(instance);
    }

    static bool checkValidationLayerSupport()
    {
        vk::ResultValue layers{ vk::enumerateInstanceLayerProperties() };
        if (layers.result != vk::Result::eSuccess) {
            return false;
        }

        for (const auto& layerName : s_validationLayers) {
            auto found = std::ranges::find_if(layers.value, [&layerName](const auto& layer) {
                return std::strcmp(layerName, layer.layerName) == 0;    // same string
            });
            if (found != layers.value.end()) {
                return true;
            }
        }
        return false;
    }

    static vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface) noexcept(false)
    {
        auto [result, devices] = instance.enumeratePhysicalDevices();
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to enumerate physical devices: {}", vk::to_string(result)));
        }
        if (devices.empty()) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        // sorted from best to worst [O(nlogn))]
        std::multimap<int, vk::PhysicalDevice, std::greater<>> candidates;
        for (const auto& device : devices) {
            int score = rateDeviceSuitability(device, surface);
            if (score > 0) {
                candidates.emplace(score, device);
                std::cout << "INFO: [Vulkan] Found suitable GPU: " << device.getProperties().deviceName << " (score: " << score << ")\n";
            }
        }
        if (candidates.empty()) {
            throw std::runtime_error("Failed to find a suitable GPU");
        }
        return candidates.begin()->second;
    }

    static int rateDeviceSuitability(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
    {
        int         score{ 0 };
        const auto& properties{ device.getProperties() };
        const auto& features{ device.getFeatures() };

        constexpr int SCORE_DISCRETE   = 1000;
        constexpr int SCORE_INTEGRATED = 100;
        constexpr int SCORE_VIRTUAL    = 10;
        constexpr int SCORE_CPU        = 1;

        // clang-format off
        // discrete gpu is better than integrated gpu
        switch (properties.deviceType) {
        using dt = vk::PhysicalDeviceType;
        case dt::eDiscreteGpu:   score += SCORE_DISCRETE;   break;
        case dt::eIntegratedGpu: score += SCORE_INTEGRATED; break;
        case dt::eVirtualGpu:    score += SCORE_VIRTUAL;    break;
        case dt::eCpu:           score += SCORE_CPU;        break;
        default: break;
        }
        // clang-format on

        // max image dimension affects texture graphics quality
        score += static_cast<int>(properties.limits.maxImageDimension2D);

        // geometry shader is required
        if (features.geometryShader == VK_FALSE) {
            return 0;
        }

        // check queue families completeness
        auto complete{ QueueFamilyIndices::checkCompleteness(device, surface) };
        if (!complete) {
            return 0;
        }

        // check device extension support
        if (!checkDeviceExtensionSupport(device)) {
            return 0;
        }

        // check swap chain support
        auto swapChainSupport{ querySwapChainSupport(device, surface) };
        if (!swapChainSupport.isAdequate()) {
            return 0;
        }

        return score;
    }

    static bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device)
    {
        auto [result, availableExtensions] = device.enumerateDeviceExtensionProperties();
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to enumerate device extension properties: {}", vk::to_string(result)));
        }

        for (const auto& extension : s_deviceExtensions) {
            auto found = std::ranges::find_if(availableExtensions, [&extension](const auto& availableExtension) {
                return std::strcmp(extension, availableExtension.extensionName) == 0;
            });
            if (found == availableExtensions.end()) {
                return false;
            }
        }

        return true;
    }

    static vk::UniqueDevice createLogicalDevice(const vk::PhysicalDevice& physicalDevice, const QueueFamilyIndices& indices) noexcept(false)
    {
        constexpr float queuePriority = 1.0F;

        auto uniqueIndices{ indices.getUniqueIndices() };

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        queueCreateInfos.reserve(uniqueIndices.size());
        for (const auto& index : uniqueIndices) {
            queueCreateInfos.push_back({
                .queueFamilyIndex = index,
                .queueCount       = 1,
                .pQueuePriorities = &queuePriority,
            });
        }

        vk::PhysicalDeviceFeatures deviceFeatures{
            // use default values for now
        };

        vk::DeviceCreateInfo createInfo{
            .queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos       = queueCreateInfos.data(),
            .enabledExtensionCount   = s_deviceExtensions.size(),
            .ppEnabledExtensionNames = s_deviceExtensions.data(),
            .pEnabledFeatures        = &deviceFeatures,
        };

        // setup validation layers (newer version of vulkan ignores this)
        if constexpr (s_enableValidation) {
            createInfo.enabledLayerCount   = s_validationLayers.size();
            createInfo.ppEnabledLayerNames = s_validationLayers.data();
        }

        auto [result, device] = physicalDevice.createDeviceUnique(createInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create logical device: {}", vk::to_string(result)));
        }

        return std::move(device);
    }

    static vk::UniqueSurfaceKHR createSurface(GLFWwindow* const window, const vk::Instance& instance) noexcept(false)
    {
        VkSurfaceKHR surface;
        if (auto vkResult{ glfwCreateWindowSurface(instance, window, nullptr, &surface) }; vkResult != VK_SUCCESS) {
            vk::Result result{ static_cast<vk::Result>(vkResult) };
            throw std::runtime_error(std::format("Failed to create window surface: {}", vk::to_string(result)));
        }
        return vk::UniqueSurfaceKHR{ surface, instance };
    }

    static SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface) noexcept(false)
    {
        auto [resultCap, capabilities] = device.getSurfaceCapabilitiesKHR(surface);
        auto [resultFormats, formats]  = device.getSurfaceFormatsKHR(surface);
        auto [resultModes, modes]      = device.getSurfacePresentModesKHR(surface);

        if (resultCap != vk::Result::eSuccess || resultFormats != vk::Result::eSuccess || resultModes != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to query swap chain support");
        }
        return {
            .m_capabilities = capabilities,
            .m_formats      = formats,
            .m_presentModes = modes,
        };
    }

    static std::tuple<vk::UniqueSwapchainKHR, vk::Format, vk::Extent2D> createSwapChain(
        const vk::PhysicalDevice& physicalDevice,
        const vk::Device&         device,
        const vk::SurfaceKHR&     surface,
        const QueueFamilyIndices& queueIndices,
        GLFWwindow*               window
    )
    {
        const auto swapChainSupport{ querySwapChainSupport(physicalDevice, surface) };
        const auto surfaceFormat{ swapChainSupport.chooseSurfaceFormat() };
        const auto presentMode{ swapChainSupport.choosePresentMode() };
        const auto extent{ swapChainSupport.chooseExtent(window) };

        uint32_t imageCount{ swapChainSupport.m_capabilities.minImageCount + 1 };
        if (swapChainSupport.m_capabilities.maxImageCount > 0 && imageCount > swapChainSupport.m_capabilities.maxImageCount) {
            imageCount = swapChainSupport.m_capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{
            .surface          = surface,
            .minImageCount    = imageCount,
            .imageFormat      = surfaceFormat.format,
            .imageColorSpace  = surfaceFormat.colorSpace,
            .imageExtent      = extent,
            .imageArrayLayers = 1,                                                   // always 1 unless stereoscopic 3d app
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,            // for post-processing, use vk::ImageUsageFlagBits::eTransferDst
            .preTransform     = swapChainSupport.m_capabilities.currentTransform,    // no transformation (use supportedTransform if you want to apply transformation)
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,              // ignore alpha channel
            .presentMode      = presentMode,
            .clipped          = VK_TRUE,
            .oldSwapchain     = nullptr,
        };

        // specify swap chain handling used across multiple queue families
        // - vk::SharingMode::eExclusive    : an image is owned by one queu family at a time (explicit ownership)
        // - vk::SharingMode::eConcurrent   : images can be used across multiple queue families
        if (queueIndices.m_graphicsFamily != queueIndices.m_presentFamily) {
            createInfo.imageSharingMode      = vk::SharingMode::eConcurrent;    // concurrent is easier to work with -- use exclusive for better performance.
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueIndices.asArray().data();
        } else {
            std::cout << "INFO: [Vulkan] Using exclusive swap chain since the family queue are the same\n";
            createInfo.imageSharingMode      = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;          // optional
            createInfo.pQueueFamilyIndices   = nullptr;    // optional
        }

        auto [result, swapChain] = device.createSwapchainKHRUnique(createInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create swap chain: {}", vk::to_string(result)));
        }
        return { std::move(swapChain), surfaceFormat.format, extent };
    }

    static std::vector<vk::UniqueImageView> createImageViews(
        const vk::Device&             device,
        const std::vector<vk::Image>& swapchainImages,
        const vk::Format&             swapchainFormat
    )
    {
        std::vector<vk::UniqueImageView> imageViews;
        imageViews.reserve(swapchainImages.size());

        for (const auto& image : swapchainImages) {
            vk::ImageViewCreateInfo createInfo{
                .image      = image,
                .viewType   = vk::ImageViewType::e2D,
                .format     = swapchainFormat,
                .components = /* swizzle the color channels around */ {
                    .r = vk::ComponentSwizzle::eIdentity,    // default mapping
                    .g = vk::ComponentSwizzle::eIdentity,
                    .b = vk::ComponentSwizzle::eIdentity,
                    .a = vk::ComponentSwizzle::eIdentity,
                },
                .subresourceRange = /* describe what the image purpose is */ {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,    // color target
                    .baseMipLevel   = 0,                                  // no mipmap
                    .levelCount     = 1,                                  //
                    .baseArrayLayer = 0,                                  //
                    .layerCount     = 1,                                  // single layer
                }
            };
            auto [result, imageView] = device.createImageViewUnique(createInfo);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error(std::format("Failed to create image view: {}", vk::to_string(result)));
            }
            imageViews.push_back(std::move(imageView));
        }

        return imageViews;
    }

    static vk::UniqueRenderPass createRenderPass(const vk::Device& device, const vk::Format& swapChainImageFormat) noexcept(false)
    {
        // single color buffer attachment
        vk::AttachmentDescription colorAttachment{
            .format         = swapChainImageFormat,
            .samples        = vk::SampleCountFlagBits::e1,
            .loadOp         = vk::AttachmentLoadOp::eClear,
            .storeOp        = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout  = vk::ImageLayout::eUndefined,
            .finalLayout    = vk::ImageLayout::ePresentSrcKHR,
        };

        vk::AttachmentReference colorAttachmentRef{
            .attachment = 0,    // index of the attachment in the attachment descriptions array
            .layout     = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::SubpassDescription subpassDesciption{
            .pipelineBindPoint    = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachmentRef,
        };

        vk::SubpassDependency subpassDependency{
            .srcSubpass    = VK_SUBPASS_EXTERNAL,    // implicit subpass before the render pass
            .dstSubpass    = 0,                      // first subpass, our subpass
            .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        };

        vk::RenderPassCreateInfo renderPassInfo{
            .attachmentCount = 1,
            .pAttachments    = &colorAttachment,
            .subpassCount    = 1,
            .pSubpasses      = &subpassDesciption,
            .dependencyCount = 1,
            .pDependencies   = &subpassDependency,
        };

        auto [result, renderPass] = device.createRenderPassUnique(renderPassInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create render pass: {}", vk::to_string(result)));
        }

        return std::move(renderPass);
    }

    static std::pair<vk::UniquePipelineLayout, vk::UniquePipeline> createPipelineLayout(vk::Device& device, vk::RenderPass& renderPass) noexcept(false)
    {
        const auto loadShader = [](std::filesystem::path&& filepath) {
            if (!std::filesystem::exists(filepath)) {
                throw std::runtime_error(std::format("No such file '{}'", filepath.string()));
            }

            std::ifstream file{ filepath.string(), std::ios::ate | std::ios::binary };    // start reading from the end of the file
            if (!file.is_open()) {
                throw std::runtime_error(std::format("Failed to open file '{}'", filepath.string()));
            }

            auto              fileSize{ static_cast<std::size_t>(file.tellg()) };
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

            return buffer;
        };

        const auto createShaderModule = [&device](const std::vector<char>& shaderCode) {
            vk::ShaderModuleCreateInfo createInfo{
                .codeSize = shaderCode.size(),
                .pCode    = reinterpret_cast<const uint32_t*>(shaderCode.data()),    // hmm, is this safe?
            };
            auto [result, shaderModule] = device.createShaderModuleUnique(createInfo);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error(std::format("Failed to create shader module: {}", vk::to_string(result)));
            }
            return std::move(shaderModule);
        };

        // load shader code
        auto vertShaderCode{ loadShader("assets/shader/vert.spv") };
        auto fragShaderCode{ loadShader("assets/shader/frag.spv") };

        // create shader modules
        vk::UniqueShaderModule vertShaderModule{ createShaderModule(vertShaderCode) };
        vk::UniqueShaderModule fragShaderModule{ createShaderModule(fragShaderCode) };

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = vertShaderModule.get(),
            .pName               = "main",     // entry point
            .pSpecializationInfo = nullptr,    // (optional) specify values for shader constants
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage               = vk::ShaderStageFlagBits::eFragment,
            .module              = fragShaderModule.get(),
            .pName               = "main",
            .pSpecializationInfo = nullptr,
        };
        std::array shaderStages{
            vertShaderStageInfo,
            fragShaderStageInfo,
        };

        // describe vertex input (none at the moment)
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount   = 0,
            .pVertexBindingDescriptions      = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions    = nullptr,
        };

        // describe input assembly
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .topology               = vk::PrimitiveTopology::eTriangleList,    // triangle from every 3 vertices without reuse
            .primitiveRestartEnable = VK_FALSE,
        };

        // // static viewport and scissor states
        // vk::Viewport viewport{
        //     .x        = 0.0F,
        //     .y        = 0.0F,
        //     .width    = static_cast<float>(swapChainExtent.width),     // NOTE: swap chain extent is in pixels, not in screen coordinates,
        //     .height   = static_cast<float>(swapChainExtent.height),    // the swap chain images will be used as framebuffers later on, so we should stick to its size
        //     .minDepth = 0.0F,
        //     .maxDepth = 1.0F,
        // };
        // vk::Rect2D scissor{
        //     .offset = { 0, 0 },
        //     .extent = swapChainExtent,
        // };
        // vk::PipelineViewportStateCreateInfo viewportStateInfo{
        //     .viewportCount = 1,
        //     .pViewports    = &viewport,
        //     .scissorCount  = 1,
        //     .pScissors     = &scissor,
        // };

        // dynamic viewport and scissor states
        std::array dynamicStates{
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data(),
        };
        vk::PipelineViewportStateCreateInfo viewportStateInfo{
            .viewportCount = 1,
            // .pViewports    = nullptr,    // dynamic, will be set later at draw time
            .scissorCount = 1,
            // .pScissors     = nullptr,    // dynamic, will be set later at draw time
        };

        // rasterizer
        vk::PipelineRasterizationStateCreateInfo rasterizerInfo{
            .depthClampEnable        = VK_FALSE,                       // enabling this requires enabling gpu feature
            .rasterizerDiscardEnable = VK_FALSE,                       //
            .polygonMode             = vk::PolygonMode::eFill,         // using mode other than fill requires enabling gpu feature
            .cullMode                = vk::CullModeFlagBits::eBack,    //
            .frontFace               = vk::FrontFace::eClockwise,      //
            .depthBiasEnable         = VK_FALSE,                       //
            // .depthBiasConstantFactor = 0.0F,
            // .depthBiasClamp          = 0.0F,
            // .depthBiasSlopeFactor    = 0.0F,
            .lineWidth = 1.0F,    // using line width other than 1.0f requires enabling gpu feature
        };

        // multisampling (disable for now)
        vk::PipelineMultisampleStateCreateInfo multisamplingInfo{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = VK_FALSE,
            // .minSampleShading      = 1.0F,
            // .pSampleMask           = nullptr,
            // .alphaToCoverageEnable = VK_FALSE,
            // .alphaToOneEnable      = VK_FALSE,
        };

        // depth and stencil testing (none for now)
        /* literally nothing */

        // color blending
        using c = vk::ColorComponentFlagBits;
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            // // common blending values
            // .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            // .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            // .colorBlendOp        = vk::BlendOp::eAdd,
            // .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            // .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            // .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask = c::eR | c::eG | c::eB | c::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlendInfo{
            .logicOpEnable = VK_FALSE,    // set to true to use logic operations using bitwise operators
            // .logicOp         = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments    = &colorBlendAttachment,
            .blendConstants  = std::array{ 0.0F, 0.0F, 0.0F, 0.0F },
        };

        // pipeline layout (empty for now)
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount         = 0,
            .pSetLayouts            = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges    = nullptr,
        };

        auto [result, pipelineLayout] = device.createPipelineLayoutUnique(pipelineLayoutInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create pipeline layout: {}", vk::to_string(result)));
        }

        // finally, the real pipeline creation
        vk::GraphicsPipelineCreateInfo graphicsPipelineInfo{
            .stageCount          = static_cast<uint32_t>(shaderStages.size()),
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState      = &viewportStateInfo,
            .pRasterizationState = &rasterizerInfo,
            .pMultisampleState   = &multisamplingInfo,
            .pDepthStencilState  = nullptr,    // no depth and stencil testing for now
            .pColorBlendState    = &colorBlendInfo,
            .pDynamicState       = &dynamicStateInfo,
            .layout              = *pipelineLayout,
            .renderPass          = renderPass,
            .subpass             = 0,    // index of the subpass where this pipeline will be used
            // .basePipelineHandle  = nullptr,    // optional
            // .basePipelineIndex   = -1,         // optional
        };

        auto [result2, graphicsPipeline] = device.createGraphicsPipelineUnique(nullptr, graphicsPipelineInfo);
        if (result2 != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create graphics pipeline: {}", vk::to_string(result2)));
        }
        return { std::move(pipelineLayout), std::move(graphicsPipeline) };
    }

    static std::vector<vk::UniqueFramebuffer> createFramebuffers(
        const vk::Device&                       device,
        const vk::RenderPass&                   renderPass,
        const std::vector<vk::UniqueImageView>& swapChainImageViews,
        const vk::Extent2D&                     swapChainExtent
    ) noexcept(false)
    {
        std::vector<vk::UniqueFramebuffer> framebuffers;

        framebuffers.reserve(swapChainImageViews.size());
        for (const auto& imageView : swapChainImageViews) {
            vk::FramebufferCreateInfo framebufferInfo{
                .renderPass      = renderPass,
                .attachmentCount = 1,
                .pAttachments    = &(*imageView),    // NOTE: pointer conversion to the underlying type
                .width           = swapChainExtent.width,
                .height          = swapChainExtent.height,
                .layers          = 1,
            };

            auto [result, framebuffer] = device.createFramebufferUnique(framebufferInfo);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error(std::format("Failed to create framebuffer: {}", vk::to_string(result)));
            }
            framebuffers.push_back(std::move(framebuffer));
        }

        return framebuffers;
    }

    static vk::UniqueCommandPool createCommandPool(const vk::Device& device, const QueueFamilyIndices& queueFamilyIndices) noexcept(false)
    {
        vk::CommandPoolCreateInfo commandPoolInfo{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,    // allow command buffer to be reset individually
            .queueFamilyIndex = queueFamilyIndices.m_graphicsFamily,
        };

        auto [result, commandPool] = device.createCommandPoolUnique(commandPoolInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to create command pool: {}", vk::to_string(result)));
        }
        return std::move(commandPool);
    }

    static std::vector<vk::UniqueCommandBuffer> createCommandBuffers(const vk::Device& device, const vk::CommandPool& commandPool) noexcept(false)
    {
        vk::CommandBufferAllocateInfo commandBufferInfo{
            .commandPool        = commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,    // primary: can be submitted to a queue for execution, but cannot be called from other command buffers
            .commandBufferCount = s_maxFramesInFlight,
        };

        auto [result, commandBuffers] = device.allocateCommandBuffersUnique(commandBufferInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to allocate command buffer: {}", vk::to_string(result)));
        }
        return std::move(commandBuffers);
    }

    static std::vector<SyncObject> createSyncObjects(const vk::Device& device)
    {
        vk::SemaphoreCreateInfo semaphoreInfo{
            .flags = {},    // default for now
        };
        vk::FenceCreateInfo fenceInfo{
            .flags = vk::FenceCreateFlagBits::eSignaled,    // start with signaled state
        };

        std::vector<SyncObject> syncObjects(s_maxFramesInFlight);
        for (auto& sync : syncObjects) {
            auto imageAvailableSemaphore{ device.createSemaphoreUnique(semaphoreInfo) };
            auto renderFinishedSemaphore{ device.createSemaphoreUnique(semaphoreInfo) };
            auto inFlightFence{ device.createFenceUnique(fenceInfo) };

            if (
                imageAvailableSemaphore.result != vk::Result::eSuccess
                || renderFinishedSemaphore.result != vk::Result::eSuccess
                || inFlightFence.result != vk::Result::eSuccess
            ) {
                throw std::runtime_error("Failed to create synchronization objects for a frame");
            }

            sync.m_imageAvailableSemaphore = std::move(imageAvailableSemaphore.value);
            sync.m_renderFinishedSemaphore = std::move(renderFinishedSemaphore.value);
            sync.m_inFlightFence           = std::move(inFlightFence.value);
        }

        return syncObjects;
    }

public:
    void recordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t imageIndex) noexcept(false)
    {
        constexpr vk::ClearValue clearValue{
            .color = { std::array{ 0.01F, 0.01F, 0.02F, 1.0F } }
        };

        vk::CommandBufferBeginInfo commandBeginInfo{
            .flags            = {},    // (optional) use default flags
            .pInheritanceInfo = {},    // (optional) only relevant for secondary command buffers
        };
        auto result = commandBuffer.begin(commandBeginInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to begin recording command buffer: {}", vk::to_string(result)));
        }

        vk::RenderPassBeginInfo renderPassBeginInfo{
            .renderPass  = *m_renderPass,
            .framebuffer = *m_swapChainFramebuffers[imageIndex],
            .renderArea  = {
                 .offset = { 0, 0 },
                 .extent = m_swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues    = &clearValue,
        };
        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);

        vk::Viewport viewport{
            .x        = 0.0F,
            .y        = 0.0F,
            .width    = static_cast<float>(m_swapChainExtent.width),    // NOTE: swap chain extent is in pixels, not in screen coordinates
            .height   = static_cast<float>(m_swapChainExtent.height),
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{
            .offset = { 0, 0 },
            .extent = m_swapChainExtent,
        };
        commandBuffer.setScissor(0, scissor);

        commandBuffer.draw(3, 1, 0, 0);

        commandBuffer.endRenderPass();

        result = commandBuffer.end();
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to record command buffer: {}", vk::to_string(result)));
        }
    }

    void drawFrame()
    {
        /* Outline of a frame:
         * 1. wait for the previous frame to finish
         * 2. acquire an image from the swap chain
         * 3. record a command buffer which draws the scene onto that image
         * 4. submit the recorded command buffer
         * 5. present the swap chain image
         */

        /* Syncronization order:
         * 1. acquire image from swap chain
         * 2. execute command that draw onto the acquired image
         * 3. present image to screen
         */
        constexpr auto timeoutNano{ std::numeric_limits<uint64_t>::max() };    // wait forever, effectively disabling timeout

        // NOLINTBEGIN: I'm too lazy to check every single function call for error
        std::array fences{ *m_syncs[m_currentFrameIndex].m_inFlightFence };
        auto       _1 = m_device->waitForFences(fences, VK_TRUE, timeoutNano);
        auto       _2 = m_device->resetFences(fences);
        // NOLINTEND

        // acquire image from swap chain
        auto [result, imageIndex] = m_device->acquireNextImageKHR(
            *m_swapChain, timeoutNano, *m_syncs[m_currentFrameIndex].m_imageAvailableSemaphore
        );
        if (result != vk::Result::eSuccess) {
            /* to be filled later */
        }

        // record command buffer
        m_commandBuffers[m_currentFrameIndex]->reset();    // reset the command buffer not the unique pointer itself
        recordCommandBuffer(m_commandBuffers[m_currentFrameIndex].get(), imageIndex);

        // submit command buffer
        std::array<vk::Semaphore, 1>          waitSemaphore{ *m_syncs[m_currentFrameIndex].m_imageAvailableSemaphore };
        std::array<vk::PipelineStageFlags, 1> waitStages{ vk::PipelineStageFlagBits::eColorAttachmentOutput };

        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount   = waitSemaphore.size(),
            .pWaitSemaphores      = waitSemaphore.data(),
            .pWaitDstStageMask    = waitStages.data(),
            .commandBufferCount   = 1,
            .pCommandBuffers      = &m_commandBuffers[m_currentFrameIndex].get(),
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &m_syncs[m_currentFrameIndex].m_renderFinishedSemaphore.get(),
        };

        auto submitResult = m_graphicsQueue.submit(1, &submitInfo, *m_syncs[m_currentFrameIndex].m_inFlightFence);
        if (submitResult != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("Failed to submit draw command buffer: {}", vk::to_string(result)));
        }

        // present image to screen
        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &m_syncs[m_currentFrameIndex].m_renderFinishedSemaphore.get(),
            .swapchainCount     = 1,
            .pSwapchains        = &m_swapChain.get(),
            .pImageIndices      = &imageIndex,
            .pResults           = nullptr,    // (optional) specify the result of each swap chain presentation (if multiple swap chains)
        };
        auto presentResult = m_presentQueue.presentKHR(presentInfo);
        if (presentResult != vk::Result::eSuccess) {
            /* to be filled later */
        }

        m_currentFrameIndex = (m_currentFrameIndex + 1) % s_maxFramesInFlight;
    }
};

#endif /* end of include guard: VULKAN_HPP_GVYQUUQ0 */
