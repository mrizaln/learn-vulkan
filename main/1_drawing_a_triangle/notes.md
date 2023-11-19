# Drawing a Triangle

### Step 1 - Instance and physical device selection

A application starts by setting up the Vulkan API through a `VkInstance`. After creating the instance, you can query for Vulkan supported hardware and select one or more `VkPhysicalDevice`s to use for operations.

### Step 2 - Logical device and queue families

You need to create a `VkDevice` (logical device), where you describe more specifically which `VkPhysicalDeviceFeatures` you will be using. You also need to specify which queue families you would like to use. Most operations performed with Vulkan are asynchronously executed by submitting them to a `VkQueue`.

### Step 3 - Window surface and swap chain

Unless you are only interested in offscreen rendring, you will need to create a window to present rendered images to (can be created for example using GLFW).

We need two more components to actually render to a window: a window surface (`VkSurfaceKHR`) and a swap chain (`VkSwapchainKHR`). Note the `KHR` postfix, which means that these objects are part of a Vulkan extension.

### Step 4 - Image views and framebuffers

To draw to an image acquired from the swap chain, we have to wrap it into a `VkImageView` and `VkFramebuffer`. An image view references a specific part of an image to be used, and a framebuffer references image views that are to be used for color, depth , and stencil targets.

### Step 5 - Render passes

Render passes in Vulkan describe the type of images that are used during rendering operations, how they will be used, and how their contents should be treated.

### Step 6 - Graphics pipeline

The graphics pipeline in Vulkan is ste up by creating a `VkPipeline` object. It describes the configurable state of the graphics card, like the viewport size and depth buffer operation and the programmable state using `VkShaderModule` objects. The `VkShaderModule` objects are created from shader byte code.

### Step 7 - Command pools and command buffers

As mentioned earlier, many Vulkan operations need to be submitted to a queue. These operations first need to be recorded into a `VkCommandBuffer` before they can be submitted. These command buffers are allocated from a `VkCommandPool` that is associated with a specific queue family.

### Step 8 - Main loop

The main loop is quite straightforward. We first acquire an image from the swap chain with `vkAcquireNextImageKHR`. We can then select the appropriate command buffer for that image and execute it with `vkQueueSubmit`. Finally, we return the image to the swap chain for presentation to the screen with `vkQueuePresentKHR`.

## Summary

- Create a `VkInstance`
- Select a supported graphics card (`VkPhysicalDevice`)
- Create a `VkDevice` and `VkQueue` for drawing and presentation
- Create a window, window surface, and swap chain
- Wrap the swap chain images into `VkImageView`
- Create a render pass that specifies the render targets and usage
- Create framebuffers for the render pass
- Set up the graphics pipeline
- Allocate and record a command buffer with the draw commands for every possible swap chain image
- Draw frames bu acquiring images, submitting the rights draw command buffer and returning the images back to the swap chain
