#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "display_transform.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
static cudaGraphicsResource *pboResource = nullptr;
static bool useCudaGLInterop = false;
static uchar4 *devicePreview = nullptr;
static std::vector<uchar4> hostPreview;
GLuint displayImage;

GLFWwindow *window;

void checkCudaInterop(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width * 3, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[24], texcoords[24];
    GLushort indices[18];
    for (int panel = 0; panel < 3; ++panel) {
        float x0 = -1 + 2.0f * panel / 3, x1 = -1 + 2.0f * (panel + 1) / 3;
        float u0 = panel / 3.0f, u1 = (panel + 1) / 3.0f;
        const GLfloat v[] = {x0, -1, x1, -1, x1, 1, x0, 1};
        const GLfloat t[] = {u1, 1, u0, 1, u0, 0, u1, 0};
        const GLushort ix[] = {0, 1, 3, 3, 1, 2};
        for (int i = 0; i < 8; ++i) { vertices[8 * panel + i] = v[i]; texcoords[8 * panel + i] = t[i]; }
        for (int i = 0; i < 6; ++i) indices[6 * panel + i] = 4 * panel + ix[i];
    }

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
    const char *attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        if (pboResource) {
            cudaGraphicsUnregisterResource(pboResource);
            pboResource = nullptr;
        }

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (devicePreview) {
        cudaFree(devicePreview);
        devicePreview = nullptr;
    }
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {
    fprintf(stdout, "OpenGL renderer: %s\n", glGetString(GL_RENDERER));
    fflush(stdout);
    unsigned int deviceCount = 0;
    int device = 0;
    const char *vendor = reinterpret_cast<const char *>(glGetString(GL_VENDOR));
    // Non-NVIDIA display contexts cannot support CUDA interop. Avoid making an
    // unsupported CUDA call just to discover that on hybrid-graphics machines.
    cudaError_t result = cudaSuccess;
    if (vendor && std::string(vendor).find("NVIDIA") != std::string::npos)
        result = cudaGLGetDevices(&deviceCount, &device, 1, cudaGLDeviceListAll);
    useCudaGLInterop = result == cudaSuccess && deviceCount > 0;
    if (!useCudaGLInterop) {
        // Hybrid graphics: CUDA computes on NVIDIA, OpenGL displays on the iGPU.
        if (result != cudaSuccess) cudaGetLastError();
        device = 0;
    }
    checkCudaInterop(cudaSetDevice(device), "Selecting CUDA device");
    cudaDeviceProp properties;
    checkCudaInterop(cudaGetDeviceProperties(&properties, device), "Querying CUDA device");
    fprintf(stdout, "CUDA device: %s\nPreview transfer: %s\n", properties.name,
            useCudaGLInterop ? "CUDA/OpenGL interop" : "host staging (hybrid graphics)");
    fflush(stdout);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * 3 * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    if (useCudaGLInterop) {
        checkCudaInterop(cudaGraphicsGLRegisterBuffer(&pboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard),
                         "Registering the OpenGL pixel buffer with CUDA");
    } else {
        hostPreview.resize(num_texels);
        checkCudaInterop(cudaMalloc(&devicePreview, size_tex_data), "Allocating the preview buffer");
    }

}

uchar4 *mapPreviewBuffer() {
    if (!useCudaGLInterop) return devicePreview;
    uchar4 *pixels = nullptr;
    size_t mappedSize = 0;
    checkCudaInterop(cudaGraphicsMapResources(1, &pboResource), "Mapping the pixel buffer");
    checkCudaInterop(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &mappedSize, pboResource),
                     "Getting the mapped pixel buffer");
    return pixels;
}

void unmapPreviewBuffer() {
    if (useCudaGLInterop) {
        checkCudaInterop(cudaGraphicsUnmapResources(1, &pboResource), "Unmapping the pixel buffer");
    } else {
        const size_t size = hostPreview.size() * sizeof(uchar4);
        checkCudaInterop(cudaMemcpy(hostPreview.data(), devicePreview, size, cudaMemcpyDeviceToHost),
                         "Copying the rendered preview");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, size, hostPreview.data());
    }
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    if (hiddenPreview) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    int initialWidth = mode ? glm::min(width * 3 + 320, mode->width - 80) : width * 3 + 320;
    int initialHeight = glm::max(640, (initialWidth - 320) * height / (width * 3) + 48);
    if (mode) initialHeight = glm::min(initialHeight, mode->height - 100);
    window = glfwCreateWindow(initialWidth, initialHeight, "CUDA Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSwapInterval(1); // Enable vsync

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    {

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    return true;
}

static ImGuiWindowFlags windowFlags= ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
static bool ui_autoresize = true;
static bool ui_hide = false;
static float panelLeft = 320, panelWidth = 400, panelTop = 32;

void drawGui(int windowWidth, int windowHeight) {
    const bool previousDenoise = ui_denoise_enable;
    const bool previousTemporal = ui_temporal_enable;
    const bool previousSpatial = ui_spatial_enable;
    const bool previousSeparateColor = ui_sepcolor;
    const bool previousAddColor = ui_addcolor;
    // Dear imgui new frame
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    // Dear imgui define
    {
        ImVec2 minSize(300.f, glm::min(600.f, (float)windowHeight));
        ImVec2 maxSize(310.f, (float)windowHeight);
        ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

        ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

        if (ImGui::IsKeyPressed('H')) {
            ui_hide = !ui_hide;
        }

        ImGui::Begin("Control Panel", 0, windowFlags);
        ImGui::SetWindowFontScale(1);
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::Text("Trace %.1f | Old %.1f | New %.1f ms", framePerformance.tracing,
                    framePerformance.originalDenoise, framePerformance.optimizedDenoise);

        // Capture keyboard
        if (ImGui::IsKeyPressed('A')) {
            ui_automate_camera = !ui_automate_camera;
        }
        if (ImGui::IsKeyPressed(' ')) {
            ImGui::SetWindowCollapsed(!ImGui::IsWindowCollapsed());
        }
        
        ImGui::Checkbox("Auto-Resize", &ui_autoresize);
        if (ui_autoresize) {
            windowFlags |= ImGuiWindowFlags_AlwaysAutoResize;
        } else {
            windowFlags &= ~ImGuiWindowFlags_AlwaysAutoResize;
        }

        if (ImGui::CollapsingHeader("Ray Tracing", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Run", &ui_run);
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                ui_step = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                ui_reset_denoiser = true;
            }
            ImGui::SliderInt("Max. Depth", &ui_tracedepth, 1, 10);
            ImGui::Checkbox("Use KD-Tree", &ui_usekdtree);
            ImGui::Separator();
            ImGui::Checkbox("Trace Shadow Ray", &ui_shadowray);
            ImGui::SameLine();
            ImGui::Checkbox("Reduce Var.", &ui_reducevar);
            ImGui::SliderFloat("SR Int.", &ui_sintensity, 0.0f, 20.0f);
            ImGui::SliderFloat("Sample Rad.", &ui_lightradius, 0.0f, 2.0f);
            ImGui::Checkbox("Pulse light (anti-lag test)", &ui_pulse_light);
        }

        if (ImGui::CollapsingHeader("Display")) {
            ImGui::Checkbox("Tone map + sRGB", &displayToneMap);
            ImGui::SliderFloat("Exposure (EV)", &displayExposureEV, -4.0f, 4.0f);
        }

        if (ImGui::CollapsingHeader("Denoising", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Original denoiser (center)");
            if (ImGui::IsKeyPressed('D')) {
                ui_denoise_enable = !ui_denoise_enable;
            }
            else if (ImGui::IsKeyPressed('T')) {
                ui_temporal_enable = !ui_temporal_enable;
            }
            else if (ImGui::IsKeyPressed('F')) {
                ui_spatial_enable = !ui_spatial_enable;
            }
            if (ImGui::Checkbox("Enable(D)", &ui_denoise_enable)) {
                if (ui_denoise_enable) {

                }
                else {
                    camchanged = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("None")) {
                ui_denoise_enable = false;
                ui_temporal_enable = false;
                ui_spatial_enable = false;
                ui_sepcolor = false;
                ui_addcolor = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("All")) {
                ui_denoise_enable = true;
                ui_temporal_enable = true;
                ui_spatial_enable = true;
                ui_sepcolor = true;
                ui_addcolor = true;
            }
            ImGui::Checkbox("Temporal(T)", &ui_temporal_enable);
            ImGui::SameLine();
            ImGui::Checkbox("Spatial(F)", &ui_spatial_enable);
            ImGui::Checkbox("Rmv 1st Albedo", &ui_sepcolor);
            ImGui::SameLine();
            ImGui::Checkbox("Add 1st Albedo", &ui_addcolor);
            ImGui::Separator();
            ImGui::Text("Temporal Acc.");
            ImGui::SliderFloat("C. Alpha", &ui_color_alpha, 0.0f, 1.0f);
            ImGui::SliderFloat("M. Alpha", &ui_moment_alpha, 0.0f, 1.0f);
            if (ImGui::Button("Set Default Param.##1")) {
                ui_color_alpha = 0.2f;
                ui_moment_alpha = 0.2f;
            }
            ImGui::Separator();
            ImGui::Text("Variance Est.");
            ImGui::Checkbox("Blur Var.", &ui_blurvariance);
            ImGui::SliderFloat("Sigma L.", &ui_sigmal, 0.0f, 2.0f);
            ImGui::SliderFloat("Sigma X.", &ui_sigmax, 0.0f, 1.0f);
            ImGui::SliderFloat("Sigma N.", &ui_sigman, 0.0f, 1.0f);
            if (ImGui::Button("Set Default Param.##2")) {
                ui_sigmal = 0.7f;
                ui_sigmax = 0.35f;
                ui_sigman = 0.2f;
            }
            ImGui::Separator();
            ImGui::Text("A-Trous Wavelet");
            ImGui::SliderInt("Num. Lv.", &ui_atrous_nlevel, 0, 7);
            ImGui::SliderInt("Hist. Lv.", &ui_history_level, 0, ui_atrous_nlevel);
        }

        if (ImGui::CollapsingHeader("SVGF variants (right)", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool changed = false;
            if (ImGui::Button("Previous CUDA")) {
                optimizedSettings = OptimizedDenoiseSettings{};
                optimizedSettings.falcorAligned = false;
                optimizedSettings.antiLag = optimizedSettings.historyClamping = true;
                optimizedSettings.alpha = 0.2f; optimizedSettings.sigmaLuminance = 4;
                optimizedSettings.levels = 5;
                optimizedSettings.feedbackTap = 0;
                changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Previous SVGF preset")) {
                optimizedSettings = OptimizedDenoiseSettings{};
                optimizedSettings.alpha = 0.2f; optimizedSettings.sigmaLuminance = 2;
                optimizedSettings.feedbackTap = 0;
                changed = true;
            }
            if (ImGui::Button("Falcor defaults")) {
                optimizedSettings = OptimizedDenoiseSettings{};
                optimizedSettings.alpha = 0.05f; optimizedSettings.sigmaLuminance = 10;
                optimizedSettings.feedbackTap = 1;
                changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Adaptive")) {
                optimizedSettings = OptimizedDenoiseSettings{};
                optimizedSettings.antiLag = optimizedSettings.historyClamping = true;
                changed = true;
            }
            changed |= ImGui::Checkbox("Anti-lag", &optimizedSettings.antiLag);
            ImGui::SameLine();
            changed |= ImGui::Checkbox("History clamp", &optimizedSettings.historyClamping);
            changed |= ImGui::SliderInt("Fast history", &optimizedSettings.fastHistoryFrames, 1, 8);
            changed |= ImGui::SliderFloat("Clamp sigma", &optimizedSettings.clampSigma, 1.0f, 4.0f);
            changed |= ImGui::SliderFloat("Anti-lag strength", &optimizedSettings.antiLagStrength, 0.0f, 2.0f);
            changed |= ImGui::Checkbox("Temporal##optimized", &optimizedSettings.temporal);
            ImGui::SameLine();
            changed |= ImGui::Checkbox("Spatial##optimized", &optimizedSettings.spatial);
            changed |= ImGui::Checkbox("Preserve albedo detail", &optimizedSettings.demodulate);
            changed |= ImGui::SliderFloat("Alpha##optimized", &optimizedSettings.alpha, 0.02f, 1.0f);
            if (optimizedSettings.falcorAligned) {
                changed |= ImGui::SliderFloat("Moments alpha", &optimizedSettings.momentsAlpha, 0.02f, 1.0f);
                changed |= ImGui::SliderInt("Feedback tap", &optimizedSettings.feedbackTap, 0, 3);
            }
            changed |= ImGui::SliderFloat("Sigma L##optimized", &optimizedSettings.sigmaLuminance, 0.5f, 16.0f);
            changed |= ImGui::SliderInt("Levels##optimized", &optimizedSettings.levels, 0, 7);
            if (ImGui::Button("Reset optimized history")) changed = true;
            if (changed) optimizedDenoiseReset();
            const char *views[] = {"Filtered", "History length", "Anti-lag response"};
            ImGui::Combo("Debug##optimized", &optimizedSettings.debugView, views, IM_ARRAYSIZE(views));
        }

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::IsKeyPressed('Z') || ImGui::Button("Reset Camera(Z)")) {
                scene->resetCamera();
                resetCamera();
            }

            if (ImGui::TreeNodeEx("Camera Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                Camera & cam = scene->state.camera;
                ImGui::Text("Camera Up: (%.3f, %.3f, %.3f)", cam.up.x, cam.up.y, cam.up.z);
                ImGui::Text("Camera Right: (%.3f, %.3f, %.3f)", cam.right.x, cam.right.y, cam.right.z);
                ImGui::Text("Camera View: (%.3f, %.3f, %.3f)", cam.view.x, cam.view.y, cam.view.z);
                ImGui::Text("Camera Pos: (%.3f, %.3f, %.3f)", cam.position.x, cam.position.y, cam.position.z);
                ImGui::Text("Theta: %.3f, Phi: %.3f", theta, phi);
                ImGui::Text("Zoom: %.3f", zoom);
                ImGui::TreePop();
            }
            ImGui::Separator();
            ImGui::SliderFloat("Spd. X", &ui_camera_speed_x, 0.0f, 1.5f);
            ImGui::SliderFloat("Spd. Y", &ui_camera_speed_y, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Z", &ui_camera_speed_z, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Theta", &ui_camera_speed_theta, 0.0f, 0.5f);
            ImGui::SliderFloat("Spd. Phi", &ui_camera_speed_phi, 0.0f, 0.5f);
            ImGui::Checkbox("Automate Camera Motion(A)", &ui_automate_camera);
            ImGui::TextWrapped("A: start/stop orbit from this view. Drag to take control.");
        }

        if (ImGui::CollapsingHeader("Debug View", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* listbox_items_left[] = { "1 spp" };
            ImGui::ListBox("Left View", &ui_left_view_option, listbox_items_left, IM_ARRAYSIZE(listbox_items_left), 3);
            const char* listbox_items_right[] = { "Filtered", "HistoryLenth", "Variance" };
            ImGui::ListBox("Center View", &ui_right_view_option, listbox_items_right, IM_ARRAYSIZE(listbox_items_right), 3);
        }

        ImGui::End();
    }

    if (previousDenoise != ui_denoise_enable || previousTemporal != ui_temporal_enable ||
        previousSpatial != ui_spatial_enable || previousSeparateColor != ui_sepcolor ||
        previousAddColor != ui_addcolor) {
        ui_reset_denoiser = true;
    }

    const char *labels[] = {"Raw / 1 spp", referenceRender ? "Unfiltered reference mean" : "Original denoiser",
        referenceRender ? "Unfiltered reference mean" : (optimizedSettings.falcorAligned ?
            (optimizedSettings.antiLag || optimizedSettings.historyClamping ? "Adaptive SVGF (experimental)" :
            "SVGF (Falcor-aligned CUDA)") : "Previous CUDA (RELAX-inspired)")};
    for (int panel = 0; panel < 3; ++panel) {
        float x = panelLeft + panelWidth * (panel + 0.5f) - ImGui::CalcTextSize(labels[panel]).x * 0.5f;
        ImGui::GetForegroundDrawList()->AddText(ImVec2(x, glm::max(5.0f, panelTop - 22)), IM_COL32(240, 240, 240, 255), labels[panel]);
    }

    // Dear imgui render
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

void mainLoop() {
    if (disableVsync) glfwSwapInterval(0);
    const double renderStart = glfwGetTime();
    double previousFrameTime = renderStart;
    int renderedFrames = 0;
    while (!glfwWindowShouldClose(window)) {
        const double now = glfwGetTime();
        renderDeltaSeconds = glm::clamp((float)(now - previousFrameTime), 0.0f, 0.1f);
        if (fixedDeltaSeconds > 0) renderDeltaSeconds = fixedDeltaSeconds;
        previousFrameTime = now;
        glfwPollEvents();

        // Path trace 1 frame
        if (ui_run || ui_step) {
            runCuda();
            ++renderedFrames;
            if (referenceRender && renderedFrames % 256 == 0) {
                printf("Reference: %d / %d spp\n", renderedFrames, renderFrameLimit); fflush(stdout);
            }
            if (ui_step) {
                ui_step = false;
            }
        }
        string title = "CUDA Path Tracer | Raw / Original / Optimized | Frame " + utilityCore::convertIntToString(frame);
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width * 3, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        int display_w, display_h, logical_w, logical_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glfwGetWindowSize(window, &logical_w, &logical_h);
        if (display_w <= 0 || display_h <= 0 || logical_w <= 0 || logical_h <= 0) continue;
        float sx = (float)display_w / logical_w, sy = (float)display_h / logical_h;
        int sidebar = ui_hide ? 0 : (int)(320 * sx);
        int availableWidth = glm::max(display_w - sidebar, 1);
        int availableHeight = glm::max(display_h - (int)(32 * sy), 1);
        float aspect = (float)(width * 3) / height;
        int viewWidth = glm::min(availableWidth, (int)(availableHeight * aspect));
        int viewHeight = glm::max(1, (int)(viewWidth / aspect));
        int vx = sidebar + (availableWidth - viewWidth) / 2, vy = (availableHeight - viewHeight) / 2;
        panelLeft = vx / sx; panelWidth = viewWidth / (3 * sx);
        panelTop = (display_h - vy - viewHeight) / sy;
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(vx, vy, viewWidth, viewHeight);
        glDrawElements(GL_TRIANGLES, 18,  GL_UNSIGNED_SHORT, 0);

        // Draw imgui
        glViewport(0, 0, display_w, display_h);
        drawGui(logical_w, logical_h);

        // Display content
        glfwSwapBuffers(window);
        if (renderFrameLimit > 0 && renderedFrames >= renderFrameLimit) glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if (!capturePrefix.empty()) saveLinearCapture(capturePrefix);
    if (!comparisonOutput.empty()) saveComparison(comparisonOutput);
    const double elapsed = glfwGetTime() - renderStart;
    printFrameProfiling();
    printf("Rendered %d frames in %.3f seconds (%.2f FPS, includes all three views).\n",
           renderedFrames, elapsed, elapsed > 0 ? renderedFrames / elapsed : 0);

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    pathtraceFree();
    denoiseFree();
    cleanupCuda();
    glfwDestroyWindow(window);
    glfwTerminate();
}
