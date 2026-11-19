#include "main.h"
#include "preview.h"
#include <cstring>
#include <chrono>
#include "display_transform.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

bool camchanged = true;
float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

// For camera automation
static float camera_tx = 0.0f;
static float camera_ty = 0.0f;
static float camera_tz = 0.0f;
static float camera_ttheta = 0.0f;
static float camera_tphi = 0.0f;

Scene *scene;
RenderState *renderState;
int frame = 0;

int width;
int height;
int renderFrameLimit = 0;
float renderDeltaSeconds = 1.0f / 60.0f;
bool hiddenPreview = false;
std::string comparisonOutput;
std::string capturePrefix;
std::string recordSequence;
bool referenceRender = false;
int sampleSeedOffset = 0;
bool disableVsync = false;
float fixedDeltaSeconds = 0;

/////////// GUI Controlled State ///////////
// Path Tracer
bool ui_run = true;
bool ui_pulse_light = false;
bool ui_step = false;
bool ui_reset_denoiser = false;
int ui_tracedepth = 4;
bool ui_shadowray = true;
bool ui_reducevar = true;
float ui_sintensity = 2.7f;
float ui_lightradius = 1.4f;
bool ui_usekdtree = true;

// Denoise
bool ui_denoise_enable = true;
bool ui_temporal_enable = true;
bool ui_spatial_enable = true;
float ui_color_alpha = 0.2;
float ui_moment_alpha = 0.2;
bool ui_blurvariance = true;
float ui_sigmal = 0.45f;
float ui_sigmax = 0.35f;
float ui_sigman = 0.2f;
int ui_atrous_nlevel = 5;   // How man levels of A-trous filter used in denoising?
int ui_history_level = 1;   // Which level of A-trous output is sent to history buffer?
bool ui_sepcolor = false;
bool ui_addcolor = false;

// Camera
bool ui_automate_camera = false;
float ui_camera_speed_x = 0.0;
float ui_camera_speed_y = 0.0;
float ui_camera_speed_z = 0.0;
float ui_camera_speed_theta = 0.4f;
float ui_camera_speed_phi = 0.5f;

// Debug View
int ui_left_view_option = 0;
int ui_right_view_option = 0;
////////////////////////////////////////////

void resetCamera() {
    ui_automate_camera = false;
    // Set up camera stuff from loaded path tracer settings
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position - cam.lookAt;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    zoom = glm::max(glm::length(cameraPosition), 1e-4f);
    phi = atan2f(cameraPosition.x, cameraPosition.z);
    theta = acosf(glm::clamp(cameraPosition.y / zoom, -1.0f, 1.0f));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    camchanged = true;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];
    bool startCameraAutomation = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            renderFrameLimit = atoi(argv[++i]);
            if (renderFrameLimit <= 0) { fprintf(stderr, "--frames requires a positive integer.\n"); return 1; }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            comparisonOutput = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0) {
            hiddenPreview = true;
        } else if (strcmp(argv[i], "--auto-camera") == 0) {
            startCameraAutomation = true;
        } else if (strcmp(argv[i], "--capture") == 0 && i + 1 < argc) {
            capturePrefix = argv[++i];
        } else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            recordSequence = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            referenceRender = true;
            renderFrameLimit = atoi(argv[++i]);
            if (renderFrameLimit <= 0) { fprintf(stderr, "--reference requires a positive sample count.\n"); return 1; }
        } else if (strcmp(argv[i], "--seed-offset") == 0 && i + 1 < argc) {
            sampleSeedOffset = atoi(argv[++i]);
            if (sampleSeedOffset < 0) { fprintf(stderr, "Seed offset must be nonnegative.\n"); return 1; }
        } else if (strcmp(argv[i], "--no-vsync") == 0) {
            disableVsync = true;
        } else if (strcmp(argv[i], "--fixed-dt") == 0 && i + 1 < argc) {
            fixedDeltaSeconds = (float)atof(argv[++i]);
            if (!(fixedDeltaSeconds > 0 && fixedDeltaSeconds <= 0.1f)) { fprintf(stderr, "Fixed dt must be in (0, 0.1].\n"); return 1; }
        } else if (strcmp(argv[i], "--denoiser") == 0 && i + 1 < argc) {
            const std::string mode = argv[++i];
            optimizedSettings = OptimizedDenoiseSettings{};
            if (mode == "previous") {
                optimizedSettings.falcorAligned = false;
                optimizedSettings.antiLag = optimizedSettings.historyClamping = true;
                optimizedSettings.alpha = 0.2f; optimizedSettings.sigmaLuminance = 4;
                optimizedSettings.levels = 5;
                optimizedSettings.feedbackTap = 0;
            } else if (mode == "falcor") {
                optimizedSettings.alpha = 0.05f; optimizedSettings.sigmaLuminance = 10;
                optimizedSettings.feedbackTap = 1;
            } else if (mode == "adaptive") {
                optimizedSettings.antiLag = optimizedSettings.historyClamping = true;
            } else if (mode != "svgf") {
                fprintf(stderr, "Denoiser must be previous, svgf, falcor or adaptive.\n"); return 1;
            }
        } else if (strcmp(argv[i], "--display") == 0 && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode != "legacy" && mode != "srgb") { fprintf(stderr, "Display must be legacy or srgb.\n"); return 1; }
            displayToneMap = mode == "srgb";
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]); return 1;
        }
    }
    if (referenceRender && startCameraAutomation) {
        fprintf(stderr, "Reference accumulation requires a stationary camera.\n"); return 1;
    }
    if (!recordSequence.empty() && (!hiddenPreview || renderFrameLimit <= 0 || referenceRender || startCameraAutomation)) {
        fprintf(stderr, "--record requires --hidden --frames and a stationary, non-reference run.\n"); return 1;
    }
    if (!capturePrefix.empty() && renderFrameLimit <= 0) {
        fprintf(stderr, "--capture requires --frames or --reference.\n"); return 1;
    }

    // Load scene file
    try {
        scene = new Scene(sceneFile);
        ui_lightradius = scene->state.lightRadius;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
    resetCamera();
    ui_automate_camera = startCameraAutomation;

    // Initialize CUDA and GL components
    if (!init()) {
        fprintf(stderr, "Failed to initialize the CUDA/OpenGL preview.\n");
        return EXIT_FAILURE;
    }

    // GLFW main loop
    frame = 0;
    mainLoop();

    return 0;
}

void saveImage() {
    retrieveOriginalImage();
    float samples = frame;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix));
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    static bool resourcesInitialized = false;
    static bool wasAutomating = false;
    static glm::vec3 automationLookAt;
    static float automationTheta = 0, automationPhi = 0;
    // camera automatic motion
    if (ui_automate_camera) {
        Camera &cam = renderState->camera;
        if (!wasAutomating) {
            automationLookAt = cam.lookAt;
            automationTheta = theta;
            automationPhi = phi;
            camera_tx = camera_ty = camera_tz = camera_ttheta = camera_tphi = 0;
        }
        const float dt = wasAutomating ? renderDeltaSeconds : 0;
        camera_tx = fmodf(camera_tx + ui_camera_speed_x * dt, TWO_PI);
        camera_ty = fmodf(camera_ty + ui_camera_speed_y * dt, TWO_PI);
        camera_tz = fmodf(camera_tz + ui_camera_speed_z * dt, TWO_PI);
        camera_ttheta = fmodf(camera_ttheta + ui_camera_speed_theta * dt, TWO_PI);
        camera_tphi = fmodf(camera_tphi + ui_camera_speed_phi * dt, TWO_PI);
        cam.lookAt = automationLookAt + zoom * glm::vec3(0.1f * sinf(camera_tx),
                      0.05f * sinf(camera_ty), 0.1f * sinf(camera_tz));
        theta = glm::clamp(automationTheta + PI / 60 * sinf(camera_ttheta), 0.001f, PI - 0.001f);
        phi = automationPhi + PI / 22.5f * sinf(camera_tphi);
        camchanged = true;
    }
    wasAutomating = ui_automate_camera;

    if (camchanged) {
        if (!ui_denoise_enable) frame = 0;

        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    ui_reset_denoiser |= (frame == 0);

    if (!resourcesInitialized) {
        pathtraceInit(scene);
        denoiseInit(scene);
        resourcesInitialized = true;
    }

    if (ui_reset_denoiser == true) {
        pathtraceReset();
        denoiseReset();
        frame = 0;
        ui_reset_denoiser = false;
    }

    // Use direct interop or a staging buffer when OpenGL runs on another GPU.
    uchar4 *pbo_dptr = mapPreviewBuffer();
    pathtrace(pbo_dptr, frame++);        // execute the kernel
    waitForFrame();
    const auto transferStart = std::chrono::steady_clock::now();
    unmapPreviewBuffer();
    finishFrameProfiling(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - transferStart).count());
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_R:
        ui_automate_camera = false;
        camchanged = true;
        cam.lookAt = ogLookAt;
        break;
      
      }
    }
   
    if (action == GLFW_REPEAT || action == GLFW_PRESS) {
        if (key == GLFW_KEY_UP || key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT ||
            key == GLFW_KEY_RIGHT || key == GLFW_KEY_COMMA || key == GLFW_KEY_PERIOD)
            ui_automate_camera = false;
        switch (key) {
        case GLFW_KEY_UP:
            camchanged = true;
            cam.lookAt += cam.view * 0.5f;
            break;
        case GLFW_KEY_DOWN:
            camchanged = true;
            cam.lookAt -= cam.view * 0.5f;
            break;
        case GLFW_KEY_LEFT:
            camchanged = true;
            cam.lookAt -= cam.right * 0.5f;
            break;
        case GLFW_KEY_RIGHT:
            camchanged = true;
            cam.lookAt += cam.right * 0.5f;
            break;
        case GLFW_KEY_COMMA:
            camchanged = true;
            phi -= 0.1f;
            break;
        case GLFW_KEY_PERIOD:
            camchanged = true;
            phi += 0.1f;
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  if (action == GLFW_PRESS) ui_automate_camera = false;
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f; 
    forward = glm::normalize(forward);

    cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  else if (rightMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 up = cam.up;
    up.x = 0.0f;
    up.z = 0.0f;
    up = glm::normalize(up);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * up * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
