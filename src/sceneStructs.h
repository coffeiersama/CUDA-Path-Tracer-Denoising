#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "utilities.h"
#include <stb_image.h>
#include "boundingbox.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define ENABLE_MIS_LIGHTING 0;
#define USE_KDTREE 1
#define SHOW_TEXTURE 1

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Vertex {
    Point3f pos;
    Normal3f normal;
    UV2f uv;
    Vertex() {};
    Vertex(Point3f p, Normal3f n) : pos(p), normal(n) { uv = UV2f(0.0); };
    Vertex(Point3f p, Normal3f n, UV2f _uv) : pos(p), normal(n), uv(_uv) {};
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Mesh
    int T_startidx;             // RECORD THE START AND END INDEX FOR VARIOUS OF MESH
    int T_endidx;
    int BoundIdx;               // BOUNDING BOX IN THE WORLD SPACE
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    int matid;                  // ADDED BY YAN
    int texid;                  // LOAD TEXTURE PART!
    int norid;

    Material() {
        color = glm::vec3(0.f);
        specular.color = glm::vec3(0.f);
        matid = -1;
        texid = -1;             // if id is -1 then not have any texture!
        emittance = 1;
        indexOfRefraction = 1.0f;
        hasReflective = hasRefractive = 0;
        specular.exponent = 0;
        norid = -1;
    }
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    float lightRadius = 1.4f;
    glm::vec3 skyRadiance = glm::vec3(0.0f);
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    bool diffuse;
    bool specular;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId;
  bool outside;
  glm::vec2 uv;
};

struct GBufferTexel {
    glm::vec3 emission = glm::vec3(0);
    glm::vec3 normal;
    glm::vec3 position;
    glm::vec3 albedo;
    glm::vec3 ialbedo;
    int geomId;
};

struct Triangle {
    int id;
    int materialId = -1;
    int geomId = -1;
    Vertex verts[3];		//	THE THREE VERTEXES OF THE TRIANGLE
    Normal3f normal;		//	THE NORMAL OF THE TRIANGLE ITS SELF	

    struct {
        Point3f maxCorner;	// TRIANGLE MAX XYZ
        Point3f minCorner;	// TRIANGLE	MIN XYZ
    } boundingbox;

    Triangle() {};

    __host__ __device__ void interpolate(glm::vec2 bary, ShadeableIntersection *isect) const {
        const float w = 1.0f - bary.x - bary.y;
        isect->uv = verts[0].uv * w + verts[1].uv * bary.x + verts[2].uv * bary.y;
        isect->surfaceNormal = glm::normalize(verts[0].normal * w + verts[1].normal * bary.x + verts[2].normal * bary.y);
    }

    __host__ __device__ Point3f calculateMidPoint() {
        return (boundingbox.maxCorner + boundingbox.minCorner) / 2.f;
    }

    __host__ __device__ void calculateMinandMax(glm::vec3 &min, glm::vec3 &max) const {
        min.x = glm::min(verts[0].pos.x, glm::min(verts[1].pos.x, verts[2].pos.x));
        min.y = glm::min(verts[0].pos.y, glm::min(verts[1].pos.y, verts[2].pos.y));
        min.z = glm::min(verts[0].pos.z, glm::min(verts[1].pos.z, verts[2].pos.z));

        max.x = glm::max(verts[0].pos.x, glm::max(verts[1].pos.x, verts[2].pos.x));
        max.y = glm::max(verts[0].pos.y, glm::max(verts[1].pos.y, verts[2].pos.y));
        max.z = glm::max(verts[0].pos.z, glm::max(verts[1].pos.z, verts[2].pos.z));
    }

    __host__ __device__ BoundingBox GetWorldBoundbox() {
        glm::vec3 min(-1.f), max(-1.f);
        calculateMinandMax(min, max);
        return BoundingBox(min, max);
    }

    __host__ __device__ float Area() {
        return glm::length(glm::cross(verts[0].pos - verts[1].pos, verts[2].pos - verts[1].pos)) * 0.5f;
    }

    __host__ __device__ bool Intersect(const Ray& r, ShadeableIntersection* isect, float maxDistance = FLT_MAX) const {
        glm::vec3 bari(0.f);
        if (glm::intersectRayTriangle(r.origin, r.direction, verts[0].pos, verts[1].pos, verts[2].pos, bari) && bari.z > 1e-5f && bari.z < maxDistance) {
            isect->t = bari.z;
            interpolate(glm::vec2(bari), isect);
            return true;
        }
        else {
            isect->t = -1.0f;
            return false;
        }
    }
};

// Intersection-only cache; shade only the final closest hit.
struct alignas(16) AccelTriangle {
    glm::vec3 position;
    int materialId;
    glm::vec3 edge1;
    int geomId;
    glm::vec3 edge2;
    int padding = 0;

    AccelTriangle() = default;
    explicit AccelTriangle(const Triangle &t) : position(t.verts[0].pos), materialId(t.materialId),
        edge1(t.verts[1].pos - t.verts[0].pos), geomId(t.geomId), edge2(t.verts[2].pos - t.verts[0].pos) {}

    __host__ __device__ bool intersect(const Ray &ray, float limit, ShadeableIntersection *hit) const {
        glm::vec3 p = glm::cross(ray.direction, edge2);
        float determinant = glm::dot(edge1, p);
        // Preserve GLM's front-face and epsilon convention.
        if (determinant < FLT_EPSILON) return false;
        float inverse = 1.0f / determinant;
        glm::vec3 delta = ray.origin - position;
        float u = inverse * glm::dot(delta, p);
        if (u < 0 || u > 1) return false;
        glm::vec3 q = glm::cross(delta, edge1);
        float v = inverse * glm::dot(ray.direction, q);
        if (v < 0 || u + v > 1) return false;
        float distance = inverse * glm::dot(edge2, q);
        if (distance <= 1e-5f || distance >= limit) return false;
        hit->t = distance;
        hit->uv = glm::vec2(u, v); // barycentrics, expanded after traversal
        return true;
    }
};

//TEXTURE
struct Texture {
    int width;
    int height;
    int components;
    unsigned char *image;
    unsigned char *dev_image;
    bool alphaTest = false;

    Texture() {
        width = -1;
        height = -1;
        components = -1;
        image = NULL;
        dev_image = NULL;
    }

    void Load(char const *filename) {
        image = stbi_load(filename, &width, &height, &components, 4);
        components = 4;
        if (image) {
            for (int p = 0; p < width * height; ++p) {
                if (image[4 * p + 3] < 128) { alphaTest = true; break; }
            }
        }
        if (image == NULL) {
            printf("FAILED TO LOAD TEXTURE!\n");
        }
        else {
            printf("SUCCESS TO LOAD TEXTURE~\n");
        }
    }

    __host__ __device__ glm::vec3 getColor(const glm::vec2& uv) const {
        int X = glm::clamp((int)(width * (uv.x - floorf(uv.x))), 0, width - 1);
        int Y = glm::clamp((int)(height * (1.0f - (uv.y - floorf(uv.y)))), 0, height - 1);
        int texel_index = Y * width + X;
        if (components >= 3 && dev_image) {
            glm::vec3 col = glm::vec3(dev_image[texel_index * components],
                dev_image[texel_index * components + 1],
                dev_image[texel_index * components + 2]);
            col = COLORDIVIDOR * col;
            //printf("texture color: %f, %f, %f", col.x, col.y, col.z);
            return col;
        }
        return glm::vec3(1.0f);
    }

    __host__ __device__ float getOpacity(const glm::vec2& uv) const {
        if (!alphaTest || components != 4 || !dev_image) return 1.0f;
        int x = glm::clamp((int)(width * (uv.x - floorf(uv.x))), 0, width - 1);
        int y = glm::clamp((int)(height * (1.0f - (uv.y - floorf(uv.y)))), 0, height - 1);
        return dev_image[(x + y * width) * components + 3] * COLORDIVIDOR;
    }
};

// LIGHT PART
enum class LightType {
    AREALIGHT,
    SPOT,
    POINT,
};

struct Light {
    enum LightType type;
    Geom geom;
    int geomIdx;
    int matIdx;
};
