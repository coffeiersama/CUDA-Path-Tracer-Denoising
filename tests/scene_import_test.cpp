#include "scene.h"
#include "intersections.h"
#include <cmath>
#include <cstdlib>
#include <random>

void require(bool condition, const char *message) {
    if (!condition) { std::cerr << "FAIL: " << message << std::endl; exit(1); }
}
int main(int argc, char **argv) {
    require(argc == 2, "fixture path");
    Scene scene(argv[1]);
    require(glm::length(scene.state.skyRadiance - glm::vec3(0.1f, 0.2f, 0.3f)) < 1e-6f, "scene sky radiance");
    require(scene.triangles.size() == 2 && scene.materials.size() == 2, "OBJ/MTL import counts");
    Ray ray{glm::vec3(0.25f, 0.25f, 1), glm::vec3(0, 0, -1)};
    ShadeableIntersection hit{}; hit.t = FLT_MAX;
    int index = -1;
    require(IntersectBVH(ray, &hit, index, scene.bvh_nodes, scene.triangles.data()), "positive triangle hit");
    require(index >= 0 && scene.triangles[index].geomId == 0, "ownership after BVH reorder");
    require(scene.materials[scene.triangles[index].materialId].color.r == 1, "red material assignment");
    require(glm::length(hit.surfaceNormal - glm::normalize(glm::vec3(0.25f, 0.25f, 0.5f))) < 1e-5f, "barycentric normal interpolation");
    ray.origin = glm::vec3(2.25f, 0.25f, 1); hit.t = FLT_MAX; index = -1;
    require(IntersectBVH(ray, &hit, index, scene.bvh_nodes, scene.triangles.data()), "second material hit");
    require(scene.materials[scene.triangles[index].materialId].color.g == 1, "green material assignment");
    ray.origin.z = -1; hit.t = FLT_MAX; index = -1;
    require(!IntersectBVH(ray, &hit, index, scene.bvh_nodes, scene.triangles.data()), "reject behind-origin triangles");
    unsigned char rgba[] = {255, 0, 0, 255, 0, 255, 0, 0};
    Texture texture; texture.width = 2; texture.height = 1; texture.components = 4;
    texture.dev_image = rgba; texture.alphaTest = true;
    require(texture.getColor(glm::vec2(-0.25f, 2.5f)).g == 1, "negative tiled UV and RGBA sampling");
    require(texture.getOpacity(glm::vec2(-0.25f, 2.5f)) == 0, "opacity sampling");
    puts("PASS: OBJ/MTL materials, BVH ownership, normals, ray direction, tiled RGBA and opacity");

    std::mt19937 random(42);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    std::vector<Triangle> triangles;
    for (int i = 0; i < 200; ++i) {
        Triangle t;
        glm::vec3 center(value(random), value(random), value(random));
        t.verts[0].pos = center;
        t.verts[1].pos = center + glm::vec3(0.7f, 0, 0);
        t.verts[2].pos = center + glm::vec3(0, 0.7f, 0);
        t.verts[0].normal = glm::vec3(0, 0, 1);
        t.verts[1].normal = glm::normalize(glm::vec3(0.2f, 0, 1));
        t.verts[2].normal = glm::normalize(glm::vec3(0, 0.3f, 1));
        t.verts[0].uv = glm::vec2(0.1f, 0.2f);
        t.verts[1].uv = glm::vec2(0.8f, 0.2f);
        t.verts[2].uv = glm::vec2(0.1f, 0.9f);
        t.id = i; t.geomId = 0; t.materialId = 0;
        triangles.push_back(t);
    }
    int nodesCount = 0;
    BVH_ArrNode *nodes = BuildBVHTree(nodesCount, triangles);
    std::vector<AccelTriangle> cache;
    for (const auto &t : triangles) cache.emplace_back(t);
    for (int i = 0; i < 1500; ++i) {
        Ray test{glm::vec3(value(random), value(random), 5), glm::normalize(glm::vec3(0.2f * value(random), 0.2f * value(random), -1))};
        float limit = i % 2 ? FLT_MAX : 4.0f;
        ShadeableIntersection expected{}; expected.t = limit;
        bool bruteHit = false;
        for (const auto &t : triangles) {
            ShadeableIntersection candidate;
            if (t.Intersect(test, &candidate, expected.t)) { expected = candidate; bruteHit = true; }
        }
        ShadeableIntersection actual{}; actual.t = limit;
        int found = -1;
        bool bvhHit = IntersectBVH(test, &actual, found, nodes, triangles.data(), nullptr, nullptr, false, cache.data());
        require(bvhHit == bruteHit, "bounded cached BVH agrees with brute-force visibility");
        if (bvhHit) {
            require(fabsf(actual.t - expected.t) < 1e-4f, "cached closest distance");
            require(glm::length(actual.uv - expected.uv) < 1e-5f, "cached shading UV");
            require(glm::length(actual.surfaceNormal - expected.surfaceNormal) < 1e-5f, "cached shading normal");
        }
        actual.t = limit; found = -1;
        require(IntersectBVH(test, &actual, found, nodes, triangles.data(), nullptr, nullptr, true, cache.data()) == bruteHit,
                "bounded any-hit agrees with brute-force visibility");
    }
    BoundingBox bounds(glm::vec3(0), glm::vec3(1));
    Ray parallel{glm::vec3(0, 0.5f, 2), glm::vec3(0, 0, -1)};
    glm::vec3 inverse(INFINITY, INFINITY, -1);
    require(bounds.intersectBefore(parallel, inverse, 2), "parallel ray on slab boundary");
    require(!bounds.intersectBefore(parallel, inverse, 0.5f), "box beyond finite ray segment");
    DeleteBVH(nodes);
    puts("PASS: 1500 randomized bounded closest/any-hit queries vs brute force, cache attributes, parallel slabs");
    return 0;
}
