#include <iostream>
#include "scene.h"
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION 

Scene::Scene(string filename) {
    const auto loadStart = std::chrono::steady_clock::now();
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);	

    file_name = filename;
    current_Triangle_id = 0;

    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw std::runtime_error("Cannot open scene file: " + filename);
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

#ifdef USE_KDTREE
    Node_count = 0;
    bvh_nodes = BuildBVHTree(Node_count, triangles);
    cout << "BVH has " << Node_count << " nodes" << endl << endl;
#endif 
    loadLight();
    cout << "Scene ready: " << triangles.size() << " triangles, " << materials.size()
         << " materials, " << textures.size() << " textures, "
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count()
         << " seconds" << endl;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        bool loadingmesh = false;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
                loadingmesh = true;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        for (int i = 0; i < 3; i++) {
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        //THE CURRENT THINGS IS MESH
        if (loadingmesh) {
            utilityCore::safeGetline(fp_in, line);
            loadMesh(line, newGeom, newGeom.transform, newGeom.invTranspose);
        }

        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            utilityCore::safeGetline(fp_in, line);
        }
        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy = 45.0f;

    //load static properties
    for (int i = 0; i < 3; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera_position_default_ = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            camera.position = camera_position_default_;
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera_lookat_default_ = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            camera.lookAt = camera_lookat_default_;
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera_up_default_ = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            camera.up = camera_up_default_;
        } else if (tokens[0] == "LIGHTRADIUS") {
            state.lightRadius = std::stof(tokens[1]);
        } else if (tokens[0] == "SKY") {
            state.skyRadiance = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
      camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
                              , 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }

        // load extra information (Texture)
        newMaterial.texid = -1;
        newMaterial.matid = id;
        string line;
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                string texturePath = (std::filesystem::path(file_name).parent_path() / "Textures" / tokens[1]).string();
                if (!std::filesystem::exists(texturePath)) {
                    texturePath = (std::filesystem::path(file_name).parent_path() / tokens[1]).string();
                }
                Texture newT;
                newT.Load(texturePath.c_str());
                if (!newT.image) throw std::runtime_error("Cannot load texture: " + texturePath);
                newMaterial.texid = textures.size();
                textures.push_back(newT);
            }
            utilityCore::safeGetline(fp_in, line);
        }

        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::loadMesh(string objPath, Geom& newGeom, const glm::mat4& transform, const glm::mat4& invTranspose) {
    objPath = (std::filesystem::path(file_name).parent_path() / "Models" / objPath).string();
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;

    cout << "obj file: " << objPath << endl;
    const string materialDirectory = std::filesystem::path(objPath).parent_path().string() + "/";
    std::string errors = tinyobj::LoadObj(shapes, objMaterials, objPath.c_str(), materialDirectory.c_str());
    std::cout << errors << std::endl;

    if (errors.size() != 0) {
        throw std::runtime_error("OBJ import failed: " + objPath + ": " + errors);
    }

    // material -1 opts into OBJ/MTL materials; explicit scene IDs still override.
    const bool importMaterials = newGeom.materialid == -1;
    const int materialBase = (int)this->materials.size();
    std::map<string, int> textureCache;
    if (importMaterials) {
        for (const auto &m : objMaterials) {
            Material material;
            material.color = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
            material.specular.color = glm::vec3(m.specular[0], m.specular[1], m.specular[2]);
            material.specular.exponent = m.shininess;
            material.emittance = glm::max(m.emission[0], glm::max(m.emission[1], m.emission[2]));
            if (material.emittance > 0) material.color = glm::vec3(m.emission[0], m.emission[1], m.emission[2]) / material.emittance;
            material.matid = (int)this->materials.size();
            if (!m.diffuse_texname.empty()) {
                string textureName = m.diffuse_texname;
                std::replace(textureName.begin(), textureName.end(), '\\', '/');
                string texturePath = (std::filesystem::path(materialDirectory) / textureName).lexically_normal().string();
                auto found = textureCache.find(texturePath);
                if (found == textureCache.end()) {
                    Texture texture;
                    texture.Load(texturePath.c_str());
                    if (!texture.image) throw std::runtime_error("Cannot load OBJ texture: " + texturePath);
                    material.texid = (int)textures.size();
                    textureCache[texturePath] = material.texid;
                    textures.push_back(texture);
                } else material.texid = found->second;
            }
            this->materials.push_back(material);
        }
        if (objMaterials.empty()) throw std::runtime_error("No MTL materials loaded: " + objPath);
    }

    newGeom.BoundIdx = BoudningBoxs.size();
    newGeom.T_startidx = triangles.size();

    float maxx_total = -FLT_MAX, maxy_total = -FLT_MAX, maxz_total = -FLT_MAX;
    float minx_total = FLT_MAX, miny_total = FLT_MAX, minz_total = FLT_MAX;

    for (int i = 0; i < shapes.size(); i++) {
        std::vector<float> &p = shapes[i].mesh.positions;
        std::vector<float> &n = shapes[i].mesh.normals;
        std::vector<float> &uv = shapes[i].mesh.texcoords;
        std::vector<unsigned int> &ind = shapes[i].mesh.indices;

        for (int j = 0; j < ind.size(); j += 3) {
            glm::vec3 p1 = glm::vec3(p[ind[j] * 3], p[ind[j] * 3 + 1], p[ind[j] * 3 + 2]);
            glm::vec3 p2 = glm::vec3(p[ind[j + 1] * 3], p[ind[j + 1] * 3 + 1], p[ind[j + 1] * 3 + 2]);
            glm::vec3 p3 = glm::vec3(p[ind[j + 2] * 3], p[ind[j + 2] * 3 + 1], p[ind[j + 2] * 3 + 2]);

            glm::vec3 worldp1 = glm::vec3(transform * glm::vec4(p1, 1));
            glm::vec3 worldp2 = glm::vec3(transform * glm::vec4(p2, 1));
            glm::vec3 worldp3 = glm::vec3(transform * glm::vec4(p3, 1));

            Triangle newt;
            newt.geomId = (int)geoms.size();
            int localMaterial = j / 3 < shapes[i].mesh.material_ids.size() ? shapes[i].mesh.material_ids[j / 3] : -1;
            newt.materialId = importMaterials ? materialBase + (localMaterial >= 0 ? localMaterial : 0) : newGeom.materialid;
            newt.verts[0].pos = worldp1;
            newt.verts[1].pos = worldp2;
            newt.verts[2].pos = worldp3;

            glm::vec3 min_t(0.f), max_t(0.f);
            utilityCore::compareThreeVertex(worldp1, worldp2, worldp3, min_t, max_t);

            minx_total = glm::min(minx_total, min_t.x);
            miny_total = glm::min(miny_total, min_t.y);
            minz_total = glm::min(minz_total, min_t.z);
            maxx_total = glm::max(maxx_total, max_t.x);
            maxy_total = glm::max(maxy_total, max_t.y);
            maxz_total = glm::max(maxz_total, max_t.z);

            if (n.size() >= p.size()) {
                glm::vec3 n1 = glm::vec3(n[ind[j] * 3], n[ind[j] * 3 + 1], n[ind[j] * 3 + 2]);
                glm::vec3 n2 = glm::vec3(n[ind[j + 1] * 3], n[ind[j + 1] * 3 + 1], n[ind[j + 1] * 3 + 2]);
                glm::vec3 n3 = glm::vec3(n[ind[j + 2] * 3], n[ind[j + 2] * 3 + 1], n[ind[j + 2] * 3 + 2]);

                glm::vec4 worldn1 = invTranspose * glm::vec4(n1, 0.0);
                glm::vec4 worldn2 = invTranspose * glm::vec4(n2, 0.0);
                glm::vec4 worldn3 = invTranspose * glm::vec4(n3, 0.0);

                newt.verts[0].normal = glm::normalize(glm::vec3(worldn1));
                newt.verts[1].normal = glm::normalize(glm::vec3(worldn2));
                newt.verts[2].normal = glm::normalize(glm::vec3(worldn3));
            } else {
                const glm::vec3 normal = glm::normalize(glm::cross(worldp2 - worldp1, worldp3 - worldp1));
                for (auto &v : newt.verts) v.normal = normal;
            }

            if (uv.size() >= p.size() / 3 * 2) {
                newt.verts[0].uv = glm::vec2(uv[ind[j] * 2], uv[ind[j] * 2 + 1]);
                newt.verts[1].uv = glm::vec2(uv[ind[j + 1] * 2], uv[ind[j + 1] * 2 + 1]);
                newt.verts[2].uv = glm::vec2(uv[ind[j + 2] * 2], uv[ind[j + 2] * 2 + 1]);
            }
            else for (auto &v : newt.verts) v.uv = glm::vec2(0);
            newt.id = (int)triangles.size();
            triangles.push_back(newt);
        }
    }
    BoudningBoxs.push_back(BoundingBox(glm::vec3(minx_total, miny_total, minz_total),
        glm::vec3(maxx_total, maxy_total, maxz_total)));
    newGeom.T_endidx = triangles.size();
}

void Scene::loadLight() {
    for (int i = 0; i < geoms.size(); i++) {				//	GO THROUGHT ALL GEMOS!
        if (geoms[i].materialid >= 0 && materials[geoms[i].materialid].emittance > 0) {
            Light light;
            light.geomIdx = i;
            light.matIdx = geoms[i].materialid;
            light.type = LightType::AREALIGHT;				//	AT PRESENT THERE ARE ONLY AREALIGHTS
            light.geom = geoms[i];
            lights.push_back(light);
        }
    }
}
