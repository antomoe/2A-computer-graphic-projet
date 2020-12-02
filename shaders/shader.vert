#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;
layout(location = 3) in vec4 bone_ids;
layout(location = 4) in vec4 bone_weights;

uniform mat4 model, view, projection, matrix;

const int MAX_VERTEX_BONES=4, MAX_BONES=128;
uniform mat4 boneMatrix[MAX_BONES];

// position and normal for the fragment shader, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
out vec3 w_position, w_normal;   // in world coordinates
out vec2 frag_tex_coords;
out vec4 glPosition;

void main() {
    mat4 skinMatrix = mat4(0);
    for (int b=0; b < MAX_VERTEX_BONES; b++){
        skinMatrix +=  bone_weights[b] * boneMatrix[int(bone_ids[b])];
    }
    vec4 wPosition4 = skinMatrix * vec4(position, 1.0);
    gl_Position = projection * view * matrix * wPosition4;
    mat3 nit_matrix = transpose(inverse(mat3(model)));
    w_normal = normalize(nit_matrix * normal);
    vec4 w_pos = model * vec4(position,1);
    w_position = w_pos.xyz / w_pos.w;
    frag_tex_coords = tex_coord;
    glPosition = gl_Position;
}
