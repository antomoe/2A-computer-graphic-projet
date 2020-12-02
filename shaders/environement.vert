#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

uniform mat4 model, view, projection, matrix;

// position and normal for the fragment shader, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
out vec3 w_position, w_normal;   // in world coordinates
out vec2 frag_tex_coords;
out vec4 glPosition;

void main() {
    gl_Position = projection * view * model * matrix* vec4(position, 1);
    mat3 nit_matrix = transpose(inverse(mat3(model)));
    w_normal = normalize(nit_matrix * normal);
    vec4 w_pos = model * vec4(position,1);
    w_position = w_pos.xyz / w_pos.w;
    frag_tex_coords = tex_coord;
    glPosition = gl_Position;
}
