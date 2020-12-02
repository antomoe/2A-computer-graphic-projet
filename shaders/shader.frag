#version 330 core

// fragment position and normal of the fragment, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
in vec3 w_position, w_normal;   // in world coodinates

// light dir, in world coordinates
uniform vec3 light_dir;

// material properties
uniform vec3 k_d;
uniform vec3 k_s;
uniform vec3 k_a;
uniform float s;
uniform float d_1;
uniform float d_2;
uniform float timer;

// world camera position
uniform vec3 w_camera_position;

uniform sampler2D diffuse_map;
in vec2 frag_tex_coords;
in vec4 glPosition;

out vec4 out_color;

void main() {
    float fog_maxdist = 8.0;
    float fog_mindist = 0.1;
    vec4  fog_colour = vec4(0.4, 0.4, 0.4, 1.0);
    vec3 l_1 =  normalize(light_dir);//+vec3(0,1,0)+ vec3(-1 + mod((timer/5),2), 0, 1));
    vec3 n =  normalize(w_normal);
    vec3 v = normalize(w_camera_position - w_position);
    vec3 r_1 = reflect(-l_1,n);
    float lumiere_1 = max(dot(n, l_1), 0.0);
    float time = 0;
    if(mod(timer,20)<10){
      time = mod(timer,20);
    }else{
      time = 20-mod(timer,20);
    }
    float reflection_1 = max(pow(dot(r_1,v),s)*time+10,0.0);
    vec3  rgb = texture(diffuse_map, frag_tex_coords).xyz;      // original color of the pixel
    float distance = distance(glPosition.xyz,w_position);// camera to point distance
    vec3  rayDir =   w_position - glPosition.xyz ; // camera to point vector
    float fogAmount = 1.0 - exp( -distance*0.00000007 );
    vec3  fogColor  = mix( vec3(0.5,0.6,0.9), vec3(1.0,0.9,0.7), pow(lumiere_1,8.0) );
    out_color = vec4(mix( rgb, fogColor, fogAmount ),1)* vec4(k_a + (1/(d_1*d_1))*(k_d * lumiere_1 + k_s*reflection_1) ,1);
    //out_color = texture(diffuse_map, frag_tex_coords);
}
