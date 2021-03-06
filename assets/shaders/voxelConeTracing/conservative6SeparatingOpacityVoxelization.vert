#version 430

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_bitangent;
layout(location = 4) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_modelIT;

out Vertex
{
    vec3 normalW;
    vec2 uv;
};

void main()
{
    gl_Position = u_model * vec4(in_pos, 1.0);
    normalW = (u_modelIT * vec4(in_normal, 0.0)).xyz;
    uv = in_uv;
}
