#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys
import random
import math

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader

from transform import *
from PIL import Image               # load images for textures
from itertools import cycle

from bisect import bisect_left      # search sorted keyframe lists
from skybox import *

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers.append(GL.glGenBuffers(1))
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)

MAX_VERTEX_BONES = 4
MAX_BONES = 128
# ------------  Scene object classes ------------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, children=(), transform=identity()):
        self.transform = transform
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model):
        """ Recursive draw, passing down updated model matrix. """
        for child in self.children:
            child.draw(projection, view, model@ self.transform)  # TODO: hierarchical update?

    def key_handler(self, key):
        """ Dispatch keyboard events to children """
        for child in self.children:
            if hasattr(child, 'key_handler'):
                child.key_handler(key)

# --------------Animation -------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        # if time <= self.times[0] or time >= self.times[-1]:
        #     return self.values[0 if time <= self.times[0] else -1]

        # 2. search for closest index entry in self.times, using bisect_left function
        i = bisect_left(self.times, time%self.times[-1]) # _i is the time index just before t

        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = (time%self.times[-1]-self.times[i-1]) / (self.times[i]-self.times[i-1])
        return self.interpolate(self.values[i-1], self.values[i], fraction)

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate = KeyFrames(translate_keys)
        self.rotate = KeyFrames(rotate_keys,  quaternion_slerp)
        self.scale = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_value = translate(self.translate.value(time))
        rotate_value = quaternion_matrix(self.rotate.value(time))
        scale_value = scale(self.scale.value(time))
        interpolate = translate_value @ rotate_value @scale_value
        return interpolate

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        super().__init__()
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model)

# -------- Skinning Control for Keyframing Skinning Mesh Bone Transforms ------
class SkinningControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, *keys, transform=identity()):
        super().__init__(transform=transform)
        self.keyframes = TransformKeyFrames(*keys) if keys[0] else None
        self.world_transform = identity()

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        if self.keyframes:  # no keyframe update should happens if no keyframes
            self.transform = self.keyframes.value(glfw.get_time())

        # store world transform for skinned meshes using this node as bone
        self.world_transform = model @ self.transform

        # default node behaviour (call children's draw method)
        super().draw(projection, view, model)

# mesh to refactor all previous classes
class Mesh:

    def __init__(self, shader, attributes, bone_nodes, bone_offsets,index=None):
        self.shader = shader
        names = ['view', 'projection', 'model', 'matrix']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray(attributes, index)
        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

    def draw(self, projection, view, model, matrix,  primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)
        GL.glUniformMatrix4fv(self.loc['matrix'], 1, True, matrix)

        if self.bone_nodes is not None and self.bone_offsets is not None:
            for bone_id, node in enumerate(self.bone_nodes):
                bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

                bone_loc = GL.glGetUniformLocation(self.shader.glid, 'boneMatrix[%d]' % bone_id)
                GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)
        # GL.glUseProgram(0)

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            # GL.glDisable(GL.GL_CULL_FACE)
            tex = np.asarray(Image.open(file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

class MixedMesh(Mesh):
    """ Simple first textured object """

    def __init__(self, shader, diffuse_map, attributes, matrix, bone_nodes, bone_offsets,index=None,alpha = 1,
                    light_dir=(0, -1, 0),   # directionnal light (in world coords)
                    k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):

        super().__init__(shader, attributes, bone_nodes, bone_offsets,index)

        self.light_dir = light_dir
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s
        self.matrix = matrix
        self.alpha = alpha

        # retrieve OpenGL locations of shader variables at initialization
        names = ['light_dir', 'k_a', 's', 'k_s', 'k_d', 'w_camera_position', 'timer', 'd_1', 'd_2', 'diffuse_map', 'alpha']

        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        # loc.put(GL.glGetUniformLocation(shader.glid, 'diffuse_map'))
        self.loc.update(loc)
        # setup texture and upload it to GPU
        self.texture = diffuse_map


    def key_handler(self, key):
        # some interactive elements
        if key == glfw.KEY_F6:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        if key == glfw.KEY_F7:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # setup light parameters
        GL.glUniform3fv(self.loc['light_dir'], 1, self.light_dir)

        # setup material parameters
        GL.glUniform3fv(self.loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))

        # world camera position for Phong illumination specular component
        w_camera_position = np.linalg.inv(model[0:3, 0:3]).T  #(0, 0, 0)   # TODO: to update
        GL.glUniform3fv(self.loc['w_camera_position'], 1, w_camera_position)

        GL.glUniform1f(self.loc['timer'], glfw.get_time())
        GL.glUniform1f(self.loc['d_1'], 3.0)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)

        GL.glUniform1f(self.loc['alpha'], self.alpha)

        # model = model @self.matrix
        super().draw(projection, view, model, self.matrix, primitives)


def load_mixed_squelette(file, shader,  light_dir, matrix , tex_file=None, alpha = 1):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []
    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.mTime / ticks_per_second: key.mValue for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.mAnimations:
        anim = scene.mAnimations[0]
        for channel in anim.mChannels:
            # for each animation bone, store TRS dict with {times: transforms}
            transform_keyframes[channel.mNodeName] = (
                conv(channel.mPositionKeys, anim.mTicksPerSecond),
                conv(channel.mRotationKeys, anim.mTicksPerSecond),
                conv(channel.mScalingKeys, anim.mTicksPerSecond)
            )
    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}                                       # nodes name -> node lookup
    nodes_per_mesh_id = [[] for _ in scene.mMeshes]  # nodes holding a mesh_id

    def make_nodes(assimp_node):
        """ Recursively builds nodes for our graph, matching assimp nodes """
        trs_keyframes = transform_keyframes.get(assimp_node.mName, (None,))
        skin_node = SkinningControlNode(*trs_keyframes,
                                        transform=assimp_node.mTransformation)
        nodes[assimp_node.mName] = skin_node
        for mesh_index in assimp_node.mMeshes:
            nodes_per_mesh_id[mesh_index].append(skin_node)
        skin_node.add(*(make_nodes(child) for child in assimp_node.mChildren))
        return skin_node

    root_node = make_nodes(scene.mRootNode)
    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            # Textures[mesh_name]=(Texture(file=tex_file))
            mat.properties['diffuse_map'] = Texture(file=tex_file)
        # tex_file = None
    # ---- create SkinnedMesh objects
    meshes = []
    for mesh_id, mesh in enumerate(scene.mMeshes):
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)]*MAX_BONES] * mesh.mNumVertices,
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.mBones[:MAX_BONES]):
            for entry in bone.mWeights:  # weight,id pairs necessary for sorting
                v_bone[entry.mVertexId][bone_id] = (entry.mWeight, bone_id)

        v_bone.sort(order='weight')             # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.mName] for bone in mesh.mBones]
        bone_offsets = [bone.mOffsetMatrix for bone in mesh.mBones]

        # initialize skinned mesh and store in assimp mesh for node addition
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        # name = os.path.basename(mat['NAME'])
        attrib = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0], v_bone['id'], v_bone['weight']]
        # mat['diffuse_map']
        mesh = MixedMesh(shader, mat['diffuse_map'], attrib, matrix, bone_nodes, bone_offsets,mesh.mFaces,alpha,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir)
        for node in nodes_per_mesh_id[mesh_id]:
            node.add(mesh)
        # meshes.append(mesh)
    # meshes.append(root_node)
    nb_triangles = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (scene.mNumMeshes, nb_triangles, len(nodes), scene.mNumAnimations))
    return [root_node]
    # return meshes

def load_mixed(file, shader, light_dir, matrix, tex_file=None, alpha = 1):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    # path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    # index = 0
    # Textures = {}
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
            # mesh_name = os.path.basename(mat.properties['NAME'])
        if tex_file:
            # Textures[mesh_name]=(Texture(file=tex_file))
            mat.properties['diffuse_map'] = Texture(file=tex_file)
        # tex_file = None

    # ---- create SkinnedMesh objects
    meshes = []
    for mesh in scene.mMeshes:
        # initialize skinned mesh and store in assimp mesh for node addition
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        # name = os.path.basename(mat['NAME'])
        attrib = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0]]
        # mat['diffuse_map']
        bone_nodes=None
        bone_offsets=None
        mesh = MixedMesh(shader, mat['diffuse_map'], attrib, matrix, bone_nodes, bone_offsets, mesh.mFaces,alpha,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir)
        meshes.append(mesh)
    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes



# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=1280, height=960):
        super().__init__()

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)


        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_window_size_callback(self.win, self.on_size)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glDepthFunc(GL.GL_LEQUAL)           # used to render the skybox
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)
        GL.glEnable(GL.GL_BLEND)           # enable blending
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        self.WorldTransform = identity()

    def on_size(self, win, width, height):
        """ window size update => update viewport to new framebuffer size """
        GL.glViewport(0, 0, *glfw.get_framebuffer_size(win))

    def run(self,transform = identity()):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()@self.WorldTransform
            projection = self.trackball.projection_matrix(win_size)

            # draw our scene objects
            self.draw(projection, view@transform, identity())


            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """

        m = 10000 # intensité du mouvement

        if action == glfw.PRESS or action == glfw.REPEAT:
            if action == glfw.PRESS or action == glfw.REPEAT:
                if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                    glfw.set_window_should_close(self.win, True)
                if key == glfw.KEY_W:
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
                if key == glfw.KEY_LEFT:
                    self.WorldTransform = self.WorldTransform@translate(vec(m,0,0))
                if key == glfw.KEY_RIGHT:
                    self.WorldTransform = self.WorldTransform@translate(vec(-m,0,0))
                if key == glfw.KEY_UP:
                    self.WorldTransform = self.WorldTransform@translate(vec(0,-m,0))
                if key == glfw.KEY_DOWN:
                    self.WorldTransform= self.WorldTransform@translate(vec(0,m,0))
                if key == glfw.KEY_S:
                    self.WorldTransform = self.WorldTransform@scale(1.1)
                if key == glfw.KEY_F:
                    self.WorldTransform = self.WorldTransform@scale(0.9)
                if key == glfw.KEY_D:
                    self.WorldTransform= self.WorldTransform@rotate(vec(1,0,0),1)
                if key == glfw.KEY_C:
                    self.WorldTransform= self.WorldTransform@rotate(vec(1,0,0),-1)
                if key == glfw.KEY_X:
                    self.WorldTransform= self.WorldTransform@rotate(vec(0,1,0),1)
                if key == glfw.KEY_V:
                    self.WorldTransform= self.WorldTransform@rotate(vec(0,1,0),-1)

                self.key_handler(key)




class RotationControlNode(Node):
    def __init__(self, key_xup, key_xdown, key_yup, key_ydown,key_zup, key_zdown,key_up, key_down,x=0, y=0, z=0,angle = 0):
        super().__init__()
        self.x, self.y, self.z = x, y, z
        self.angle = angle
        self.key_xup, self.key_xdown = key_xup, key_xdown
        self.key_yup, self.key_ydown = key_yup, key_ydown
        self.key_zup, self.key_zdown = key_zup, key_zdown
        self.key_up, self.key_down = key_up, key_down

    def key_handler(self, key):
        self.x += 1 * int(key == self.key_xup)
        self.x -= 1 * int(key == self.key_xdown)
        self.y += 1 * int(key == self.key_yup)
        self.y -= 1 * int(key == self.key_ydown)
        self.z += 1 * int(key == self.key_zup)
        self.z -= 1 * int(key == self.key_zdown)
        self.angle += 1 * int(key == self.key_up)
        self.angle -= 1 * int(key == self.key_down)
        self.transform = translate(self.x, self.y,self.z)@rotate(vec(0,1,0),self.angle)
        super().key_handler(key)


# -------------- main program and scene setup --------------------------------
def load_fish(viewer, shader, light_dir):
    list_obj = ["Fish/Barracuda/Barracuda2anim.obj", "Fish/BlueStarfish/BluieStarfish.obj",
                "Fish/BlueTang/BlueTang.obj","Fish/BottlenoseDolphin/BottlenoseDolphin.obj",
                "Fish/ClownFish2/ClownFish2.obj", "Fish/GiantGrouper/GiantGrouper.obj",
                "Fish/LionFish/LionFish.obj", "Fish/NurseShark/NurseShark.obj",
                "Fish/ReefFish0/ReefFish0.obj","Fish/ReefFish3/ReefFish3.obj",
                "Fish/ReefFish4/ReefFish4.obj", "Fish/ReefFish5/ReefFish5.obj",
                "Fish/ReefFish7/ReefFish7.obj", "Fish/ReefFish8/reeffish8.obj",
                "Fish/ReefFish12/ReefFish12.obj", "Fish/ReefFish14/reeffish14.obj",
                "Fish/ReefFish16/ReefFish16.obj", "Fish/ReefFish17/reeffish17.obj",
                "Fish/ReefFish20/reeffish20.obj", "Fish/SeaHorse/SeaHorse.obj",
                "Fish/SeaSnake/seasnake.obj","Fish/TinyYellowFish/TinyYellowfish.obj",
                "Fish/WhaleShark/WhaleShark.obj", "Fish/YellowTang/yellowtang.obj"]

    list_fbx = ["Fish/Barracuda/Barracuda2anim.fbx", "Fish/BlueStarfish/BluieStarfish.fbx",
                "Fish/BlueTang/BlueTang.fbx","Fish/BottlenoseDolphin/BottleNoseDolphin.fbx",
                "Fish/ClownFish2/ClownFish2.fbx", "Fish/GiantGrouper/GiantGrouper.fbx",
                "Fish/LionFish/LionFish.fbx", "Fish/NurseShark/NurseShark.fbx",
                "Fish/ReefFish0/ReefFish0.fbx","Fish/ReefFish3/ReefFish3.fbx",
                "Fish/ReefFish4/ReefFish4.fbx", "Fish/ReefFish5/ReefFish5.fbx",
                "Fish/ReefFish7/ReefFish7.fbx", "Fish/ReefFish8/reeffish8.fbx",
                "Fish/ReefFish12/ReefFish12.fbx", "Fish/ReefFish14/reeffish14.fbx",
                "Fish/ReefFish16/ReefFish16.fbx", "Fish/ReefFish17/reeffish17.fbx",
                "Fish/ReefFish20/reeffish20.fbx", "Fish/SeaHorse/SeaHorse.fbx",
                "Fish/SeaSnake/seasnake.fbx","Fish/TinyYellowFish/TinyYellowfish.fbx",
                "Fish/WhaleShark/WhaleShark.fbx", "Fish/YellowTang/yellowtang.fbx"]

    list_png = ["Fish/Barracuda/Barracuda_Base Color.png",
                "Fish/BlueStarfish/BlueStarfish_Base_Color.png",
                "Fish/BlueTang/BlueTang_Base_Color.png",
                "Fish/BottlenoseDolphin/BottlenoseDolphin_Base_Color.png",
                "Fish/ClownFish2/ClownFish2_Base_Color.png",
                "Fish/GiantGrouper/GiantGrouper_Base_Color.png",
                "Fish/LionFish/LionFish_Base_Color.png",
                "Fish/NurseShark/NurseShark_Base_Color.png",
                "Fish/ReefFish0/reefFish0_Base_Color.png",
                "Fish/ReefFish3/ReefFish3_Base_Color.png",
                "Fish/ReefFish4/reefFish4_Base_Color.png",
                "Fish/ReefFish5/ReefFish5_Base_Color.png",
                "Fish/ReefFish7/ReefFish7_Base_Color.png",
                "Fish/ReefFish8/reefFish8_Base_Color.png",
                "Fish/ReefFish12/reefFish12_Base_Color.png",
                "Fish/ReefFish14/ReefFish14_Base_Color.png",
                "Fish/ReefFish16/ReefFish16_Base_Color.png",
                "Fish/ReefFish17/ReefFish17_Base_Color.png",
                "Fish/ReefFish20/ReefFish20_Base_Color.png",
                "Fish/SeaHorse/SeaHorse_Base_Color.png",
                "Fish/SeaSnake/SeaSnake_Base_Color.png",
                "Fish/TinyYellowFish/TinyYellowFish_Base_Color.png",
                "Fish/WhaleShark/WhaleShark_Base_Color.png",
                "Fish/YellowTang/YellowTang_Base_Color.png"]
    for i in range(0, len(list_obj)):
        time = random.randint(10, 20)

        ratio = random.randint(10,30)
        limit = random.randint(10,50)
        translate_valeur_x = random.randint(-limit, limit) *ratio
        limit = random.randint(10,50)
        translate_valeur_y = random.randint(-limit, limit)*ratio
        limit = random.randint(10,50)
        translate_valeur_z = random.randint(-limit, limit) *ratio
        distance = random.randint(100,500)
        distance = distance *100
        distance_ = random.randint(-distance,distance)
        rotate_valeur = random.randint(0, 70)
        angle = random.randint(15, 25)

        rand = random.randint(0,2)

        signe = 1
        angle_inital = 0
        if list_fbx[i]=="Fish/SeaHorse/SeaHorse.fbx" or list_fbx[i]=="Fish/SeaSnake/seasnake.fbx":
            rand = 0
        if rand == 0 :

            if translate_valeur_x < 0 :
                signe = -1
                distance = -1*distance

            translate_keys = {0: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z),
                            2*time: vec(translate_valeur_x+ distance,translate_valeur_y+distance_ , translate_valeur_z+0.8*distance_),
                            5*time: vec(translate_valeur_x + 3*distance,translate_valeur_y+2*distance_ , translate_valeur_z),
                            8*time: vec(translate_valeur_x+ distance,translate_valeur_y+0.6*distance_ , translate_valeur_z),
                            10*time: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z),}
            rotate_keys = {0:  quaternion_from_axis_angle(vec(0,signe,0), degrees=90+angle, radians=None),
                            2*time-1: quaternion_from_axis_angle(vec(0,signe,0), degrees=90+2*angle, radians=None),
                           2*time: quaternion_from_axis_angle(vec(0,signe,0), degrees=90+rotate_valeur, radians=None),
                           5*time-1 : quaternion_from_axis_angle(vec(0,signe,0), degrees=90+rotate_valeur, radians=None),
                           5*time : quaternion_from_axis_angle(vec(0,-signe,0), degrees=90, radians=None),
                           8*time-1 : quaternion_from_axis_angle(vec(0,-signe,0), degrees=90+angle, radians=None),
                           8*time : quaternion_from_axis_angle(vec(0,-signe,0), degrees=90+rotate_valeur, radians=None),
                           10*time-1:  quaternion_from_axis_angle(vec(0,-signe,0), degrees=90, radians=None),
                           10*time:  quaternion_from_axis_angle(vec(0,signe,0), degrees=90+angle, radians=None)}

        if rand == 1 :
            if translate_valeur_y < 0 :
                signe = -1
                distance = -1*distance


            translate_keys = {0: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z),
                            2*time: vec(translate_valeur_x+distance_,translate_valeur_y + distance, translate_valeur_z-0.7*distance_),
                            5*time: vec(translate_valeur_x+2*distance_ ,translate_valeur_y + 3*distance, translate_valeur_z),
                            8*time: vec(translate_valeur_x-0.5*distance_,translate_valeur_y + distance, translate_valeur_z),
                            10*time: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z),}

            rotate_keys = {0:  quaternion_from_axis_angle(vec(-signe,0,0), degrees=90+angle, radians=None),
                            2*time-1 : quaternion_from_axis_angle(vec(-signe,0,0), degrees=90, radians=None),
                           2*time : quaternion_from_axis_angle(vec(-signe,0,0), degrees=90+2*angle, radians=None),
                           5*time-1 : quaternion_from_axis_angle(vec(-signe,0,0), degrees=90+rotate_valeur, radians=None),
                           5*time : quaternion_from_axis_angle(vec(signe,0,0), degrees=90, radians=None),
                           8*time-1 : quaternion_from_axis_angle(vec(signe,0,0), degrees=90, radians=None),
                           8*time : quaternion_from_axis_angle(vec(signe,0,0), degrees=90+rotate_valeur, radians=None),
                           10*time-1:  quaternion_from_axis_angle(vec(signe,0,0), degrees=90, radians=None),
                           10*time:  quaternion_from_axis_angle(vec(-signe,0,0), degrees=90+angle, radians=None)}


        if rand == 2 :
            if translate_valeur_z < 0 :
                angle_inital = 180
                distance = -1*distance
            translate_keys = {0: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z),
                            2*time: vec(translate_valeur_x+distance_,translate_valeur_y+distance_ , translate_valeur_z+ distance),
                            5*time: vec(translate_valeur_x+2*distance_,translate_valeur_y+0.5*distance , translate_valeur_z+ 3*distance),
                            8*time: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z+ distance),
                            10*time: vec(translate_valeur_x,translate_valeur_y , translate_valeur_z)}
            rotate_keys = {0:  quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital+angle, radians=None),
                            2*time-1 : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital, radians=None),
                           2*time : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital+rotate_valeur, radians=None),
                           5*time-1 : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital+rotate_valeur, radians=None),
                           5*time : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital-180, radians=None),
                           8*time-1 : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital-180, radians=None),
                           8*time : quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital-180 + rotate_valeur, radians=None),
                           10*time-1:  quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital-180 + rotate_valeur, radians=None),
                           10*time:  quaternion_from_axis_angle(vec(0,1,0), degrees=angle_inital+angle, radians=None)}

        s = 3
        scale_keys = {0: s, 5*time: s, 10*time: s}
        matrix = rotate(vec(1, 0, 0), 0) @ rotate(vec(0, 1, 0),  0) @ scale(100,100,100) @ translate(translate_valeur_x,translate_valeur_y,translate_valeur_z)

        keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
        try:
            keynode.add(*[mesh for mesh in load_mixed_squelette(
                "obj/"+list_fbx[i], shader,light_dir, matrix, "obj/"+list_png[i])])
        except:
            print("Error in Fish loading")

        viewer.add(keynode)

def load_special_fish(viewer, shader, light_dir):
    list_obj = ["Shark002DAE/Correctshark002.dae"]
    list_png = ["Shark002DAE/Sharktexture002.png"]

    for i in range(0, len(list_obj)):

        matrix = rotate(vec(1, 0, 0), 0) @ rotate(vec(0, 1, 0),  0) @ scale(100000,100000,100000) @ translate(0,0,0)
        node = RotationControlNode(glfw.KEY_U, glfw.KEY_R, glfw.KEY_T, glfw.KEY_Y,glfw.KEY_I, glfw.KEY_K,glfw.KEY_J, glfw.KEY_L)
        node.add(*[mesh for mesh in load_mixed_squelette(
            "obj/"+list_obj[i], shader, light_dir, matrix, "obj/"+list_png[i])])
        viewer.add(node)

def load_environnement(viewer, shader, light_dir):

    # viewer.add(*[m for m in load_mixed("obj/grass/allGrass_001.obj", shader, light_dir, matrix)])
    matrix = rotate(vec(1, 0, 0), 90) @ rotate(vec(0, 1, 0),  45) @ scale(800000,800000,800000 ) @ translate(12,-35,10)
    viewer.add(*[m for m in load_mixed("obj/submarine/submarine.obj", shader, light_dir, matrix)])

    s = 100000
    matrix = rotate(vec(1, 0, 0), 0) @ rotate(vec(0, 1, 0),  0) @ scale(s,0.90*s,s) @ translate(0,-100,0)
    viewer.add(*[mesh for mesh in load_mixed(
        "obj/Terrain/Models_OBJ/Terrain_300000.obj", shader, light_dir, matrix, "obj/Terrain/Textures/Light_Map.tif")])
    load_bubble(viewer, shader, light_dir)

def load_bubble(viewer, shader, light_dir):
    for i in range(0, 10):
        time = random.randint(10, 20)
        translate_valeur_x = random.randint(0,10)
        translate_valeur_z = random.randint(0,10)
        translate_keys = {0: vec(translate_valeur_x ,0 ,translate_valeur_z), time: vec(translate_valeur_x, 1000000 , translate_valeur_z)}
        rotate_keys = {0:  quaternion(), time: quaternion()}
        scale_keys = {0:  vec(1,1,1), time: vec(1,1,1)}
        matrix = rotate(vec(1, 0, 0), 0) @ rotate(vec(0, 1, 0),  0) @ scale(10000,10000,10000) @ translate(translate_valeur_x,100,350)
        keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
        keynode.add(*[mesh for mesh in load_mixed(
            "obj/bubble/only_quad_sphere.obj", shader,light_dir, matrix, alpha =0.4)])
        viewer.add(keynode)


def main():
    """ create a window, add scene objects, then run rendering loop """
    #viewer = Viewer(1800, 900)
    viewer = Viewer()
    # default color shader
    shader = Shader("shaders/shader.vert", "shaders/shader.frag")
    environement_shader = Shader("shaders/environement.vert", "shaders/environement.frag")
    skybox_shader = Shader("shaders/skybox.vert", "shaders/skybox.frag")
    skybox = Skybox(skybox_shader, "obj/cubemap")
    viewer.add(skybox)

    light_dir = (0, 500, 0)

    load_fish(viewer, shader, light_dir)
    load_special_fish(viewer, shader,  light_dir)
    load_environnement(viewer, environement_shader, light_dir)

    # start rendering loop
    viewer.run()

if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
