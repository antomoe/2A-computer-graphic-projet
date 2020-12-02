#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
import sys
import time

from itertools import cycle
from bisect import bisect_left      # search sorted keyframe lists
from PIL import Image
import random
import math

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader
import random

from transform import *
from viewer import *

import glob
from ctypes import *
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *

def load_cubemap(cubemap_folder):
    id = glGenTextures(1)
    faces = ["right.jpg", "left.jpg", "top.jpg", "bottom.jpg", "back.jpg", "front.jpg"]

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, id)

    for i, face in enumerate(faces):
        img = np.asarray(Image.open(cubemap_folder + "/" + face).convert('RGBA'))
        width, height = img.shape[1], img.shape[0]
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA,
            width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    print("Cubemap loaded, id : " + str(id))
    return id

SKYBOX_VERTICES = np.array((
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    ( 1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0),

    (-1.0, -1.0,  1.0),
    (-1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0,  1.0,  1.0),
    (-1.0, -1.0,  1.0),

     (1.0, -1.0, -1.0),
     (1.0, -1.0,  1.0),
     (1.0,  1.0,  1.0),
     (1.0,  1.0,  1.0),
     (1.0,  1.0, -1.0),
     (1.0, -1.0, -1.0),

    (-1.0, -1.0,  1.0),
    (-1.0,  1.0,  1.0),
     (1.0,  1.0,  1.0),
     (1.0,  1.0,  1.0),
     (1.0, -1.0,  1.0),
    (-1.0, -1.0,  1.0),

    (-1.0,  1.0, -1.0),
     (1.0,  1.0, -1.0),
     (1.0,  1.0,  1.0),
     (1.0,  1.0,  1.0),
    (-1.0,  1.0,  1.0),
    (-1.0,  1.0, -1.0),

    (-1.0, -1.0, -1.0),
    (-1.0, -1.0,  1.0),
     (1.0, -1.0, -1.0),
     (1.0, -1.0, -1.0),
    (-1.0, -1.0,  1.0),
     (1.0, -1.0,  1.0)
), 'f')

class Skybox:
    def __init__(self, shader, cubemap_folder):
        self.shader = shader
        self.cubemap = cubemap_folder
        self.cubemap_tex = load_cubemap(cubemap_folder)

        names = ['view', 'projection']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}

    def draw(self, projection, view, model):
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)

        skybox_VAO = VertexArray([SKYBOX_VERTICES])
        glBindVertexArray(skybox_VAO.glid)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap_tex);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        glDepthMask(GL_TRUE)
