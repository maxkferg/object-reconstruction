
# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Simple demonstration of Mesh visual.
"""

import numpy as np
from vispy import scene
from vispy import app, gloo, visuals
from vispy.gloo import Program
from vispy.geometry import create_sphere
from vispy.visuals.transforms import (STTransform, MatrixTransform,
                                      ChainTransform)



vertex = """
    uniform   mat4 u_model;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = u_model * vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    } """

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    } """


def checkerboard(grid_num=8, grid_size=32):
    row_even = grid_num // 2 * [0, 1]
    row_odd = grid_num // 2 * [1, 0]
    Z = np.row_stack(grid_num // 2 * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)




class Canvas(app.Canvas):

    def __init__(self, vertices, normals, faces):
        app.Canvas.__init__(self, keys='interactive', size=(800, 550))

        self.meshes = []
        self.grids = []
        self.rotation = MatrixTransform()

        vertices = vertices / 10

        nv = vertices.size//3
        vcolor = np.ones((nv, 4), dtype=np.float32)
        vcolor[:, 0] = np.linspace(1, 0, nv)
        vcolor[:, 1] = 0.2#np.linspace(1, 0, nv)
        vcolor[:, 2] = np.linspace(0, 1, nv)

        mesh = visuals.MeshVisual(vertices, faces, vcolor, shading='smooth')#, face_colors=fcolor)
        mesh.shininess = 0.01
        mesh.cmap = 'viridis'
        self.meshes.append(mesh)

        grid = visuals.GridLinesVisual(color=(0 , 0, 0, 0.5))
        self.grids.append(grid)


        self.model = np.eye(4, dtype=np.float32)
        self.program = Program(vertex, fragment, count=4)
        self.program['position'] = [(6, 1), (-1, 1),
                                    (1, -1), (-1, -1)]
        self.program['texcoord'] = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.program['texture'] = checkerboard()
        self.program['u_model'] = self.model


        # Mesh with color indexed into a colormap
        #verts = mdata.get_vertices(None)
        #faces = mdata.get_faces()
        #values = rng.randn(len(verts))
        #mesh = visuals.MeshVisual(vertices=verts, faces=faces,
        #                          vertex_values=values, shading='smooth')
        #mesh.clim = [-1, 1]
        #mesh.cmap = 'viridis'
        #mesh.shininess = 0.01
        #self.meshes.append(mesh)

        # Lay out meshes in a grid
        grid = (3, 3)
        s = 300. / max(grid)
        for i, mesh in enumerate(self.meshes):
            x = 800. * (i % grid[0]) / grid[0] + 400. / grid[0] - 2
            y = 800. * (i // grid[1]) / grid[1] + 400. / grid[1] + 2
            transform = ChainTransform([STTransform(translate=(x, y),
                                                    scale=(s, s, s)),
                                        self.rotation])
            mesh.transform = transform
            mesh.transforms.scene_transform = STTransform(scale=(1, 1, 0.01))

        self.show()

        #self.rotation.rotate(-20, (0, 0, 2))
        self.timer = app.Timer(connect=self.rotate)
        self.timer.start(0.016)

    def rotate(self, event):
        self.rotation.rotate(0.4, (0, 1, 0))
        #self.rotation.rotate(0.3 ** 0.5, (0, 1, 0))
        #self.rotation.rotate(0.5 ** 0.5, (0, 0, 1))
        self.update()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        for mesh in self.meshes:
            mesh.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, ev):
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.clear(color='black', depth=True)
        self.program.draw('triangle_strip')

        for grid in self.grids:
            grid.draw()

        for mesh in self.meshes:
            mesh.draw()

    def run(self):
        app.run()


if __name__ == '__main__':
    win = Canvas()
    import sys
    if sys.flags.interactive != 1:
        app.run()