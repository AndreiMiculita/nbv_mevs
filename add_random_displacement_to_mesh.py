# import bpy
import noise
import numpy as np
from PIL import Image

# Import mesh
# mesh = bpy.import_mesh("/home/andrei/datasets/teapot.obj")

# Generate perlin noise texture
shape = (1024, 1024)
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale,
                                    j/scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=0)
# Show perlin noise
img = Image.fromarray(world * 255)
img.show()

# convert to mode "L"
img = img.convert("L")

# save image
img.save("/home/andrei/perlin_noise.png")


