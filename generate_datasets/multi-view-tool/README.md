## Multi-View Projections Generator

Generates projections from viewpoints regularly positioned on a sphere around an object.

This implementation renders plain images and depth images of views from viewpoints regularly distributed on a specified number rings parallel to the x-y plain, spaced vertically on a sphere around the object.

It also generates a csv file with the entropy values of the object views alongside the camera spherical coordinates.

### Usage:
Ensure you are running Python 3

Import the required libraries by running:

```
pip install -r requirements.txt
```

To run the script with default options:

```
python generate_views.py [filename]
```
filename must be a 3D object file with `.off` extension (other extensions are also compatible, check Open3D documentation).
The script will create a `out` folder with two subfolders `depth` and `image` which will contain respectively the depth images and the greyscale images of the rendered views.

Setting the flag `--csv` the script can also generate a `.csv` file with the entropy values of the different views alongside their positions. The positions are coded as `phi` and `theta` values, which stand for the degrees of rotation of the object around the x-axis and y-axis from its original position, hence simulating the camera moving on a spherical structure around the object.
