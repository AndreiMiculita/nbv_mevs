# Finding the Next Best View for Object Recognition through Maximum Entropy Viewpoint Selection

A collection of scripts related to my Master's thesis - a method for finding the most informative camera positions for multiview object recognition.

**Thesis Report PDF: https://drive.google.com/file/d/1bxV0k1IZEmBeeNDRbrTXAjw1fXbQj_C8/view** (soon to be published at https://fse.studenttheses.ub.rug.nl/31411/)

There are two methods: one based on differentiable rendering and one based on point clouds (see thesis report for details).
They are both implemented in PyTorch.

## Setup 🧑‍🔧

1. Create a conda environment:
   ```bash
   conda create --name nbv_mevs_env python=3.8
   conda activate nbv_mevs_env
   ```

2. Install PyTorch (the important part is to use some version that has the [entr](https://pytorch.org/docs/master/special.html) function):
   ```bash
   pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
   ```

3. Install other dependencies using pip and conda (I would have preferred to use only conda, but the `neural_renderer` is not available there):
   ```bash
   pip install -r requirements_pip.txt
   conda install --file requirements.txt
   ```

You might need a separate conda environment for the method based on point clouds, since it was developed in a different version of PyTorch. 
For that, check out the `am/thesis` branch on my fork of [PointNet2_PyTorch](https://github.com/AndreiMiculita/Pointnet2_PyTorch/tree/am/thesis).
The pipeline script (below) will work on either environment.
The graph triangulation script `build_graph_from_spherical_coords` (which uses `stripy`) might also need a separate environment.

## Usage 🧑‍💻
The main script is `classification_pipeline` in the `pipeline` directory.
Given the paths of the object mesh, the checkpoint files and the desired method, it will run the pipeline for the given method.
For more info, run:

```bash
python3 pipeline/classification_pipeline.py --help
```

The script `evaluate_pipeline` was used to run the pipeline on the entire test set.

Datasets are not tracked and can be obtained by running the scripts in the `generate_datasets` directory.
You can get more info by running each script with the `--help` flag, e.g.:
```bash
python3 generate_datasets/generate_view_dataset.py --help
```
Note that you need ModelNet10 downloaded and extracted, (`classification_pipeline` assumes in `~/datasets/ModelNet10`).
You can get it [here](https://modelnet.cs.princeton.edu/).

When training, caching is used (with `lmdb`) to speed up loading data.
However, the cache database files can get quite large for images (in my experience, 10-20 times the size of a png dataset), so make sure you have plenty of disk space.

You might need to prepend `PYTHONPATH=.` to the commands for the imports to work.

## Visualization 📊
The figures in the thesis can be generated using the scripts in the `visualization` directory.
The output will be saved in the `assets` directory.

![](assets/entropy_views_40_animated.gif)

## License
[MIT License](https://choosealicense.com/licenses/mit/)
