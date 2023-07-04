# Finding the next best view for object recognition through Maximum Entropy Viewpoint Selection

A collection of scripts related to my Master's thesis - a method for finding the most informative camera positions for multiview object recognition.
There are two methods: one based on differentiable rendering and one based on point clouds.
They are both implemented in PyTorch.

## Installation
There are some packages that can only be installed through `pip` and some that can only be installed through `conda`.
I would have liked to use `conda` exclusively, but the `neural_renderer` is only on `PyPI`.

* `conda create --name <env>`
* `conda activate <env>`
* `pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html` (some version that has the [entr](https://pytorch.org/docs/master/special.html) function)
* `pip install -r requirements_pip.txt`
* `conda install --file requirements.txt`

## Usage
The `classification_pipeline.py` script in the `pipeline` dir is the main script.
Given the paths of the object mesh, the ckpt files and the desired method, it will run the pipeline for the given method.
For more info, run:

```bash
python3 pipeline/classification_pipeline.py --help
```

Data is not tracked and can be obtained by running the scripts in the `generate_datasets` dir.
You can get more info by running each script with the `--help` flag, e.g.:
```bash
python3 generate_datasets/generate_view_dataset.py --help
```
Note that you need ModelNet40 downloaded and extracted.
You can get it [here](https://modelnet.cs.princeton.edu/).

You might need to prepend `PYTHONPATH=.` to the commands for the imports to work.

## Visualization
The figures in the report can be generated using the scripts in the `visualization` dir.


## License
[MIT](https://choosealicense.com/licenses/mit/)