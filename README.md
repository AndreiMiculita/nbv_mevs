# Finding the next best view for object recognition through Maximum Entropy Viewpoint Selection

A collection of scripts related to my Master's thesis - a method for finding the most informative camera positions for multiview object recognition.

Some of the scripts are based on the following repositories:

|Script | Source|
|---|----|
|`multi-view-tool` | https://github.com/tparisotto/multi-view-tool|
|`soft_renderer` | https://github.com/craigleili/3DLocalMultiViewDesc|
|`saliency_map.py` | https://github.com/utkuozbulak/pytorch-cnn-visualizations/|
|`neural_renderer` | https://github.com/ZhengZerong/neural_renderer (fork of https://github.com/daniilidis-group/neural_renderer) |


There are some packages that can only be installed through `pip` and some that can only be installed through `conda`. I would have liked to use only `conda`, but the `neural_renderer` package is only available through `pip`.

To create the environment:
* `conda create --name <env>`
* `conda activate <env>`
* `pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html` (some version that has the [entr](https://pytorch.org/docs/master/special.html) function)
* `pip install -r requirements_pip.txt`
* `conda install --file requirements.txt`