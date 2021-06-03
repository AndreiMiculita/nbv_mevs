At the moment, a collection of scripts (not written by me, but maybe modified by me) related to my Master's thesis - a method for finding the most informative camera positions for multiview object recognition.

|Script | Source|
|---|----|
|`multi-view-tool` | https://github.com/tparisotto/multi-view-tool|
|`soft_renderer` | https://github.com/craigleili/3DLocalMultiViewDesc|
|`saliency_map.py` | https://github.com/utkuozbulak/pytorch-cnn-visualizations/|
|`neural_renderer` | https://github.com/ZhengZerong/neural_renderer (fork of https://github.com/daniilidis-group/neural_renderer) |


To run `find_highest_entropy.py`:
* `pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html` (some version that has the [entr](https://pytorch.org/docs/master/special.html) function)
* `cd neural renderer`
* `pip install .`  
* `pip install tqdm scikit-image`