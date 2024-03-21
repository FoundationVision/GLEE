# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 12.1
- Python 3.9.2
- Pytorch 2.1.0
- Torchvison 0.16.0

## Install environment for GLEE

```
pip3 install shapely==1.7.1
pip3 install lvis
pip3 install scipy
pip3 install fairscale
pip3 install einops 
pip3 install xformers
pip3 install tensorboard 
pip3 install opencv-python-headless 
pip3 install timm
pip3 install ftfy
pip3 install transformers==4.36.0

pip3 install -e .  
pip3 install git+https://github.com/wjf5203/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI" --user



# compile Deformable DETR
cd projects/GLEE/glee/models/pixel_decoder/ops/
python3 setup.py build install --user

```