# install 

conda create --name openmmlab python=3.10 -y
conda activate openmmlab

conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

## 服务器 SSL 验证问题

pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.openmmlab.com mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

wget --no-check-certificate

## GCC版本

#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."

```bash
conda install -c conda-forge gcc gxx -y
# 设置环境变量（添加到~/.bashrc）
echo "export CC=$CONDA_PREFIX/bin/gcc" >> ~/.bashrc
echo "export CXX=$CONDA_PREFIX/bin/g++" >> ~/.bashrc
source ~/.bashrc
```
