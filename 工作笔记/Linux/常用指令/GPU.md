# 常用命令

## 显存管理

### 查看哪些进程占用了 GPU

fuser /dev/nvidia*

### 杀死所有正在使用 NVIDIA GPU 设备的进程

fuser -k /dev/nvidia* 

### 显存占用查看

nvidia-smi
