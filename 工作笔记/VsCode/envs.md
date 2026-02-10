# envs
## 环境配置
### 以 install .e 安装的包无法识别
需要手动指定 extraPaths 
1. 打开项目根目录下的 .vscode/settings.json。
2. 添加以下配置：
```json
{
    "python.analysis.extraPaths": [
        "/data1/liruoyu/MyProjects/Myclassify"
    ]
}
```
