## tmux
### 挂起任务
tmux new -s lry0704

### 退出tmux
Ctrl + B，然后按 D

### 重新进入
tmux attach -t lry0704

### 查看当前会话
tmux ls

### 关闭会话
输入exit或Ctrl + D

### 删除某个会话
tmux kill-session -t xx

### 一次性删所有 tmux 会话
tmux kill-server

### 上滚查看历史记录
Ctrl + B 再按 [

### 创建新窗口
Ctrl+b + %                  # 水平分屏
Ctrl+b + "                  # 垂直分屏  
Ctrl+b + 方向键              # 切换分屏
Ctrl+b + x                  # 关闭分屏
Ctrl+b + q                  # 快速显示所有面板编号（按数字后消失）
