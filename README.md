# dqn_kof
双DQN训练拳皇

capture文件用于获取图像
dqn定义网络
main训练函数主页面
replaybuffer数据池
training_logger存储训练日志

需要安装
pip install pynput mss pygetwindow
最好先使用conda创建虚拟环境再配置环境
conda create -n rl_kof python=3.8

使用：
训练前需要先执行capture，修改需要截取的画面和双方血条位置，dqn输入为四帧，4*120*120

奖励函数你可以自己修改
