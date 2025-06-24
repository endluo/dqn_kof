# dqn_kof
双DQN训练拳皇:

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

训练前需要先执行capture，修改需要截取的画面和双方血条位置，dqn输入为四帧，(4,240,240)

奖励函数可以自己修改

截取到的血条，用于计算reward

![image](https://github.com/user-attachments/assets/102a168f-aa36-4829-8688-f0d47a8d070b)

实际用于模型训练的画面


![image](https://github.com/user-attachments/assets/c122c0de-2f53-47d1-ad82-590474068130)

有记录reward和loss变化


![reward_20250624_114013](https://github.com/user-attachments/assets/2a9de6da-112c-4971-87c7-f5275e282a07)


![loss_20250624_114013](https://github.com/user-attachments/assets/d8a4286c-50f4-4e3b-9cfa-9579e6febc33)
