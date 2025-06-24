import mss
import cv2
import numpy as np
import time
class ScreenCapture:
    def __init__(self, monitor_region=None, stack_size=4):
        self.sct = mss.mss()
        self.monitor = monitor_region
        self.stack_size = stack_size
        # self.state_stack = [np.zeros((self.monitor['height'], self.monitor['width']), dtype=np.uint8) for _ in range(stack_size)]
        self.state_stack = [np.zeros((240, 240), dtype=np.uint8) for _ in range(stack_size)]
        # 调整这两个区域坐标以匹配图像
        self.self_health_region = {'top': 826, 'left': 125, 'width': 276, 'height': 28}
        self.enemy_health_region = {'top': 826, 'left': 487, 'width': 276, 'height': 28}

        # 满血默认设为100
        self.max_health = 100
        self.prev_enemy_health = self.max_health
        self.prev_self_health = self.max_health

    def grab(self):
        img = np.array(self.sct.grab(self.monitor))  # (H, W, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img

    def get_state(self):

        new_frame = self.grab()
        new_frame = cv2.resize(new_frame, (240, 240))  # 注意: 顺序是 (width, height)
        self.state_stack.pop(0)
        self.state_stack.append(new_frame)
        return np.stack(self.state_stack, axis=0)

    def reset(self):
        blank = np.zeros((240, 240), dtype=np.uint8)
        self.state_stack = [blank for _ in range(self.stack_size)]
        self.prev_enemy_health = self.max_health
        self.prev_self_health = self.max_health

    def _extract_health_ratio(self, region):
        img = np.array(self.sct.grab(region))  # (H, W, 4)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # 调整阈值以匹配 Winkawaks 血条的实际颜色
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        current = np.sum(mask > 0)
        total = region['width'] * region['height']

        if current < 10:
            return 0.0
        else:
            return current / total

    def get_reward(self):
        cur_enemy_health = int(self._extract_health_ratio(self.enemy_health_region) * self.max_health)
        cur_self_health = int(self._extract_health_ratio(self.self_health_region) * self.max_health)

        reward = 0
        done = False
        if self.prev_enemy_health is not None and self.prev_self_health is not None:
            delta_enemy = self.prev_enemy_health - cur_enemy_health
            delta_self = self.prev_self_health - cur_self_health

            # 击打奖励
            if delta_enemy > 0:
                reward += delta_enemy * 1.0  # 敌人掉血，每点 +1
            if delta_self > 0:
                reward -= delta_self * 1.0  # 自己掉血，每点 -1

            # 无事发生惩罚（激励主动出击）
            if delta_enemy == 0 and delta_self == 0:
                reward -= 0.1  # 每帧无动作微惩罚

            # 最终奖励
            if cur_enemy_health <= 0:
                reward += 10
                done = True
            elif cur_self_health <= 0:
                reward -= 10
                done = True
        self.prev_enemy_health = cur_enemy_health
        self.prev_self_health = cur_self_health

        return reward,done


if __name__ == "__main__":
    # 1. 先确认你的 Winkawaks 窗口左上角坐标，比如 top=100, left=200
    # capture = ScreenCapture(monitor_region={'top': 776, 'left': 40, 'width': 800, 'height': 600})
    capture = ScreenCapture(monitor_region={'top': 880, 'left': 40, 'width': 800, 'height': 430})
    img = capture.grab()
    cv2.imshow("Full Screen", img)

    # 也显示一下血条部分
    self_hp = np.array(capture.sct.grab(capture.self_health_region))
    enemy_hp = np.array(capture.sct.grab(capture.enemy_health_region))

    cv2.imshow("Self HP", cv2.cvtColor(self_hp, cv2.COLOR_BGRA2BGR))
    cv2.imshow("Enemy HP", cv2.cvtColor(enemy_hp, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 2. 获取一组状态
    state = capture.get_state()  # shape: (4, 240, 320)
    print(state.shape)
    reward = capture.get_reward()
    print("当前奖励：", reward)


    # 初始化ScreenCapture对象
    # capture = ScreenCapture(monitor_region={'top': 776, 'left': 40, 'width': 800, 'height': 600})
    #
    # # 1. 获取初始状态
    # state = capture.get_state()  # 获取初始的状态堆栈
    #
    # # 2. 循环执行游戏操作和奖励计算
    # while True:
    #     # 在这里可以加入你控制游戏角色的代码，例如：
    #     # PressKey('some_key')
    #     # 或者通过强化学习模型选择动作等
    #
    #     # 获取当前的奖励
    #     reward = capture.get_reward()
    #     print("当前奖励：", reward)
    #
    #     # 3. 进行下一轮状态更新
    #     state = capture.get_state()
    #
    #     # 4. 判断是否结束游戏
    #     if reward == 100:  # 你可以根据具体逻辑，设置何时结束
    #         print("游戏胜利！")
    #         break
    #     elif reward == -100:
    #         print("游戏失败！")
    #         break
    #
    #     # 你可以加入延时，控制循环频率
    #     time.sleep(0.1)  # 可以根据实际情况调整
