import random
import time
from pynput.keyboard import Key, Controller, Listener
import pygetwindow as gw
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN, select_action
from capture import ScreenCapture
from replaybuffer import ReplayBuffer
from training_logger import TrainingLogger

# ========== 全局控制 ==========
keyboard = Controller()
should_run = False
exit_flag = False
start_time = None  # 添加开始计时变量

# ========== 按键操作 ==========
def PressKey(key_str):
    keyboard.press(key_str)

def ReleaseKey(key_str):
    keyboard.release(key_str)

def PressAndRelease(key_str, duration=0.05):
    PressKey(key_str)
    time.sleep(duration)
    ReleaseKey(key_str)

# ========== 窗口检测 ==========
def is_winkawaks_active():
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            title = active_window.title.lower()
            return "kawaks" in title or "slugfest" in title
    except Exception as e:
        print(f"[窗口判断异常] {e}")
    return False

# ========== 动作空间类 ==========
class ComboActionSpace:
    def __init__(self, model, target_model, capture, device, optimizer, criterion, epsilon=0.2, buffer=None):
        self.move_keys = ['w', 's', 'a', 'd']
        self.punch_keys = ['j', 'k', 'u', 'i','t','e']
        self.actions = []
        self.model = model
        self.target_model = target_model
        self.capture = capture
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.epsilon = epsilon
        self.replay_buffer = buffer or ReplayBuffer()
        self.batch_size = 32

        for k in self.move_keys + self.punch_keys:
            self.actions.append([k])

    def __len__(self):
        return len(self.actions)

    def play(self):
        state = self.capture.get_state()

        if random.random() < self.epsilon:
            action = random.randint(0, len(self.actions) - 1)
        else:
            action = select_action(self.model, state, self.device)

        return action

    def get_keys(self, action_id):
        return self.actions[action_id]

    def execute(self, action_id):
        keys = self.get_keys(action_id)
        print("出招：", keys)

        for k in keys:
            if k in self.move_keys:
                hold_time = 0.3
            elif k in self.punch_keys:
                hold_time = 0.05
            else:
                hold_time = 0.06

            PressKey(k)
            time.sleep(hold_time)
            ReleaseKey(k)

    def train(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        print("经验池已累计：", len(self.replay_buffer))
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device) / 255.0
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device) / 255.0
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        dones_tensor = torch.tensor(dones).to(self.device)

        current_q = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # ✅ Double DQN：由当前网络选动作，由目标网络评估该动作的 Q 值
        next_actions = self.model(next_states_tensor).argmax(1)
        next_q = self.target_model(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q = rewards_tensor + 0.99 * next_q * (1 - dones_tensor)

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# ========== 热键监听 ==========
def on_press(key):
    global should_run, exit_flag, start_time
    try:
        if key.char == 'r':
            should_run = not should_run
            print(f"{'▶️ 开始' if should_run else '⏸️ 暂停'}自动操作")
            if should_run:
                start_time = time.time()
        elif key.char == 'q':
            exit_flag = True
            print("❌ 退出程序")
            return False
    except AttributeError:
        if key == Key.esc:
            exit_flag = True
            print("❌ 退出程序")
            return False

# ========== 主程序入口 ==========
def main():
    logger = TrainingLogger()
    episode_reward = 0
    global start_time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    capture = ScreenCapture(monitor_region={'top': 880, 'left': 40, 'width': 800, 'height': 430})

    action_space = ComboActionSpace(
        model=model,
        target_model=target_model,
        capture=capture,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        buffer=replay_buffer
    )

    step_counter = 0
    update_target_every = 1000

    print("🎮 按 'r' 开始/暂停，按 'q' 或 'Esc' 退出程序（仅在 Winkawaks 窗口时执行）")

    listener = Listener(on_press=on_press)
    listener.start()

    while not exit_flag:
        if should_run and is_winkawaks_active():
            state = capture.get_state()
            action_id = action_space.play()
            action_space.execute(action_id)
            next_state = capture.get_state()
            reward, done = capture.get_reward()
            episode_reward += reward
            print("累计奖励：", episode_reward)

            loss = action_space.train(state, action_id, reward, next_state, done)
            print("损失函数loss:", loss)

            if loss is not None:
                logger.update_loss(loss)

            step_counter += 1
            if step_counter % update_target_every == 0:
                target_model.load_state_dict(model.state_dict())
                print("🔁 更新目标网络")

            if start_time and (time.time() - start_time > 60):
                print("⏱️ 运行超过60秒，执行F7重置")
                PressAndRelease(Key.f7)
                start_time = time.time()

            if done:
                logger.update_reward(episode_reward)
                episode_reward = 0
                print("⏱️ 一局结束重新开始")
                PressAndRelease(Key.f7)
                start_time = time.time()
        else:
            time.sleep(0.2)

    listener.join()

if __name__ == '__main__':
    main()
