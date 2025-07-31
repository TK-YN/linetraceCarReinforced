import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import time
import math
from environment.linetrace_env_node import LineTraceEnv
import rclpy
from rclpy.node import Node


# --- ニューラルネットワークモデル (EnhancedPIDPolicyNetwork) ---
class EnhancedPIDPolicyNetwork(nn.Module):
    def __init__(self, input_dim, lstm_hidden_size, output_dim):
        super(EnhancedPIDPolicyNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        #特徴抽出器: センサー入力からLSTM入力次元へ
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, lstm_hidden_size), # LSTMのinput_sizeに合わせる
            nn.ReLU()
        )
        
        #LSTM層
        #batch_first=True: (batch, seq_len, features) の入力形式
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True)
        
        # PID平均値のヘッド (出力は tanh でスケーリングされる前の生の値)
        self.pid_mean_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        #PID標準偏差のヘッド (出力は Softplus で正の値に)
        self.pid_std_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus() # 標準偏差は正である必要がある
        )
        
        #価値関数のヘッド
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, hidden_state=None):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        #特徴抽出 (各時間ステップごとに適用)
        x_flat = x.view(batch_size * seq_len, -1) # (batch_size * seq_len, input_dim)
        features = self.feature_extractor(x_flat) # (batch_size * seq_len, lstm_hidden_size)
        features = features.view(batch_size, seq_len, -1) # (batch_size, seq_len, lstm_hidden_size)

        # STMを適用
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            lstm_out, (h_n, c_n) = self.lstm(features, (h0, c0))
        else:
            lstm_out, (h_n, c_n) = self.lstm(features, hidden_state)
        
        last_hidden_state = lstm_out[:, -1, :] #(batch_size, lstm_hidden_size)
        
        #各ヘッドから出力を計算
        pid_mean_raw = self.pid_mean_head(last_hidden_state) #生のPID平均値 (スケール前)
        pid_std = self.pid_std_head(last_hidden_state) + 1e-6 #標準偏差 (Softplus済み、最小値でクリッピング)
        value = self.value_head(last_hidden_state)
        
        return pid_mean_raw, pid_std, value, (h_n, c_n)

# --- エージェント (EnhancedLineTracePIDAgent) ---
class EnhancedLineTracePIDAgent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len,
                 lr,
                 gamma,
                 gae_lambda,
                 eps_clip,
                 lstm_hidden_size,
                 min_pid=(0.0, 0.0, 0.0),
                 max_pid=(25.0, 5.0, 25.0)):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.lstm_hidden_size = lstm_hidden_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = EnhancedPIDPolicyNetwork(self.input_dim, self.lstm_hidden_size, self.output_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

        #PIDパラメータの最小値・最大値をテンソルとして保持
        self.min_pid = torch.tensor(min_pid, dtype=torch.float32).to(self.device)
        self.max_pid = torch.tensor(max_pid, dtype=torch.float32).to(self.device)
        
        self.last_pid_params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_wheel_speeds = np.array([0.0, 0.0], dtype=np.float32)
        self.curve_detected = False

        self.state_history = collections.deque(maxlen=self.seq_len)
        self.h_n = None 
        self.c_n = None 

        self.integral_error = 0.0
        self.last_error = 0.0

        self.reset_history() 

    def reset_history(self):
        """エピソード開始時に履歴とLSTMの隠れ状態をリセット"""
        self.state_history.clear()
        #LSTMの隠れ状態をゼロで初期化 (バッチサイズ1を想定)
        self.h_n = torch.zeros(1, 1, self.lstm_hidden_size).to(self.device)
        self.c_n = torch.zeros(1, 1, self.lstm_hidden_size).to(self.device)
        self.last_pid_params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_wheel_speeds = np.array([0.0, 0.0], dtype=np.float32)
        self.curve_detected = False
        self.integral_error = 0.0
        self.last_error = 0.0

    def preprocess_state(self, current_raw_state):

        #センサーデータ（輝度値）
        brightness = current_raw_state[:8]
        
        #現在の車輪速度
        current_wheel_speeds = current_raw_state[8:10]

        brightness_array = np.array(brightness)
        
        #輝度値の差分 (隣接するセンサー間の輝度差)
        brightness_diffs = []
        for i in range(len(brightness) - 1):
            diff = brightness[i] - brightness[i+1]
            brightness_diffs.append(diff)
        
        #輝度値の2次差分 (差分の差分)
        second_diffs = []
        for i in range(len(brightness_diffs) - 1):
            second_diff = brightness_diffs[i] - brightness_diffs[i+1]
            second_diffs.append(second_diff)
        
        #正規化
        normalized_brightness = brightness_array / 255.0
        MAX_RPM = 366.0
        normalized_current_wheel_speeds = np.array(current_wheel_speeds) / MAX_RPM
        
        normalized_brightness_diffs = np.array(brightness_diffs) / 255.0
        normalized_second_diffs = np.array(second_diffs) / 255.0
        
        #前回のPIDパラメータ（エージェント自身の出力履歴）
        MAX_KP, MAX_KI, MAX_KD = self.max_pid.cpu().numpy()
        #正規化は0-1の間で行う
        normalized_last_pid_params = (self.last_pid_params - self.min_pid.cpu().numpy()) / (self.max_pid.cpu().numpy() - self.min_pid.cpu().numpy() + 1e-8)
        
        #前回の車輪速度（エージェント自身の出力履歴）
        normalized_last_wheel_speeds = self.last_wheel_speeds / MAX_RPM
        
        #カーブ検出フラグ
        #左右のセンサーの偏りを使ってカーブを検出
        curve_indicator = 0.0
        if brightness[0] > 200 and max(brightness[4:]) < 50: 
            curve_indicator = 1.0
        elif brightness[7] > 200 and max(brightness[:4]) < 50:
            curve_indicator = -1.0
        
        combined_state = np.concatenate([
            normalized_brightness,
            normalized_current_wheel_speeds,
            normalized_brightness_diffs,
            normalized_second_diffs,
            normalized_last_pid_params,
            normalized_last_wheel_speeds,
            [curve_indicator]
        ]).astype(np.float32)
        
        return combined_state

    def determine_step_pid_params(self, training=True):
        #state_history が seq_len に満たない場合はパディング
        if len(self.state_history) < self.seq_len:
            processed_state_dim = self.state_history[0].shape[0] if self.state_history else self.input_dim
            padded_states = [np.zeros(processed_state_dim, dtype=np.float32) for _ in range(self.seq_len - len(self.state_history))]
            padded_states.extend(list(self.state_history))
            state_sequence_np = np.array(padded_states, dtype=np.float32)
        else:
            state_sequence_np = np.array(list(self.state_history), dtype=np.float32)
        
        #テンソルに変換し、LSTMの入力形式 (batch_size, seq_len, input_dim) に合わせる
        state_sequence_tensor = torch.from_numpy(state_sequence_np).to(self.device).unsqueeze(0) # Unsqueezeで batch_size=1 を追加

        if training:
            self.policy_net.train()
            pid_mean_raw, pid_std, value, (h_n_out, c_n_out) = self.policy_net(state_sequence_tensor, (self.h_n, self.c_n))
        else:
            self.policy_net.eval()
            with torch.no_grad():
                pid_mean_raw, pid_std, value, (h_n_out, c_n_out) = self.policy_net(state_sequence_tensor, (self.h_n, self.c_n))

        self.h_n, self.c_n = h_n_out.detach(), c_n_out.detach()

        # --- PID平均値のスケーリング ---
        #tanhを適用して出力を -1 から 1 の範囲に正規化
        pid_mean_normalized = torch.tanh(pid_mean_raw)
        
        #正規化された値を min_pid から max_pid の範囲に線形スケーリング
        pid_mean_scaled = self.min_pid + (self.max_pid - self.min_pid) * ((pid_mean_normalized + 1) / 2)

        #ガウス分布を生成
        dist = torch.distributions.Normal(pid_mean_scaled, pid_std)
        
        #分布からPIDパラメータをサンプリング
        pid_params_sampled = dist.sample()
        
        #サンプリングされたPIDパラメータを min_pid と max_pid の範囲にクリッピング
        pid_params_clipped = torch.max(torch.min(pid_params_sampled, self.max_pid), self.min_pid).squeeze(0)
        
        log_prob = dist.log_prob(pid_params_sampled).sum() 

        self.last_pid_params = pid_params_clipped.cpu().numpy()

        #LSTMへの入力として使用された整形済みの状態シーケンスと、サンプリングされたPIDパラメータ、ログ確率、価値を返す
        return pid_params_clipped.detach().cpu().numpy(), log_prob.detach(), value.detach(), state_sequence_np

    def get_wheel_speeds(self, brightness_sensors, pid_params):
        """
        PIDパラメータとセンサー値に基づいて車輪速度を計算
        """
        Kp, Ki, Kd = pid_params

        sensor_positions_mm = np.array([-35, -25, -15, -5, 5, 15, 25, 35])
        weights = np.maximum(0, 255 - np.array(brightness_sensors)) # ラインが黒、背景が白を想定
        
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            error_mm = np.sum(sensor_positions_mm * weights) / sum_weights
        else:
            error_mm = 0.0 

        normalized_error = error_mm / 35.0 

        self.integral_error += normalized_error
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0) 

        derivative_error = normalized_error - self.last_error
        self.last_error = normalized_error

        turn_output = Kp * normalized_error + Ki * self.integral_error + Kd * derivative_error

        base_rpm = 100.0 
        max_rpm = 366.0 

        left_speed = base_rpm - turn_output
        right_speed = base_rpm + turn_output

        left_speed = np.clip(left_speed, 0, max_rpm)
        right_speed = np.clip(right_speed, 0, max_rpm)

        self.last_wheel_speeds = np.array([left_speed, right_speed], dtype=np.float32)

        return [float(left_speed), float(right_speed)]

    def calculate_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def calculate_advantages(self, rewards, values, gamma=0.99, gae_lambda=0.95):
        advantages = []
        values_tensor = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        delta = 0
        for i in reversed(range(len(rewards))):
            next_value = values_tensor[i+1] if i+1 < len(values_tensor) else torch.tensor(0.0).to(self.device)
            td_error = rewards[i] + gamma * next_value - values_tensor[i]
            delta = td_error + gamma * gae_lambda * delta
            advantages.insert(0, delta)
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

# --- メインの学習ループ ---
def main(args=None):
    rclpy.init(args=args) 

    # \パラメータ設定
    input_dim = 29 #センサー8個 + 車輪速度2個 = 10個, 前処理で追加される特徴量 (8+2+7+6+3+2+1=29)
    output_dim = 3 #PID (Kp, Ki, Kd)
    seq_len = 8 #LSTMのシーケンス長 (preprocess_stateで生成される特徴量の数に依存)
    lr = 0.0003 #学習率
    gamma = 0.99 #割引率
    gae_lambda = 0.95 #GAEのラムダ
    eps_clip = 0.2 #PPOのクリッピングパラメータ
    lstm_hidden_size = 512 # LSTMの隠れ状態サイズ

    max_episodes = 2000 
    batch_size = 4 #PPO更新を行うためのエピソード数 
    ppo_epochs = 4 #PPOの最適化ステップ数

    env_node = LineTraceEnv()
    env_node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO) 

    agent = EnhancedLineTracePIDAgent(
        input_dim, output_dim, seq_len, lr, gamma, gae_lambda, eps_clip, lstm_hidden_size
    )
    optimizer = agent.optimizer

    episode_buffer = collections.deque(maxlen=batch_size * 2) # PPOバッチサイズより少し大きめ

    print("Starting training...")
    
    global_step_count = 0

    try:

        for episode in range(max_episodes):
            print(f"\n[TRAIN] === Episode {episode+1} (Global Step: {global_step_count}) ===")

            raw_state = env_node.reset() 
            if raw_state is None:
                env_node.get_logger().error("Environment reset failed. Retrying in next episode.")
                continue 
            
            agent.reset_history() 

            episode_experiences = []

            total_episode_reward = 0.0 

            max_steps_per_episode = int(env_node.max_episode_duration * 1000) 

            for step_idx in range(max_steps_per_episode):

                processed_current_state = agent.preprocess_state(raw_state)
                agent.state_history.append(processed_current_state)

                pid_params_current_step, log_prob_current_step, value_current_step, state_sequence_for_lstm = \
                    agent.determine_step_pid_params(training=True)

                #推論されたPIDパラメータで車輪速度を計算
                wheel_speeds = agent.get_wheel_speeds(raw_state[:8], pid_params_current_step)
                
                #ROS環境を1ステップ進める
                next_raw_state, reward, done = env_node.step(wheel_speeds, step_idx)
                total_episode_reward += reward 

                #このステップの経験を保存
                episode_experiences.append({
                    'state_history': state_sequence_for_lstm, # ここで整形済みの状態シーケンスを保存
                    'pid_params': pid_params_current_step,
                    'log_prob': log_prob_current_step.item(),
                    'value': value_current_step.item(),
                    'reward': reward,
                })

                raw_state = next_raw_state

                global_step_count += 1

                if done:
                    #total_episode_reward を最終報酬として直接使用
                    final_calculated_reward = total_episode_reward
                    print(f"[TRAIN] Episode {episode+1} ended at step {step_idx+1}. Total Step Reward: {total_episode_reward:.2f}, Final Calculated Reward: {final_calculated_reward:.2f}")
                    break
            else: 
                final_calculated_reward = total_episode_reward
                print(f"[TRAIN] Episode {episode+1} completed {max_steps_per_episode} steps. Total Step Reward: {total_episode_reward:.2f}, Final Calculated Reward: {final_calculated_reward:.2f}")
            
            #エピソード終了後、このエピソードの経験をバッファに追加
            episode_buffer.append(episode_experiences)

            #PPO学習の実行
            if len(episode_buffer) >= batch_size:
                print(f"[TRAIN] Starting PPO update for batch of {len(episode_buffer)} episodes...")
                try:
                    all_states = []
                    all_actions = []
                    all_old_log_probs = []
                    all_old_values = []
                    all_rewards = []

                    for ep_exp_list in episode_buffer:
                        for exp in ep_exp_list:
                            all_states.append(exp['state_history'])
                            all_actions.append(exp['pid_params'])
                            all_old_log_probs.append(exp['log_prob'])
                            all_old_values.append(exp['value'])
                            all_rewards.append(exp['reward'])

                    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32).to(agent.device)
                    actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32).to(agent.device)
                    
                    old_log_probs_tensor = torch.tensor(np.array(all_old_log_probs), dtype=torch.float32).unsqueeze(1).to(agent.device) 
                    old_values_tensor = torch.tensor(np.array(all_old_values), dtype=torch.float32).unsqueeze(1).to(agent.device)

                    all_advantages_list = []
                    all_returns_list = []
                    current_idx = 0
                    for ep_exp_list in episode_buffer:
                        rewards_per_episode = [exp['reward'] for exp in ep_exp_list]
                        values_per_episode = [exp['value'] for exp in ep_exp_list]

                        all_advantages_list.append(agent.calculate_advantages(rewards_per_episode, values_per_episode, agent.gamma, agent.gae_lambda))
                        all_returns_list.append(agent.calculate_returns(rewards_per_episode, agent.gamma))
                    
                    advantages_tensor = torch.cat(all_advantages_list).unsqueeze(1).to(agent.device)
                    returns_tensor = torch.cat(all_returns_list).unsqueeze(1).to(agent.device)


                    #GAEの正規化
                    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                    

                    #PPO epochループ
                    for ppo_epoch in range(ppo_epochs):
                        #policy_netにstates_tensorを渡す際、LSTMの隠れ状態はNoneで渡す
                        pid_mean_raw, new_pid_std, new_value_preds, _ = agent.policy_net(states_tensor, hidden_state=None)

                        #新しいPID平均値のスケーリング
                        new_pid_mean_normalized = torch.tanh(pid_mean_raw)
                        new_pid_mean_scaled = agent.min_pid + (agent.max_pid - agent.min_pid) * ((new_pid_mean_normalized + 1) / 2)

                        #新しい分布を生成
                        new_dist = torch.distributions.Normal(new_pid_mean_scaled, new_pid_std)
                        
                        #新しい行動のログ確率を計算
                        new_log_probs = new_dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)

                        #PPO損失の計算
                        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                        surr1 = ratio * advantages_tensor
                        surr2 = torch.clamp(ratio, 1.0 - agent.eps_clip, 1.0 + agent.eps_clip) * advantages_tensor
                        policy_loss = -torch.min(surr1, surr2).mean()

                        #価値損失の計算
                        value_loss = F.mse_loss(new_value_preds, returns_tensor)

                        #エントロピー損失（探索促進のため）
                        entropy_loss = -new_dist.entropy().mean()

                        #合計損失
                        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss # エントロピー項は最大化したいのでマイナス

                        #バックプロパゲーションと最適化
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5) # 勾配クリッピング
                        optimizer.step()
                    
                    print(f"[TRAIN] PPO Update Done. Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")

                except Exception as e:
                    print(f"[TRAIN] Error during PPO update: {e}")
                    import traceback
                    traceback.print_exc() #エラーのスタックトレースを出力
                finally:
                    episode_buffer.clear() #バッチ処理後、バッファをクリア

    except KeyboardInterrupt:
        print("Training interrupted by user. Shutting down.")
    finally:
        env_node.destroy_node()
        rclpy.shutdown()
        print("Training finished.")

if __name__ == "__main__":
    main()