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
import matplotlib.pyplot as plt
import signal
import sys
import os
from datetime import datetime


# --- グローバル変数（学習データ収集用） ---
training_data = {
    'episodes': [],
    'episode_rewards': [],
    'episode_lengths': [],
    'policy_losses': [],
    'value_losses': [],
    'entropy_losses': [],
    'total_losses': [],
    'average_pid_params': [],
    'moving_avg_rewards': []
}

def signal_handler(sig, frame):
    print("\nTraining interrupted! Displaying results...")
    plot_training_results()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def plot_training_results():
    if not training_data['episodes']:
        print("No training data to plot.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RL Training Results - Line Tracing PID Control', fontsize=16)
    
    #Episode Rewards
    axes[0, 0].plot(training_data['episodes'], training_data['episode_rewards'], 'b-', alpha=0.6, label='Episode Reward')
    if training_data['moving_avg_rewards']:
        axes[0, 0].plot(training_data['episodes'], training_data['moving_avg_rewards'], 'r-', linewidth=2, label='Moving Average (10 episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    #Episode Lengths
    axes[0, 1].plot(training_data['episodes'], training_data['episode_lengths'], 'g-', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length (steps)')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    #Training Losses
    if training_data['policy_losses']:
        loss_episodes = list(range(0, len(training_data['policy_losses'])))
        axes[0, 2].plot(loss_episodes, training_data['policy_losses'], 'r-', label='Policy Loss', alpha=0.8)
        axes[0, 2].plot(loss_episodes, training_data['value_losses'], 'b-', label='Value Loss', alpha=0.8)
        axes[0, 2].plot(loss_episodes, training_data['total_losses'], 'k-', label='Total Loss', alpha=0.8)
        axes[0, 2].set_xlabel('PPO Update')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Losses')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    #PID Parameters Evolution
    if training_data['average_pid_params']:
        pid_data = np.array(training_data['average_pid_params'])
        axes[1, 0].plot(training_data['episodes'], pid_data[:, 0], 'r-', label='Kp', alpha=0.8)
        axes[1, 0].plot(training_data['episodes'], pid_data[:, 1], 'g-', label='Ki', alpha=0.8)
        axes[1, 0].plot(training_data['episodes'], pid_data[:, 2], 'b-', label='Kd', alpha=0.8)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('PID Parameter Value')
        axes[1, 0].set_title('Average PID Parameters Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    #Reward Distribution
    if len(training_data['episode_rewards']) > 5:
        axes[1, 1].hist(training_data['episode_rewards'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Episode Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    #Performance Summary
    axes[1, 2].axis('off')
    if training_data['episode_rewards']:
        recent_rewards = training_data['episode_rewards'][-10:] if len(training_data['episode_rewards']) >= 10 else training_data['episode_rewards']
        summary_text = f"""
            Training Summary:
            Total Episodes: {len(training_data['episodes'])}
            Average Reward: {np.mean(training_data['episode_rewards']):.2f}
            Recent Avg (last 10): {np.mean(recent_rewards):.2f}
            Best Reward: {np.max(training_data['episode_rewards']):.2f}
            Average Episode Length: {np.mean(training_data['episode_lengths']):.1f}
            PPO Updates: {len(training_data['policy_losses'])}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    #Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_results_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training results saved as {filename}")
    
    #Show the plot
    plt.show()

def update_moving_average(new_reward, window_size=10):
    if len(training_data['episode_rewards']) < window_size:
        return np.mean(training_data['episode_rewards'])
    else:
        return np.mean(training_data['episode_rewards'][-window_size:])


# --- Neural Network Model (EnhancedPIDPolicyNetwork) ---
class EnhancedPIDPolicyNetwork(nn.Module):
    def __init__(self, input_dim, lstm_hidden_size, output_dim):
        super(EnhancedPIDPolicyNetwork, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        #特徴抽出器: センサー入力からLSTM入力次元へ
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, lstm_hidden_size),
            nn.ReLU()
        )
        
        #LSTM層
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True)
        
        #PID平均値のヘッド
        self.pid_mean_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        #PID標準偏差のヘッド
        self.pid_std_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus()
        )
        
        #価値関数のヘッド
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.size()
        
        #特徴抽出
        x_flat = x.view(batch_size * seq_len, -1)
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, seq_len, -1)

        #LSTMを適用
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            lstm_out, (h_n, c_n) = self.lstm(features, (h0, c0))
        else:
            lstm_out, (h_n, c_n) = self.lstm(features, hidden_state)
        
        last_hidden_state = lstm_out[:, -1, :]
        
        #各ヘッドから出力を計算
        pid_mean_raw = self.pid_mean_head(last_hidden_state)
        pid_std = self.pid_std_head(last_hidden_state) + 1e-6
        value = self.value_head(last_hidden_state)
        
        return pid_mean_raw, pid_std, value, (h_n, c_n)


# --- Agent (EnhancedLineTracePIDAgent) ---
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

        #エピソード中のPIDパラメータ追跡用
        self.episode_pid_params = []

        self.reset_history() 

    def reset_history(self):
        """エピソード開始時に履歴とLSTMの隠れ状態をリセット"""
        self.state_history.clear()
        self.h_n = torch.zeros(1, 1, self.lstm_hidden_size).to(self.device)
        self.c_n = torch.zeros(1, 1, self.lstm_hidden_size).to(self.device)
        self.last_pid_params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_wheel_speeds = np.array([0.0, 0.0], dtype=np.float32)
        self.curve_detected = False
        self.integral_error = 0.0
        self.last_error = 0.0
        self.episode_pid_params = []

    def preprocess_state(self, current_raw_state):
        #センサーデータ（輝度値）
        brightness = current_raw_state[:8]
        
        #現在の車輪速度
        current_wheel_speeds = current_raw_state[8:10]

        brightness_array = np.array(brightness)
        
        #輝度値の差分（隣接するセンサー間の輝度差）
        brightness_diffs = []
        for i in range(len(brightness) - 1):
            diff = brightness[i] - brightness[i+1]
            brightness_diffs.append(diff)
        
        #輝度値の2次差分（差分の差分）
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
        
        #前回のPIDパラメータ
        normalized_last_pid_params = (self.last_pid_params - self.min_pid.cpu().numpy()) / (self.max_pid.cpu().numpy() - self.min_pid.cpu().numpy() + 1e-8)
        
        #前回の車輪速度
        normalized_last_wheel_speeds = self.last_wheel_speeds / MAX_RPM
        
        #カーブ検出
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
        
        state_sequence_tensor = torch.from_numpy(state_sequence_np).to(self.device).unsqueeze(0)

        if training:
            self.policy_net.train()
            pid_mean_raw, pid_std, value, (h_n_out, c_n_out) = self.policy_net(state_sequence_tensor, (self.h_n, self.c_n))
        else:
            self.policy_net.eval()
            with torch.no_grad():
                pid_mean_raw, pid_std, value, (h_n_out, c_n_out) = self.policy_net(state_sequence_tensor, (self.h_n, self.c_n))

        self.h_n, self.c_n = h_n_out.detach(), c_n_out.detach()

        # --- PID平均値のスケーリング ---
        pid_mean_normalized = torch.tanh(pid_mean_raw)
        pid_mean_scaled = self.min_pid + (self.max_pid - self.min_pid) * ((pid_mean_normalized + 1) / 2)

        #ガウス分布を生成
        dist = torch.distributions.Normal(pid_mean_scaled, pid_std)
        
        #分布からPIDパラメータをサンプリング
        pid_params_sampled = dist.sample()
        
        #サンプリングされたPIDパラメータをクリップ
        pid_params_clipped = torch.max(torch.min(pid_params_sampled, self.max_pid), self.min_pid).squeeze(0)
        
        log_prob = dist.log_prob(pid_params_sampled).sum() 

        self.last_pid_params = pid_params_clipped.cpu().numpy()
        
        #このエピソードのPIDパラメータを保存
        self.episode_pid_params.append(self.last_pid_params.copy())

        return pid_params_clipped.detach().cpu().numpy(), log_prob.detach(), value.detach(), state_sequence_np

    def get_wheel_speeds(self, brightness_sensors, pid_params):
        """
        PIDパラメータとセンサー値に基づいて車輪速度を計算
        """
        Kp, Ki, Kd = pid_params

        sensor_positions_mm = np.array([-35, -25, -15, -5, 5, 15, 25, 35])
        weights = np.maximum(0, 255 - np.array(brightness_sensors))
        
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

    #パラメータ設定
    input_dim = 29
    output_dim = 3
    seq_len = 8
    lr = 0.0003
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    lstm_hidden_size = 512

    max_episodes = 2000 
    batch_size = 4
    ppo_epochs = 4

    env_node = LineTraceEnv()
    env_node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO) 

    agent = EnhancedLineTracePIDAgent(
        input_dim, output_dim, seq_len, lr, gamma, gae_lambda, eps_clip, lstm_hidden_size
    )
    optimizer = agent.optimizer

    episode_buffer = collections.deque(maxlen=batch_size * 2)

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

                wheel_speeds = agent.get_wheel_speeds(raw_state[:8], pid_params_current_step)
                
                next_raw_state, reward, done = env_node.step(wheel_speeds, step_idx)
                total_episode_reward += reward 

                episode_experiences.append({
                    'state_history': state_sequence_for_lstm,
                    'pid_params': pid_params_current_step,
                    'log_prob': log_prob_current_step.item(),
                    'value': value_current_step.item(),
                    'reward': reward,
                })

                raw_state = next_raw_state
                global_step_count += 1

                if done:
                    final_calculated_reward = total_episode_reward
                    print(f"[TRAIN] Episode {episode+1} ended at step {step_idx+1}. Total Step Reward: {total_episode_reward:.2f}, Final Calculated Reward: {final_calculated_reward:.2f}")
                    break
            else: 
                final_calculated_reward = total_episode_reward
                print(f"[TRAIN] Episode {episode+1} completed {max_steps_per_episode} steps. Total Step Reward: {total_episode_reward:.2f}, Final Calculated Reward: {final_calculated_reward:.2f}")
            
            #学習データの可視化用に保存
            training_data['episodes'].append(episode + 1)
            training_data['episode_rewards'].append(total_episode_reward)
            training_data['episode_lengths'].append(len(episode_experiences))
            
            #このエピソードの平均PIDパラメータを計算
            if agent.episode_pid_params:
                avg_pid = np.mean(agent.episode_pid_params, axis=0)
                training_data['average_pid_params'].append(avg_pid)
            
            #移動平均を更新
            moving_avg = update_moving_average(total_episode_reward)
            training_data['moving_avg_rewards'].append(moving_avg)
            
            episode_buffer.append(episode_experiences)

            #PPO学習
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
                    epoch_policy_losses = []
                    epoch_value_losses = []
                    epoch_entropy_losses = []
                    epoch_total_losses = []
                    
                    for ppo_epoch in range(ppo_epochs):
                        pid_mean_raw, new_pid_std, new_value_preds, _ = agent.policy_net(states_tensor, hidden_state=None)

                        new_pid_mean_normalized = torch.tanh(pid_mean_raw)
                        new_pid_mean_scaled = agent.min_pid + (agent.max_pid - agent.min_pid) * ((new_pid_mean_normalized + 1) / 2)

                        new_dist = torch.distributions.Normal(new_pid_mean_scaled, new_pid_std)
                        new_log_probs = new_dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)

                        #PPO損失の計算
                        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                        surr1 = ratio * advantages_tensor
                        surr2 = torch.clamp(ratio, 1.0 - agent.eps_clip, 1.0 + agent.eps_clip) * advantages_tensor
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(new_value_preds, returns_tensor)
                        entropy_loss = -new_dist.entropy().mean()
                        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
                        optimizer.step()
                        
                        #損失値を保存
                        epoch_policy_losses.append(policy_loss.item())
                        epoch_value_losses.append(value_loss.item())
                        epoch_entropy_losses.append(entropy_loss.item())
                        epoch_total_losses.append(total_loss.item())
                    
                    #このPPO更新の平均損失を保存
                    training_data['policy_losses'].append(np.mean(epoch_policy_losses))
                    training_data['value_losses'].append(np.mean(epoch_value_losses))
                    training_data['entropy_losses'].append(np.mean(epoch_entropy_losses))
                    training_data['total_losses'].append(np.mean(epoch_total_losses))
                    
                    print(f"[TRAIN] PPO Update Done. Policy Loss: {np.mean(epoch_policy_losses):.4f}, Value Loss: {np.mean(epoch_value_losses):.4f}, Entropy: {np.mean(epoch_entropy_losses):.4f}")

                except Exception as e:
                    print(f"[TRAIN] Error during PPO update: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    episode_buffer.clear()
            
            #50エピソードごとに中間結果を表示
            if (episode + 1) % 50 == 0:
                print(f"\n[INFO] Showing training progress at episode {episode + 1}")
                plot_training_results()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nTraining stopped due to error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        #常に最終結果を表示
        print("\nDisplaying final training results...")
        plot_training_results()
        
        env_node.destroy_node()
        rclpy.shutdown()
        print("Training finished.")


if __name__ == "__main__":
    main()