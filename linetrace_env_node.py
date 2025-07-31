#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose
import time
import math
import numpy as np

class LineTraceEnv(Node):
    def __init__(self):
        super().__init__('linetrace_env')

        # --- サブスクライバー ---
        self.brightness_sub = self.create_subscription(
            Float64MultiArray, '/brightness_array', self.brightness_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # --- パブリッシャー ---
        self.right_wheel_pub = self.create_publisher(
            Float64MultiArray, '/right_wheel_velocity_controller/commands', 10)
        self.left_wheel_pub = self.create_publisher(
            Float64MultiArray, '/left_wheel_velocity_controller/commands', 10)

        # --- 内部状態変数 ---
        self.brightness = [0.0] * 8
        self.joint_states = JointState()
        self.current_robot_pose = Pose() 

        self.robot_name = 'linetrace_robot'
        self.set_pose_client = self.create_client(SetEntityPose, '/world/empty/set_pose')

        # --- 初期ポーズ設定 ---
        self.initial_pose = Pose()
        self.initial_pose.position.x = -0.22
        self.initial_pose.position.y = -0.332
        self.initial_pose.position.z = 0.02
        self.initial_pose.orientation.z = 0.7071
        self.initial_pose.orientation.w = 0.7071

        """
        # --- 目標位置設定 ---
        self.target_position = {
            'x': -0.25,
            'y': -0.325,
            'z': 0.02
        }
        self.target_radius = 0.35
        """

        # --- エピソード管理用変数 ---
        self.episode_start_time = 0.0
        self.max_episode_duration = 15.0
        self.current_episode_step = 0 #エピソード内のステップ数をカウント
        
        # --- 報酬計算用の履歴変数 (reset()で初期化される) ---
        self.trajectory_data = []
        self.initial_distance_to_target = 0.0
        self.line_out_flag = False
        
        self.get_logger().info('LineTrace Environment Node Started (RL Step Mode)')

        self.get_logger().info('Waiting for /world/empty/set_pose service during init (timeout 30s)...')
        if not self.set_pose_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().fatal('/world/empty/set_pose service not available. Exiting.')
            raise RuntimeError("SetEntityPose service not available.")
        self.get_logger().info('/world/empty/set_pose service is available.')


        self.model_states_sub = self.create_subscription(
            JointState, # geometry_msgs.msg.PoseStamped または gazebo_msgs.msg.ModelStates が適切
            '/model/linetrace_robot/pose', # 例: ROS 2 Humble/Iron with ign_ros2_control
            self.model_states_callback, 10
        )
        self.get_logger().info(f"Subscribing to model states on /model/linetrace_robot/pose") # デバッグ情報

    def model_states_callback(self, msg: Pose): 
        self.current_robot_pose = msg

    def brightness_callback(self, msg):
        self.brightness = msg.data

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def get_robot_position(self):
        """
        ロボットの現在位置を取得する関数
        model_states_callbackで更新された self.current_robot_pose を使用する
        """
        return self.current_robot_pose

    def calculate_distance_to_target(self, position):
        dx = position.position.x - self.target_position['x']
        dy = position.position.y - self.target_position['y']
        return math.sqrt(dx * dx + dy * dy)

    def is_at_target(self, position):
        distance = self.calculate_distance_to_target(position)
        return distance <= self.target_radius

    def wait_for_sensor_update(self, timeout_sec=0.1):
        """
        センサー値が更新されるまで待機する関数。
        単一ステップ環境では、rclpy.spin_onceを1回呼ぶだけで十分な場合が多い。
        """
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def step(self, action, stp):
        """
        環境を1ステップ進める。
        action: [right_wheel_velocity, left_wheel_velocity] (RPM)
        """
        right_wheel_speed = float(action[0])
        left_wheel_speed = float(action[1])

        right_msg = Float64MultiArray(data=[right_wheel_speed])
        left_msg = Float64MultiArray(data=[left_wheel_speed])
        self.right_wheel_pub.publish(right_msg)
        self.left_wheel_pub.publish(left_msg)

        simulation_step_duration = 0.001 
        self.wait_for_sensor_update(timeout_sec=0.1) 

        #更新された状態を観測
        next_state = self.observe_state() 
        current_position = self.get_robot_position() 

        self.current_episode_step += 1
        elapsed_time = (self.current_episode_step * simulation_step_duration) 

        reward = 0.0
        
        #ライン中心維持報酬
        line_pos = self._calculate_current_line_position(next_state[:8])
        line_deviation_penalty = (abs(line_pos) / 35.0) ** 2 * 2
        reward += (1.0 - line_deviation_penalty) 
        #self.get_logger().info(f'{next_state[0]}')

        #前進速度報酬
        target_speed = 100.0
        current_avg_speed = (abs(next_state[8]) + abs(next_state[9])) / 2.0
        speed_reward = abs(current_avg_speed - target_speed) * 0.01 
        reward += speed_reward / 1000 if elapsed_time > 1.0 else 0.005

        done = False
        if all(b > 220 for b in next_state[:8]): # ラインアウト
            self.get_logger().info(f'Current position_x: {current_position.position.x:.2f} , position_y: {current_position.position.y:.2f}')
            reward -= 50.0 
            self.line_out_flag = True
            done = True
            self.get_logger().info('Line completely lost. Big penalty applied.')
            if elapsed_time < 3.0:
                early_out_penalty = -50.0
                reward += early_out_penalty
                self.get_logger().info(f'Line out occurred early (<3s). Additional penalty: {early_out_penalty}')
        elif elapsed_time >= self.max_episode_duration: # 最大時間に達した
            done = True
            reward += 300
            self.get_logger().info(f'Max episode duration ({self.max_episode_duration}s) reached. Reward: {reward:.2f}')

        """
        elif self.is_at_target(current_position): # 目標到達
            reward += 1000.0 
            done = True
            self.get_logger().info('Target reached! Big bonus applied.')
        elif abs(next_state[8]) < 5 and abs(next_state[9]) < 5 and elapsed_time > 5.0 and self.current_episode_step > 50:
            reward -= 200.0 
            self.get_logger().info('Robot stalled. Penalty applied.')
            done = True
        """
        
        #エピソードが終了したら、ロボットを停止
        if done:
            stop_msg = Float64MultiArray(data=[0.0])
            self.right_wheel_pub.publish(stop_msg)
            self.left_wheel_pub.publish(stop_msg)

        #軌跡データに追加 (このデータは毎ステップの報酬計算には使わず、最終的な分析用とする)
        trajectory_point = {
            'time': elapsed_time,
            'position': current_position,
            'brightness': next_state[:8],
            'wheel_speeds': next_state[8:10]
        }
        self.trajectory_data.append(trajectory_point)

        return next_state, reward, done

    #calculate_line_position を _calculate_current_line_position として追加
    def _calculate_current_line_position(self, brightness):
        sensor_positions = np.array([-35, -25, -15, -5, 5, 15, 25, 35])
        weights = np.maximum(0, 220 - np.array(brightness))
        total_weight = np.sum(weights)
        if total_weight > 0:
            line_position = np.sum(sensor_positions * weights) / total_weight
        else:
            line_position = 0.0 
        return line_position

    def observe_state(self):
        wheel_speeds = list(self.joint_states.velocity)[:2] if self.joint_states.velocity else [0.0, 0.0]
        brightness_data = list(self.brightness) if self.brightness else [0.0] * 8
        return brightness_data + wheel_speeds
    
    def check_done_conditions(self, state, position, elapsed_time):
        #目標地点に到達した場合
        if self.is_at_target(position):
            self.get_logger().info(f'Target reached! Distance: {self.calculate_distance_to_target(position):.3f}m')
            return True
        
        #最大時間に達した場合
        if elapsed_time >= self.max_episode_duration:
            self.get_logger().info(f'Max episode duration ({self.max_episode_duration}s) reached')
            return True
        

        if all(b > 220 for b in state[:8]): 
            self.get_logger().info('Line completely lost (all sensors high)')
            self.line_out_flag = True # ペナルティ用フラグ
            return True
        
        #ロボットが停止しすぎている場合
        current_right_rpm = abs(state[-2]) if len(state) > 8 else 0
        current_left_rpm = abs(state[-1]) if len(state) > 9 else 0
        #5RPM以下が5秒以上続く場合
        if current_right_rpm < 5 and current_left_rpm < 5 and elapsed_time > 5.0 and self.current_episode_step > 50:
            self.get_logger().info('Robot stalled for too long')
            return True
        
        return False

    def calculate_episode_reward(self, final_position, episode_duration):
        """
        エピソード全体の報酬を計算する関数
        この関数は `done=True` になったときに一度だけ呼び出されるべきです。
        """
        reward = 0.0
        
        final_distance = self.calculate_distance_to_target(final_position)
        if self.is_at_target(final_position):
            target_reward = 1000.0
            self.get_logger().info(f'TARGET REACHED! Bonus reward: {target_reward}')
            reward += target_reward
            time_bonus = max(0, (self.max_episode_duration - episode_duration) * 10.0)
            reward += time_bonus
        else:
            distance_penalty = final_distance * -5.0 
            reward += distance_penalty

        line_following_reward = self.calculate_line_following_reward()
        reward += line_following_reward
        self.get_logger().info(f' Line following reward: {line_following_reward:.2f}')

        survival_reward = (episode_duration - 3) * 10.0
        reward += survival_reward
        self.get_logger().info(f' Survival reward: {survival_reward:.2f}')

        avg_speed = 0.0
        if self.trajectory_data:
            total_speed = sum(
                abs(point['wheel_speeds'][0]) + abs(point['wheel_speeds'][1])
                for point in self.trajectory_data
            )
            avg_speed = total_speed / (2 * len(self.trajectory_data))
        speed_reward = avg_speed * 0.1
        reward += speed_reward
        self.get_logger().info(f' Speed reward: {speed_reward:.2f}')


        if self.line_out_flag:
            line_out_penalty = -100.0 
            reward += line_out_penalty
            self.get_logger().info(f' LINE OUT! Penalty applied: {line_out_penalty}')

        self.get_logger().info(f' Total episode reward: {reward:.2f}')
        return reward


    def calculate_line_following_reward(self):
        if not self.trajectory_data:
            return 0.0

        total_line_deviation_penalty = 0.0
        for point in self.trajectory_data:
            brightness = point['brightness']
            sensor_positions = np.array([-35, -25, -15, -5, 5, 15, 25, 35])
            weights = np.maximum(0, 255 - np.array(brightness))
            total_weight = np.sum(weights)

            line_position = 0.0
            if total_weight > 0:
                line_position = np.sum(sensor_positions * weights) / total_weight

            normalized_line_position_abs = abs(line_position) / 35.0 

            penalty_per_step = normalized_line_position_abs**2 * 1.0 
            total_line_deviation_penalty += penalty_per_step

        avg_penalty = total_line_deviation_penalty / len(self.trajectory_data)

        line_following_reward = -avg_penalty * 50.0 
        return line_following_reward
        


    def reset(self):
        self.get_logger().info('Resetting robot: stopping wheels and resetting pose...')

        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0]
        self.right_wheel_pub.publish(stop_msg)
        self.left_wheel_pub.publish(stop_msg)
        self.get_logger().info('Wheels stopped.')
        
        rclpy.spin_once(self, timeout_sec=0.1)

        self.current_episode_step = 0
        self.trajectory_data = []
        self.line_out_flag = False

        rclpy.spin_once(self, timeout_sec=0.1) 

        # === 初期位置にロボットを戻す ===
        if self.set_robot_pose(self.robot_name, self.initial_pose):
            self.get_logger().info('Robot pose reset succeeded. Waiting for sensor data...')
            rclpy.spin_once(self, timeout_sec=0.5) 
            time.sleep(1.0)
            initial_state = self.observe_state()
            self.episode_start_time = time.time() 
            return initial_state
        else:
            self.get_logger().error('Robot pose reset failed. This will cause issues.')
            return None

    def set_robot_pose(self, entity_name, pose):
        """
        Gazeboシミュレーション内のエンティティ（ロボット）のポーズを設定するサービスを呼び出す関数。
        """
        if not self.set_pose_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/world/empty/set_pose service unavailable for pose reset.')
            return False

        req = SetEntityPose.Request()
        req.entity = Entity()
        req.entity.name = entity_name
        req.pose = pose

        future = self.set_pose_client.call_async(req)

        timeout_start = time.time()
        timeout_duration = 5.0
        while rclpy.ok() and not future.done() and (time.time() - timeout_start < timeout_duration):
            rclpy.spin_once(self, timeout_sec=0.1)

        if future.done() and future.result() and future.result().success:
            return True
        else:
            self.get_logger().error(f'Failed to set pose for {entity_name}. Result: {future.result()}. Future done: {future.done()}')
            return False

def main(args=None):
    rclpy.init(args=args)
    env_node = LineTraceEnv()
    try:
        rclpy.spin(env_node)
    except KeyboardInterrupt:
        env_node.get_logger().info('LineTraceEnv node interrupted. Shutting down.')
    finally:
        env_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()