# Reinforcement Learning for Line Following with Gazebo and ROS2 (Jazzy)

This project demonstrates how to use reinforcement learning to optimize PID gains for a line-following robot simulated in Gazebo.

---

## Project Overview

We use the **Proximal Policy Optimization (PPO)** algorithm to train a neural network with a **Long Short-Term Memory (LSTM)** layer. The LSTM layer is crucial for processing the time-series data from our sensors, allowing the robot to learn from the sequence of past observations.

---

## Learning Data

The learning data consists of:
* **Illuminance:** Data from 8 cameras strategically placed on the robot.
* **Illuminance Differences:** The first and second derivatives (differences) of the illuminance data. These values provide crucial information about the **rate of change** of the line's position, helping the robot anticipate turns and react more smoothly.
* **Wheel Velocities:** The rotation speeds of the robot's two wheels.

This data is used to train the model to output optimal PID values for controlling the robot's movement.

---

## How to Use

1.  **Launch the Simulation:**
    First, launch the Gazebo simulation and the ROS2-Gazebo bridge by running the launch file. The bridge configuration is specified in the included YAML file.
    ```bash
    ros2 launch robot_desc linetrace_gazebo.xml.launch
    ```

2.  **Start Training:**
    With Gazebo running, execute the `train.py` script to begin the reinforcement learning process.
    ```bash
    python3 train.py
    ```
    ![Learning in progress](images/training_in_progress.png)

3.  **Monitor and Visualize Results:**
    Training can be stopped manually by pressing `Ctrl+C` or will conclude automatically after a set number of episodes. Upon completion, graphs will be displayed showing the progress of the training, including:
    * Episode length
    * Loss values
    * PID value changes over time

These visualizations provide insight into the learning process and the performance of the trained model.
