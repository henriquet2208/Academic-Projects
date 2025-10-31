# 🚦 ATC Hand Gesture Recognition for Autonomous Vehicles (IACOS Project)

## 🧭 Overview
This project, developed for the **IACOS (Intelligent Autonomous Cooperative Systems)** course at the **Instituto Superior de Engenharia do Porto (ISEP)**, addresses the challenge of enabling autonomous vehicles to interpret and respond to **Authorized Traffic Controller (ATC)** hand signals.  

Human drivers can intuitively understand these gestures, but autonomous systems require perception and interpretation capabilities.  

This work implements a **ROS2-based system** that detects ATC gestures in real time and issues vehicle commands within a **Gazebo simulation environment** using **OpenCV**, **MediaPipe**, and **Python**.

---

## 📂 Project Files
In the main directory of this repository, you will find the following important files:

| File | Description |
|------|--------------|
| **`GestureRecognition.pdf`** | Official project statement describing the objectives and requirements. |
| **`IACOS_1190709_1200883_1201268.pdf`** | Final project report containing all documentation, implementation details, and conclusions. |
| **`ProjetoIACOS.mp4`** | Demonstration video showing the detection results and system performance. |

---

## 👨‍💻 Authors
- **Gonçalo Moreira** — 1190709  
- **Henrique Teixeira** — 1200883  
- **Rodrigo Moreira** — 1201268  
**Instituto Superior de Engenharia do Porto (ISEP), Portugal**

---

## 🧠 Abstract
This project enables autonomous vehicles to perceive and react to human ATC hand signals, ensuring safer interaction in mixed-traffic environments.  
The system integrates:
- Real-time gesture recognition using **OpenCV** and **MediaPipe**  
- ROS2 nodes for **gesture detection** and **vehicle control**  
- **Gazebo simulation** for validating interactions  

---

## 🧩 System Architecture
### 1. Perception Subsystem
- Captures real-time video from a webcam or simulation.
- Detects ATC gestures using **MediaPipe Pose** and **OpenCV**.
- Publishes recognized gestures to the `/atc/gesture` topic.

### 2. Decision-Making Subsystem
- Interprets gestures and prioritizes safety-critical commands.
- Publishes results to `/atc/orders`.

### 3. Control Subsystem
- Translates commands into motion by publishing to `/cmd_vel`.
- Simulates vehicle behavior in **Gazebo**.

### 4. Simulation Subsystem
- Uses **Gazebo** and **prius_description (real car model in Gazebo)** to simulate the vehicle and environment.

---

## 🧰 Technologies Used
| Category | Tools / Frameworks |
|-----------|--------------------|
| Robotics Middleware | **ROS2 Foxy**, **Autoware ADE** |
| Computer Vision | **OpenCV**, **MediaPipe** |
| Simulation | **Gazebo** |
| Programming | **Python** |
| Vehicle Model | **Prius URDF (prius_description)** |

---

## ⚙️ How to Run the Project

### 1. Setup ROS2 and ADE Environment
```bash
ade start --update --enter
```

### 2. Build the Workspace
```bash
colcon build
source install/setup.bash
```
### 3. Launch the Gazebo Simulation
This command opens the simulation environment and spawns the vehicle model:

```bash
ros2 launch project_iacos gazebo.launch.py
```
Alternatively, you can spawn only the vehicle model:

```bash
ros2 launch project_iacos spawn_vehicle.launch.py
```
### 4. Run the Gesture Detection Node
Start the node that detects ATC gestures via webcam:

```bash
ros2 run project_iacos sign_detector
```
### 5. Run the Vehicle Controller Node
Start the node that subscribes to ATC commands and controls the vehicle:

```bash
ros2 run project_iacos vehicle_controller
```
---

## 🧪 Results
Implemented gesture recognition using MediaPipe and OpenCV.

ROS2 nodes communicate through /atc/orders and /cmd_vel topics.

Gazebo simulation partially implemented due to environment issues.

Demonstrates modularity and potential for integrating AI-based detectors like YOLO.

## 🏁 Conclusion
This project demonstrates a proof-of-concept for gesture-based interaction between Authorized Traffic Controllers and autonomous vehicles using ROS2.
Although simulation was limited, the system effectively recognizes and processes ATC gestures, providing a foundation for safer and more adaptive autonomous driving.


