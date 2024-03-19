# S24_roboticsII
ROS2 Workspace for S24 RoboticsII
Group members: Max Destil [662032859], Jaia Jalem Foster [662008108], Philip Zhang [661998396]
## Fork the repository

Press the `Fork` on the page and fork the repo to your account. This will create a repository with exact same content in your github account.

## Cloning the repository 
On the robot, open a new terminal (via VNC or SSH)
```
cd ~/codes
git clone https://github.com/YOUR_ACCOUNT/S24_roboticsII_ws.git [team_name]_ws
```

## Building the ROS2 workspace
The step will build and ROS2 workspace and compile the packages.

**Docker**: Open/access a docker container via a terminal (VNC or SSH)
```
cd ~/codes/[team_name]_ws
colcon build
```

**Important**: You have to compile the package **EVERYTIME** you modify the code, even the code is written in **PYTHON**. You can simply build one specific package using the following command. (So you don't have to compile ALL packages.)
```
colcon build --packages-select [package_name]
```
The [package_name] are for example, `object_detection`, `tracking_control`.

## Activate ROS2 environment
Activate ROS2 environment to run ROS software

**Important**: You have to activate ROS2 environment **EVERYTIME** you open a new terminal and run/access the docker container.

The reason you don't need to do this while doing previous mini project is that there's an ROS activation command in `/root/.bashrc` in the docker container. Whenever a terminal is launched, it will exected this bash script once.

**Docker**: Open/access a docker container via a terminal (VNC or SSH)
```
cd ~/codes/[team_name]_ws
source install/setup.bash
```

## Launch tracking nodes!

### Color Detection and Tracking Node
**Docker**: Open a terminal and access docker (via VNC or SSH). Remeber to **Activate ROS2 environment**.
```
ros2 launch tracking_control tracking_color_object_launch.py
```

### Teleoperation node
**Docker**: Open another terminal and access docker (via VNC or SSH). Remeber to **Activate ROS2 environment**. In this node, you can control the robot and activate/deactive tracking.
```
ros2 run tracking_control joy_safety_ctrl
```
### Launch the robot and camera
**Docker**: Open another terminal and access docker (via VNC or SSH).
For the old camera model (astra pro, Robot 1~6)
```
ros2 launch tracking_control car_camera_pro_bringup_launch.py
```
For the new camera model (astra pro plus, Robot 7~)
```
ros2 launch tracking_control car_camera_proplus_bringup_launch.py
```

## Robot Teleoeration
At the terminal that run the teleoperation node, the terminal should show this.
```
Control Your Robot!
Press and hold the keys to move around.
Press space key to turn on/off tracking.
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

t/y : turn counterclockwise/clockwise
k : force stop
space key : turn on/off tracking
anything else : stop smoothly

CTRL-C to quit
```

As the hint suggested, the following function can be done by press and/or hold the key.

- Activate/Deactivate autonomous tracking: press space key
- Moving linear in a direction: press and hold `u    i    o` (left forward, forward, right forward), `j    l` (left right), `m    ,    .` (left backward, backward, right backward)
- Turning counterclockwise/clockwise: `t`/`y`
- Stop the robot: press `k` or anything else

Note that the keyboard teleoperation has higher priority than the autonomous tracking command. In addition, deactivate (press space) the robot if you think the robot is (almost) out of control to protect the robot for potential damages.
