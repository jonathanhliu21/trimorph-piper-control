//environment setup
/home/grasswonder/midas_hand/midas-piper-control/setup_record.sh

python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py --task-camera-device /dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0


cd /home/grasswonder/midas_hand/midas-piper-control/data_collection
conda activate piper_control
python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py


//teledex operation
python3 test_mit_controller.py --firmware v183 \
    --torque-scale 1.05 \
    --coulomb-comp 0.05 --viscous-comp 0.05

z: toggle direct kinesthetic
e: pure vision-based






python3 test_mit_controller.py --firmware v183 \
    --torque-scale 1.05 \
    --coulomb-comp 0.05 --viscous-comp 0.05






calibrate_gravity.py --calibrate payload




python calibrate_gravity.py --calibrate payload \
  --gravity 1.196 -9.687 -0.981 \
  --hand-mass 0.75


  ========== PAYLOAD CoM CALIBRATION RESULT ==========
Hand mass (given): 0.7500 kg
CoM offset in gripper frame (m): +0.0556 +0.0203 +0.2782
First moment m*r (kg*m): +0.04168 +0.01521 +0.20865
Fit RMS residual: 0.9355 N*m
Condition number: 6.6

Update the model: set the 'payload_tool' body to
  pos="0.0556 0.0203 0.2782" mass="0.7500"




(piper_control) (.venv) grasswonder@grasswonder:~/midas_hand/midas-piper-control/data_collection$ python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py
Foot pedal input: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
Press c in a display window to calibrate teleop neutral. Press q to quit.
Starting task_camera worker...
Task camera (depthai/OAK-1): 1280x720 @30fps
Starting realsense worker...
Starting piper worker...
Piper payload set to 'half': ok
Piper connected: True receive FPS: 2840.0
[piper_worker] firmware version: None
Traceback (most recent call last):
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/utils.py", line 60, in worker_wrapper
    target(*args, init_done=init_done)
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/runner.py", line 52, in _realsense_teleop_only
    run_realsense_worker(cfg, sh, rec, init_done=init_done, record=False)
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/realsense_worker.py", line 132, in run_realsense_worker
    selected = choose_color_profile(rs, device, config)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/realsense_worker.py", line 87, in choose_color_profile
    raise RuntimeError(
RuntimeError: No RealSense RGB8 profile for 1280x720@30. Supported: 424x240@6, 424x240@15, 424x240@30, 424x240@60, 640x480@6, 640x480@15, 640x480@30, 1280x720@6, 1280x720@10, 1280x720@15, 1920x1080@8
Delta wrist teleop loaded: config=/home/grasswonder/midas_hand/midas-piper-control/data_collection/config.yaml orientation_enabled=False scale_factor=0.7
[19443010E1B7F18B00] [1.12.4] [1782924261.770] [host] [warning] Device crashed, but no crash dump could be extracted.
[19443010E1B7F18B00] [1.12.4] [1782924261.773] [host] [error] Device likely crashed but did not reboot in time to get the crash dump
Error: realsense worker failed: No RealSense RGB8 profile for 1280x720@30. Supported: 424x240@6, 424x240@15, 424x240@30, 424x240@60, 640x480@6, 640x480@15, 640x480@30, 1280x720@6, 1280x720@10, 1280x720@15, 1920x1080@8