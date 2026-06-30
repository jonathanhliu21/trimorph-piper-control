python3 test_mit_controller.py --firmware v183 \
    --hold-K 0.0 --hold-B 0.0 --damping 0.0 --torque-scale 1.05


python3 test_mit_controller.py --firmware v183 \
    --torque-scale 1.05 \
    --coulomb-comp 0.05 --viscous-comp 0.05


cd /home/grasswonder/midas_hand/midas-piper-control/data_collection
conda activate piper_control
python3 collect_raw_dataset.py --curobo



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




(piper_control) (.venv) grasswonder@grasswonder:~/midas_hand/midas-piper-control/data_collection$ python3 collect_raw_dataset.py --curobo
Foot pedal input: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
Press c in a display window to calibrate teleop neutral. Press q to quit.
Starting task_camera worker...
Task camera (opencv): 1280x720 @30fps device=2
Starting realsense worker...
Warning: RealSense reports USB 2.x; use USB 3 for reliable 640x480@30.
RealSense stream: 640x480 rgb8 @30fps serial=317422074281
RealSense stream: teleop-only (no recording), serial=317422074281
Starting piper worker...
Piper payload set to 'half': ok
Piper connected: True receive FPS: 0.0
[piper_worker] firmware version: None
Starting midas worker...
Delta wrist teleop loaded: config=/home/grasswonder/midas_hand/midas-piper-control/data_collection/config.yaml orientation_enabled=False scale_factor=0.3
Midas hand connected on /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FJXX-if00-port0; motor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Starting teleop worker...
Pinch calibration not available in this version of midas_hand_teleop.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1782797514.436796  130359 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1782797514.543066  130385 gl_context.cc:385] GL version: 3.2 (OpenGL ES 3.2 NVIDIA 570.133.07), renderer: NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1782797514.573675  130365 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1782797514.585379  130384 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
MediaPipe hand detector initialized.
Midas retargeter initialized.
Starting tactile worker...
No port specified — auto-detecting Paxini board...
Paxini board found on /dev/ttyACM0
W0000 00:00:1782797514.674434  130373 landmark_projection_calculator.cc:78] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Tactile sensor connected on /dev/ttyACM0; fingers=['thumb', 'index', 'middle', 'ring']
Starting curobo worker...
[curobo_worker] Using cuRobo v2 MotionPlanner.
[curobo_worker] Warming up MotionPlanner (JIT compile)...
[curobo_worker] Warmup complete.
QFontDatabase: Cannot find font directory /home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
Arm teleop enabled: True
Arm teleop enabled; recalibrating wrist reference.
[diag] Before enable — q=['-1.0621', '+1.7219', '-0.8699', '-0.8641', '-1.1599', '-1.4636']  eef=['+0.2213', '-0.2665', '+0.2525', '-1.3980', '-0.4753', '-1.9295']
[diag] grav_comp model=piper_grav_comp_midas_hand.xml  gravity=[0.0, -9.81, 0.0]  output_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
[diag] After MIT mode — q=['-1.0621', '+1.7219', '-0.8699', '-0.8641', '-1.1599', '-1.4636']  eef=['+0.2213', '-0.2665', '+0.2525', '-1.3980', '-0.4753', '-1.9295']
[piper_worker] cuRobo MIT mode enabled.
Wrist delta calibration: 1/50
[diag/MIT] q=['-1.062', '+1.722', '-0.870', '-0.864', '-1.160', '-1.464']  grav_τ=['+2.384', '-1.667', '+1.978', '-0.372', '-0.281', '+0.515']  buf_stale=True  target_eef=[]
Wrist delta calibration: 2/50
Wrist delta calibration: 3/50
Wrist delta calibration: 4/50
Wrist delta calibration: 5/50
Wrist delta calibration: 6/50
Wrist delta calibration: 7/50
Wrist delta calibration: 8/50
Wrist delta calibration: 9/50
Wrist delta calibration: 10/50
Wrist delta calibration: 11/50
Wrist delta calibration: 12/50
Wrist delta calibration: 13/50
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
Wrist delta calibration: 14/50
Wrist delta calibration: 15/50
Wrist delta calibration: 16/50
[diag/MIT] q=['-1.073', '+1.749', '-0.870', '-0.766', '-1.072', '-1.418']  grav_τ=['+2.319', '-1.535', '+2.106', '-0.323', '-0.138', '+0.521']  buf_stale=True  target_eef=[]
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
Wrist delta calibration: 17/50
Wrist delta calibration: 18/50
Wrist delta calibration: 19/50
Wrist delta calibration: 20/50
Wrist delta calibration: 21/50
Wrist delta calibration: 22/50
Wrist delta calibration: 23/50
Wrist delta calibration: 24/50
Wrist delta calibration: 25/50
Wrist delta calibration: 26/50
Wrist delta calibration: 27/50
Wrist delta calibration: 28/50
Wrist delta calibration: 29/50
Wrist delta calibration: 30/50
Wrist delta calibration: 31/50
[diag/MIT] q=['-1.073', '+1.749', '-0.870', '-0.766', '-1.072', '-1.418']  grav_τ=['+2.319', '-1.535', '+2.106', '-0.323', '-0.138', '+0.521']  buf_stale=True  target_eef=[]
Wrist delta calibration: 32/50
Wrist delta calibration: 33/50
Wrist delta calibration: 34/50
Wrist delta calibration: 35/50
Wrist delta calibration: 36/50
Wrist delta calibration: 37/50
Wrist delta calibration: 38/50
Wrist delta calibration: 39/50
Wrist delta calibration: 40/50
Wrist delta calibration: 41/50
Wrist delta calibration: 42/50
Wrist delta calibration: 43/50
Wrist delta calibration: 44/50
Wrist delta calibration: 45/50
Wrist delta calibration: 46/50
[diag/MIT] q=['-1.073', '+1.749', '-0.870', '-0.766', '-1.072', '-1.418']  grav_τ=['+2.319', '-1.535', '+2.106', '-0.323', '-0.138', '+0.521']  buf_stale=True  target_eef=[]
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.282', '+0.242', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.764', '-1.072', '-1.418']
[diag/MIT] q=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']  grav_τ=['+2.319', '-1.535', '+2.106', '-0.323', '-0.137', '+0.520']  buf_stale=True  target_eef=['+0.219', '-0.282', '+0.247', '-1.452', '-0.367', '-2.015']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.219', '-0.282', '+0.245', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.282', '+0.253', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
[diag/MIT] q=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']  grav_τ=['+2.319', '-1.535', '+2.106', '-0.323', '-0.137', '+0.520']  buf_stale=True  target_eef=['+0.218', '-0.279', '+0.267', '-1.452', '-0.367', '-2.015']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.281', '+0.264', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.276', '+0.269', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.219', '-0.269', '+0.269', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.217', '-0.278', '+0.259', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.219', '-0.282', '+0.232', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.281', '+0.216', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.418']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.227', '-0.259', '+0.207', '-1.452', '-0.367', '-2.015']  q_current=['-1.073', '+1.749', '-0.870', '-0.765', '-1.072', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.227', '-0.260', '+0.218', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.761', '-1.071', '-1.417']
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.237', '-0.276', '+0.238', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.249', '-0.287', '+0.236', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.227', '-0.287', '+0.232', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.219', '-0.287', '+0.242', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.228', '-0.286', '+0.239', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.226', '-0.287', '+0.232', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.221', '-0.288', '+0.236', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.217', '-0.290', '+0.250', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.216', '-0.292', '+0.258', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.216', '-0.293', '+0.257', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.216', '-0.294', '+0.253', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.218', '-0.292', '+0.247', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.219', '-0.289', '+0.240', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.288', '+0.231', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
Waiting for Piper teacher mode before teleop/recording (teach_status='STOP_RECORDING(0x2)').
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
^X[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
^C
Collection stopped.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.223', '-0.289', '+0.230', '-1.452', '-0.367', '-2.015']  q_current=['-1.075', '+1.749', '-0.870', '-0.759', '-1.069', '-1.417']
^CTraceback (most recent call last):
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py", line 175, in <module>
    raise SystemExit(main())