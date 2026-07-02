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







grasswonder@grasswonder:~/midas_hand/midas-piper-control/data_collection$ python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py --curobo
Foot pedal input: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
Press c in a display window to calibrate teleop neutral. Press q to quit.
Starting task_camera worker...
Task camera (depthai/OAK-1): 1280x720 @30fps
Starting realsense worker...
Warning: RealSense reports USB 2.x; use USB 3 for reliable 640x480@30.
RealSense stream (teleop depth only): 640x480 rgb8 @30fps serial=317422074281
Starting piper worker...
Piper payload set to 'half': ok
Piper connected: True receive FPS: 2860.0
[piper_worker] firmware version: None
Starting midas worker...
Delta wrist teleop loaded: config=/home/grasswonder/midas_hand/midas-piper-control/data_collection/config.yaml orientation_enabled=False scale_factor=1.0
Midas hand connected on /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FJXX-if00-port0; motor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12]
Starting teleop worker...
Pinch calibration not available in this version of midas_hand_teleop.
Logitech finger camera opened: /dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0 (press z to switch)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1783007964.489985  491653 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1783007964.593267  491679 gl_context.cc:385] GL version: 3.2 (OpenGL ES 3.2 NVIDIA 570.133.07), renderer: NVIDIA GeForce RTX 3050Ti Laptop GPU/PCIe/SSE2
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1783007964.623586  491659 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference.Disabling support for feedback tensors.
W0000 00:00:1783007964.631944  491669 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference.Disabling support for feedback tensors.
MediaPipe hand detector initialized.
Midas retargeter initialized.
Starting tactile worker...
No port specified — auto-detecting Paxini board...
Paxini board found on /dev/ttyACM0
W0000 00:00:1783007965.373670  491666 landmark_projection_calculator.cc:78] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Tactile sensor connected on /dev/ttyACM0; fingers=['thumb', 'index', 'middle', 'ring']
Starting curobo worker...
[curobo_worker] Using cuRobo v2 ModelPredictiveControl (MPC).
[curobo_worker] MPC command window widened to 9 waypoints (75.0ms) to cover the 40.0ms replan period.
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
Arm teleop enabled: True
Arm teleop enabled; recalibrating wrist reference.
[diag] Before enable — q=['-0.2820', '+1.7660', '-0.9784', '+0.4418', '-0.6706', '-1.5786']  eef=['+0.3640', '-0.1306', '+0.2409', '-1.6504','+0.3253', '-2.1480']
[diag] grav_comp model=piper_grav_comp_midas_hand.xml  gravity=[0.0, -9.81, 0.0]  output_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
[diag] After MIT mode — q=['-0.2820', '+1.7660', '-0.9784', '+0.4418','-0.6706', '-1.5786']  eef=['+0.3640', '-0.1306', '+0.2409', '-1.6504', '+0.3253', '-2.1480']
[piper_worker] cuRobo MIT mode enabled.
Wrist delta calibration: 1/50
[diag/MIT] q=['-0.282', '+1.766', '-0.978', '+0.442', '-0.671', '-1.579']  grav_τ=['+4.839', '-0.572', '+0.739', '-0.008', '+0.654', '+0.561']  buf_stale=True  target_eef=[]
[diag/jitter] measured |v| peak-to-peak over 250ms (2 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
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
Wrist delta calibration: 14/50
Wrist delta calibration: 15/50
Wrist delta calibration: 16/50
[diag/MIT] q=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  grav_τ=['+4.866', '-0.570', '+0.764', '-0.023', '+0.702', '+0.530']  buf_stale=True  target_eef=[]
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanningmay have failed.
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
[diag/MIT] q=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  grav_τ=['+4.866', '-0.570', '+0.764', '-0.022', '+0.702', '+0.530']  buf_stale=True  target_eef=[]
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
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
[diag/MIT] q=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  grav_τ=['+4.866', '-0.570', '+0.764', '-0.022', '+0.702', '+0.530']  buf_stale=True  target_eef=[]
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[curobo_worker] Warming up MPC (initial setup + cold-start solve)...
ERROR:root:read: [TxRxResult] There is no status packet!
[diag/jitter] measured |v| peak-to-peak over 250ms (15 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/MIT] q=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  grav_τ=['+4.866', '-0.570', '+0.764', '-0.022', '+0.702', '+0.530']  buf_stale=True  target_eef=['+0.363', '-0.138', '+0.213', '-1.673', '+0.381', '-2.194']
[diag/MIT] q=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  grav_τ=['+4.866', '-0.570', '+0.764', '-0.022', '+0.702', '+0.530']  buf_stale=True  target_eef=['+0.334', '-0.131', '+0.210', '-1.673', '+0.381', '-2.194']
[diag/jitter] measured |v| peak-to-peak over 250ms (23 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (23 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (26 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
Returning to home pose... Press r again to cancel.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
ERROR:root:read: [TxRxResult] Incorrect status packet!
[curobo_worker] Warmup complete.
[diag/jitter] measured |v| peak-to-peak over 250ms (19 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (20 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/cuRobo #1] Plan OK  tgt_eef=['+0.363', '-0.136', '+0.232', '-1.673', '+0.381', '-2.194']  waypoints=9  q_start=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  q_end=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']
Home mode cancelled.
Home mode cancelled; recalibrating wrist reference.
Wrist delta calibration: 1/50
Wrist delta calibration: 2/50
[diag/jitter] measured |v| peak-to-peak over 250ms (20 samples): J1=0.058  J2=0.218  J3=0.065  J4=0.572  J5=0.212  J6=0.429 rad/s
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
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanningmay have failed.
Wrist delta calibration: 14/50
Wrist delta calibration: 15/50
Wrist delta calibration: 16/50
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
Wrist delta calibration: 32/50
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
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
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
[diag/cuRobo #2] Plan OK  tgt_eef=['+0.325', '-0.300', '+0.200', '-1.571', '-0.660', '-1.571']  waypoints=9  q_start=['-0.288', '+1.778', '-0.978', '+0.491', '-0.672', '-1.550']  q_end=['-0.281', '+1.780', '-0.976', '+0.482', '-0.679', '-1.558']
[diag/cuRobo #3] Plan OK  tgt_eef=['+0.359', '-0.140', '+0.233', '-1.708', '+0.459', '-2.263']  waypoints=9  q_start=['-0.286', '+1.798', '-0.975', '+0.541', '-0.685', '-1.522']  q_end=['-0.270', '+1.803', '-0.971', '+0.515', '-0.702', '-1.543']
[diag/joint] J1(measured_dq=-0.0070,planned_end_dq=+0.0116)  J2(measured_dq=+0.0176,planned_end_dq=+0.0238)  J4(measured_dq=+0.0627,planned_end_dq=+0.0335)  J5(measured_dq=-0.0039,planned_end_dq=-0.0226)  J6(measured_dq=+0.0383,planned_end_dq=+0.0147)
[diag/cuRobo #4] Plan OK  tgt_eef=['+0.361', '-0.142', '+0.231', '-1.708', '+0.459', '-2.263']  waypoints=9  q_start=['-0.292', '+1.798', '-0.979', '+0.547', '-0.679', '-1.516']  q_end=['-0.290', '+1.802', '-0.984', '+0.547', '-0.680', '-1.519']
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=-0.0201)  J3(measured_dq=+0.0000,planned_end_dq=-0.0136)  J4(measured_dq=-0.0019,planned_end_dq=+0.0314)  J5(measured_dq=+0.0000,planned_end_dq=+0.0220)  J6(measured_dq=+0.0000,planned_end_dq=+0.0238)
[diag/cuRobo #5] Plan OK  tgt_eef=['+0.362', '-0.144', '+0.231', '-1.708', '+0.459', '-2.263']  waypoints=9  q_start=['-0.295', '+1.800', '-0.981', '+0.552', '-0.678', '-1.512']  q_end=['-0.296', '+1.803', '-0.988', '+0.554', '-0.675', '-1.513']
[diag/joint] J4(measured_dq=+0.0132,planned_end_dq=+0.0106)
[diag/joint] J4(measured_dq=+0.0141,planned_end_dq=+0.0099)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.098  J2=0.164  J3=0.103  J4=0.556  J5=0.272  J6=0.343 rad/s
[diag/joint] J4(measured_dq=+0.0102,planned_end_dq=+0.0080)
[diag/joint] J4(measured_dq=+0.0090,planned_end_dq=+0.0115)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.072  J2=0.289  J3=0.121  J4=0.184  J5=0.589  J6=0.577 rad/s
[diag/joint] J5(measured_dq=+0.0110,planned_end_dq=+0.0133)
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.336  J2=0.600  J3=0.161  J4=0.701  J5=0.783  J6=0.666 rad/s
[diag/joint] J4(measured_dq=+0.0279,planned_end_dq=+0.0279)  J6(measured_dq=+0.0156,planned_end_dq=+0.0151)
[diag/joint] J5(measured_dq=-0.0090,planned_end_dq=-0.0126)
[diag/joint] J2(measured_dq=+0.0102,planned_end_dq=+0.0031)  J4(measured_dq=+0.0254,planned_end_dq=+0.0241)  J6(measured_dq=+0.0124,planned_end_dq=+0.0068)
[diag/joint] J6(measured_dq=-0.0073,planned_end_dq=-0.0107)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.240  J2=0.210  J3=0.142  J4=0.323  J5=0.453  J6=0.501 rad/s
[diag/joint] J6(measured_dq=+0.0090,planned_end_dq=+0.0114)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.120  J2=0.095  J3=0.044  J4=0.295  J5=0.145  J6=0.109 rad/s
[diag/joint] J5(measured_dq=-0.0086,planned_end_dq=-0.0111)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0143)  J6(measured_dq=-0.0062,planned_end_dq=+0.0123)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.286  J2=0.292  J3=0.207  J4=0.484  J5=0.419  J6=0.964 rad/s
[diag/joint] J6(measured_dq=+0.0020,planned_end_dq=+0.0176)
[diag/joint] J6(measured_dq=+0.0114,planned_end_dq=+0.0068)
[diag/joint] J4(measured_dq=+0.0042,planned_end_dq=+0.0108)
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.262  J2=0.419  J3=0.181  J4=0.365  J5=0.315  J6=0.467 rad/s
[diag/joint] J2(measured_dq=+0.0027,planned_end_dq=+0.0132)  J6(measured_dq=+0.0061,planned_end_dq=-0.0134)
[diag/joint] J5(measured_dq=-0.0123,planned_end_dq=-0.0118)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.375  J2=0.242  J3=0.122  J4=0.556  J5=0.513  J6=0.576 rad/s
[diag/joint] J6(measured_dq=+0.0108,planned_end_dq=+0.0110)
[diag/joint] J2(measured_dq=+0.0112,planned_end_dq=+0.0105)  J4(measured_dq=+0.0157,planned_end_dq=+0.0139)  J6(measured_dq=+0.0108,planned_end_dq=+0.0121)
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.121  J2=0.266  J3=0.051  J4=0.432  J5=0.306  J6=0.496 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.256  J2=0.457  J3=0.064  J4=0.413  J5=0.325  J6=0.419 rad/s
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.201  J2=0.388  J3=0.106  J4=0.327  J5=0.203  J6=0.459 rad/s
[diag/joint] J1(measured_dq=+0.0097,planned_end_dq=+0.0129)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.308  J2=0.260  J3=0.167  J4=0.929  J5=0.473  J6=0.467 rad/s
[diag/joint] J1(measured_dq=+0.0103,planned_end_dq=+0.0110)
[diag/joint] J3(measured_dq=-0.0025,planned_end_dq=-0.0101)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.354  J2=0.213  J3=0.156  J4=0.293  J5=0.632  J6=0.640 rad/s
[diag/joint] J6(measured_dq=+0.0000,planned_end_dq=+0.0173)
[diag/joint] J1(measured_dq=+0.0092,planned_end_dq=+0.0101)  J5(measured_dq=-0.0161,planned_end_dq=-0.0120)
[diag/joint] J6(measured_dq=+0.0066,planned_end_dq=-0.0155)
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.427  J2=0.187  J3=0.136  J4=0.387  J5=0.682  J6=0.494 rad/s
[diag/joint] J2(measured_dq=+0.0140,planned_end_dq=+0.0270)  J3(measured_dq=-0.0046,planned_end_dq=-0.0133)  J4(measured_dq=+0.0281,planned_end_dq=+0.0294)  J6(measured_dq=+0.0134,planned_end_dq=+0.0105)
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=-0.0150)  J3(measured_dq=+0.0000,planned_end_dq=+0.0112)
[diag/joint] J2(measured_dq=+0.0217,planned_end_dq=+0.0171)  J3(measured_dq=-0.0136,planned_end_dq=-0.0083)  J5(measured_dq=-0.0138,planned_end_dq=-0.0112)
[diag/joint] J5(measured_dq=+0.0093,planned_end_dq=+0.0110)
[diag/joint] J2(measured_dq=+0.0102,planned_end_dq=+0.0089)  J4(measured_dq=+0.0241,planned_end_dq=+0.0242)  J6(measured_dq=+0.0112,planned_end_dq=+0.0075)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.221  J2=0.218  J3=0.152  J4=0.350  J5=0.359  J6=0.499 rad/s
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0200)  J6(measured_dq=+0.0000,planned_end_dq=+0.0112)
[diag/joint] J4(measured_dq=+0.0076,planned_end_dq=-0.0129)  J6(measured_dq=+0.0000,planned_end_dq=-0.0122)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.100  J2=0.198  J3=0.051  J4=0.442  J5=0.187  J6=0.348 rad/s
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0228)  J6(measured_dq=-0.0021,planned_end_dq=+0.0158)
[diag/joint] J4(measured_dq=+0.0002,planned_end_dq=-0.0213)  J6(measured_dq=+0.0006,planned_end_dq=-0.0126)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.116  J2=0.369  J3=0.163  J4=0.553  J5=0.327  J6=0.504 rad/s
[diag/joint] J1(measured_dq=-0.0109,planned_end_dq=-0.0104)
[diag/joint] J5(measured_dq=+0.0117,planned_end_dq=+0.0107)
[diag/joint] J4(measured_dq=+0.0095,planned_end_dq=+0.0111)
[diag/joint] J5(measured_dq=+0.0188,planned_end_dq=+0.0207)  J6(measured_dq=-0.0105,planned_end_dq=-0.0102)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.149  J2=0.527  J3=0.282  J4=0.965  J5=0.611  J6=0.661 rad/s
[diag/joint] J4(measured_dq=+0.0116,planned_end_dq=+0.0077)
[diag/joint] J5(measured_dq=+0.0164,planned_end_dq=+0.0170)
[diag/joint] J2(measured_dq=-0.0112,planned_end_dq=-0.0110)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.074  J2=0.424  J3=0.193  J4=0.672  J5=0.273  J6=0.544 rad/s
[diag/joint] J4(measured_dq=-0.0043,planned_end_dq=-0.0110)  J6(measured_dq=-0.0072,planned_end_dq=-0.0116)
[diag/joint] J2(measured_dq=+0.0065,planned_end_dq=+0.0132)  J4(measured_dq=+0.0096,planned_end_dq=+0.0175)  J6(measured_dq=+0.0065,planned_end_dq=+0.0110)
[diag/joint] J6(measured_dq=-0.0110,planned_end_dq=-0.0132)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.131  J2=0.435  J3=0.073  J4=0.796  J5=0.365  J6=0.705 rad/s
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=-0.0270)  J2(measured_dq=+0.0112,planned_end_dq=+0.0230)
[diag/joint] J1(measured_dq=-0.0178,planned_end_dq=-0.0105)  J2(measured_dq=+0.0201,planned_end_dq=+0.0162)  J4(measured_dq=-0.0122,planned_end_dq=-0.0028)
[diag/joint] J4(measured_dq=+0.0075,planned_end_dq=+0.0114)
[diag/joint] J2(measured_dq=+0.0111,planned_end_dq=+0.0028)
[diag/joint] J1(measured_dq=-0.0092,planned_end_dq=+0.0103)  J2(measured_dq=+0.0109,planned_end_dq=+0.0009)  J3(measured_dq=-0.0025,planned_end_dq=-0.0135)  J4(measured_dq=+0.0000,planned_end_dq=-0.0123)
[diag/joint] J3(measured_dq=-0.0025,planned_end_dq=-0.0110)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.073  J2=0.128  J3=0.209  J4=0.262  J5=0.229  J6=0.118 rad/s
[diag/joint] J3(measured_dq=-0.0046,planned_end_dq=-0.0119)
[diag/joint] J1(measured_dq=-0.0041,planned_end_dq=-0.0109)  J2(measured_dq=+0.0131,planned_end_dq=+0.0167)  J3(measured_dq=-0.0133,planned_end_dq=-0.0111)  J4(measured_dq=+0.0054,planned_end_dq=+0.0101)
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=+0.0210)  J3(measured_dq=-0.0107,planned_end_dq=-0.0163)
[diag/joint] J2(measured_dq=+0.0101,planned_end_dq=+0.0100)
[diag/joint] J3(measured_dq=-0.0092,planned_end_dq=-0.0122)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.299  J2=0.203  J3=0.256  J4=0.653  J5=0.653  J6=0.739 rad/s
[diag/joint] J4(measured_dq=+0.0019,planned_end_dq=+0.0194)
[diag/joint] J5(measured_dq=-0.0085,planned_end_dq=-0.0101)
[diag/joint] J4(measured_dq=+0.0023,planned_end_dq=+0.0118)
[diag/joint] J1(measured_dq=+0.0071,planned_end_dq=+0.0108)  J2(measured_dq=+0.0040,planned_end_dq=+0.0163)  J3(measured_dq=-0.0023,planned_end_dq=-0.0165)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.242  J2=0.231  J3=0.189  J4=0.376  J5=0.660  J6=0.582 rad/s
[diag/joint] J4(measured_dq=+0.0017,planned_end_dq=+0.0155)  J5(measured_dq=-0.0109,planned_end_dq=-0.0105)
[diag/joint] J4(measured_dq=+0.0120,planned_end_dq=-0.0025)  J5(measured_dq=-0.0125,planned_end_dq=-0.0137)
[diag/joint] J1(measured_dq=+0.0114,planned_end_dq=+0.0117)  J2(measured_dq=+0.0102,planned_end_dq=+0.0099)  J5(measured_dq=-0.0115,planned_end_dq=-0.0139)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.537  J2=0.177  J3=0.188  J4=0.474  J5=0.921  J6=0.597 rad/s
[diag/joint] J1(measured_dq=+0.0139,planned_end_dq=+0.0154)  J2(measured_dq=+0.0116,planned_end_dq=+0.0114)  J5(measured_dq=-0.0141,planned_end_dq=-0.0180)
[diag/joint] J1(measured_dq=+0.0098,planned_end_dq=+0.0107)  J2(measured_dq=+0.0097,planned_end_dq=+0.0128)  J5(measured_dq=-0.0120,planned_end_dq=-0.0118)
[diag/joint] J2(measured_dq=+0.0125,planned_end_dq=+0.0131)  J5(measured_dq=-0.0109,planned_end_dq=-0.0108)
[diag/joint] J2(measured_dq=+0.0114,planned_end_dq=+0.0112)  J5(measured_dq=-0.0115,planned_end_dq=-0.0138)
[diag/joint] J1(measured_dq=+0.0103,planned_end_dq=+0.0111)  J2(measured_dq=+0.0084,planned_end_dq=+0.0128)  J5(measured_dq=-0.0144,planned_end_dq=-0.0140)
[diag/joint] J2(measured_dq=+0.0122,planned_end_dq=+0.0128)  J5(measured_dq=-0.0082,planned_end_dq=-0.0122)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.371  J2=0.202  J3=0.204  J4=0.435  J5=0.900  J6=0.152 rad/s
[diag/joint] J2(measured_dq=+0.0077,planned_end_dq=+0.0102)
[diag/joint] J2(measured_dq=+0.0085,planned_end_dq=+0.0237)  J3(measured_dq=-0.0048,planned_end_dq=-0.0182)
[diag/joint] J2(measured_dq=+0.0128,planned_end_dq=+0.0059)  J3(measured_dq=-0.0115,planned_end_dq=-0.0006)  J5(measured_dq=-0.0091,planned_end_dq=-0.0120)
[diag/joint] J1(measured_dq=+0.0091,planned_end_dq=+0.0117)
[diag/joint] J6(measured_dq=+0.0065,planned_end_dq=+0.0131)
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.430  J2=0.125  J3=0.194  J4=0.543  J5=0.773  J6=0.287 rad/s
[diag/joint] J2(measured_dq=+0.0101,planned_end_dq=+0.0088)  J4(measured_dq=+0.0146,planned_end_dq=+0.0143)  J6(measured_dq=+0.0159,planned_end_dq=+0.0160)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.246  J2=0.103  J3=0.063  J4=0.465  J5=0.601  J6=0.479 rad/s
[diag/joint] J6(measured_dq=+0.0116,planned_end_dq=+0.0105)
[diag/joint] J2(measured_dq=-0.0008,planned_end_dq=-0.0116)  J3(measured_dq=+0.0000,planned_end_dq=+0.0130)  J4(measured_dq=+0.0173,planned_end_dq=+0.0150)  J5(measured_dq=-0.0093,planned_end_dq=-0.0113)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.377  J2=0.531  J3=0.280  J4=0.689  J5=0.302  J6=0.685 rad/s
[diag/joint] J2(measured_dq=-0.0109,planned_end_dq=-0.0097)
[diag/joint] J1(measured_dq=+0.0098,planned_end_dq=+0.0120)  J4(measured_dq=+0.0118,planned_end_dq=+0.0100)
[diag/joint] J1(measured_dq=+0.0100,planned_end_dq=+0.0110)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.504  J2=0.346  J3=0.062  J4=0.282  J5=0.210  J6=0.254 rad/s
[diag/joint] J1(measured_dq=+0.0051,planned_end_dq=+0.0104)
[diag/joint] J4(measured_dq=+0.0048,planned_end_dq=+0.0102)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.351  J2=0.264  J3=0.175  J4=0.438  J5=0.238  J6=0.294 rad/s
[diag/joint] J2(measured_dq=-0.0015,planned_end_dq=-0.0104)  J3(measured_dq=+0.0054,planned_end_dq=+0.0148)
[diag/joint] J3(measured_dq=+0.0113,planned_end_dq=+0.0107)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.145  J2=0.404  J3=0.234  J4=0.168  J5=0.250  J6=0.423 rad/s
[diag/joint] J1(measured_dq=-0.0109,planned_end_dq=-0.0124)  J4(measured_dq=+0.0156,planned_end_dq=+0.0151)  J6(measured_dq=+0.0123,planned_end_dq=+0.0113)
[diag/joint] J4(measured_dq=+0.0056,planned_end_dq=+0.0164)
[diag/joint] J4(measured_dq=+0.0002,planned_end_dq=+0.0326)  J6(measured_dq=-0.0043,planned_end_dq=+0.0174)
[diag/joint] J4(measured_dq=+0.0105,planned_end_dq=+0.0023)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.178  J2=0.501  J3=0.345  J4=0.536  J5=0.439  J6=0.550 rad/s
[diag/joint] J4(measured_dq=+0.0127,planned_end_dq=+0.0127)
[diag/joint] J4(measured_dq=+0.0074,planned_end_dq=+0.0101)
Arm teleop enabled: False
[diag/joint] J4(measured_dq=+0.0072,planned_end_dq=-0.0270)  J6(measured_dq=+0.0000,planned_end_dq=-0.0150)
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanningmay have failed.