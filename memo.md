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
Piper connected: True receive FPS: 2840.0
[piper_worker] firmware version: None
Starting midas worker...
Delta wrist teleop loaded: config=/home/grasswonder/midas_hand/midas-piper-control/data_collection/config.yaml orientation_enabled=False scale_factor=0.7
Midas hand connected on /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FJXX-if00-port0; motor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Starting teleop worker...
Pinch calibration not available in this version of midas_hand_teleop.
Logitech finger camera opened: /dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0 (press z to switch)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1783003626.357143  466366 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1783003626.434322  466392 gl_context.cc:385] GL version: 3.2 (OpenGL ES 3.2 NVIDIA 570.133.07), renderer: NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1783003626.476709  466380 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1783003626.491905  466391 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
MediaPipe hand detector initialized.
Midas retargeter initialized.
Starting tactile worker...
No port specified — auto-detecting Paxini board...
Paxini board found on /dev/ttyACM0
W0000 00:00:1783003627.262952  466383 landmark_projection_calculator.cc:78] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Tactile sensor connected on /dev/ttyACM0; fingers=['thumb', 'index', 'middle', 'ring']
Starting curobo worker...
[curobo_worker] Using cuRobo v2 ModelPredictiveControl (MPC).
[curobo_worker] MPC command window widened to 9 waypoints (75.0ms) tocover the 40.0ms replan period.
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
[diag] Before enable — q=['-0.8191', '+1.3854', '-0.6872', '-1.0131','-0.8633', '-1.5921']  eef=['+0.2295', '-0.1596', '+0.2779', '-1.6324', '-0.7896', '-1.6446']
[diag] grav_comp model=piper_grav_comp_midas_hand.xml  gravity=[0.0, -9.81, 0.0]  output_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
[diag] After MIT mode — q=['-0.8191', '+1.3854', '-0.6872', '-1.0131', '-0.8633', '-1.5921']  eef=['+0.2295', '-0.1596', '+0.2779', '-1.6324', '-0.7896', '-1.6446']
[piper_worker] cuRobo MIT mode enabled.
Wrist delta calibration: 1/50
[diag/MIT] q=['-0.819', '+1.385', '-0.687', '-1.013', '-0.863', '-1.592']  grav_τ=['+2.190', '-1.710', '+1.415', '-0.143', '-0.469', '+0.434']  buf_stale=True  target_eef=[]
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
[diag/MIT] q=['-0.829', '+1.410', '-0.688', '-0.914', '-0.764', '-1.548']  grav_τ=['+2.137', '-1.628', '+1.533', '-0.069', '-0.326', '+0.477']  buf_stale=True  target_eef=[]
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
[diag/MIT] q=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  grav_τ=['+2.137', '-1.628', '+1.533', '-0.069', '-0.325', '+0.477']  buf_stale=True  target_eef=[]
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
[diag/MIT] q=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  grav_τ=['+2.137', '-1.628', '+1.533', '-0.069', '-0.325', '+0.477']  buf_stale=True  target_eef=[]
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[curobo_worker] Warming up MPC (initial setup + cold-start solve)...
[diag/MIT] q=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  grav_τ=['+2.137', '-1.628', '+1.533', '-0.069', '-0.325', '+0.477']  buf_stale=True  target_eef=['+0.230', '-0.170', '+0.273', '-1.676', '-0.679', '-1.739']
[diag/MIT] q=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  grav_τ=['+2.137', '-1.628', '+1.533', '-0.069', '-0.325', '+0.477']  buf_stale=True  target_eef=['+0.250', '-0.170', '+0.276', '-1.676', '-0.679', '-1.739']
ERROR:root:read: [TxRxResult] There is no status packet!
[curobo_worker] Warmup complete.
ERROR:root:read: [TxRxResult] Incorrect status packet!
ERROR:root:read: [TxRxResult] There is no status packet!
ERROR:root:read: [TxRxResult] There is no status packet!
[diag/cuRobo #1] Plan OK  tgt_eef=['+0.225', '-0.174', '+0.269', '-1.676', '-0.679', '-1.739']  waypoints=9  q_start=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  q_end=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
[diag/cuRobo #2] Plan OK  tgt_eef=['+0.264', '-0.154', '+0.281', '-1.676', '-0.679', '-1.739']  waypoints=9  q_start=['-0.829', '+1.410', '-0.688', '-0.913', '-0.764', '-1.548']  q_end=['-0.823', '+1.413', '-0.694', '-0.910', '-0.760', '-1.549']
[diag/cuRobo #3] Plan OK  tgt_eef=['+0.227', '-0.135', '+0.279', '-1.676', '-0.679', '-1.739']  waypoints=9  q_start=['-0.832', '+1.447', '-0.698', '-0.780', '-0.661', '-1.483']  q_end=['-0.810', '+1.448', '-0.711', '-0.777', '-0.652', '-1.491']
[diag/cuRobo #4] Plan OK  tgt_eef=['+0.232', '-0.129', '+0.286', '-1.676', '-0.679', '-1.739']  waypoints=9  q_start=['-0.829', '+1.439', '-0.694', '-0.786', '-0.667', '-1.487']  q_end=['-0.806', '+1.418', '-0.692', '-0.800', '-0.668', '-1.500']
[diag/cuRobo #5] Plan OK  tgt_eef=['+0.235', '-0.131', '+0.285', '-1.676', '-0.679', '-1.739']  waypoints=9  q_start=['-0.826', '+1.446', '-0.699', '-0.750', '-0.652', '-1.468']  q_end=['-0.803', '+1.421', '-0.697', '-0.773', '-0.655', '-1.482']
Arm teleop enabled: False
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
^C
Collection stopped.
^CTraceback (most recent call last):
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py", line 146, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py", line 141, in main
    return run_collection(config)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/runner.py", line 381, in run_collection
    thread.join(timeout=3.0)
  File "/home/grasswonder/miniconda3/lib/python3.12/threading.py", line 1153, in join
    self._wait_for_tstate_lock(timeout=max(timeout, 0))
  File "/home/grasswonder/miniconda3/lib/python3.12/threading.py", line 1169, in _wait_for_tstate_lock
    if lock.acquire(block, timeout):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^