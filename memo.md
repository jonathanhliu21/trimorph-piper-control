//environment setup
/home/grasswonder/midas_hand/midas-piper-control/setup_record.sh

python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py --task-camera-device /dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0



cd /home/grasswonder/midas_hand/midas-piper-control
source .venv/bin/activate
export DISPLAY=:0          # tmux/screen 세션이면 특히 필요
cd data_collection
python3 collect_raw_dataset.py --curobo

cd /home/grasswonder/midas_hand/midas-piper-control/data_collection
conda activate piper_control
python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py



z: toggle direct kinesthetic
e: pure vision-based
  r: home position



//teledex operation
python3 test_mit_controller.py --firmware v183 \
    --torque-scale 1.05 \
    --coulomb-comp 0.05 --viscous-comp 0.05



% for 
cd /home/grasswonder/midas_hand/midas-piper-control
source .venv/bin/activate
export DISPLAY=:0          # tmux/screen 세션이면 특히 필요
cd data_collection
python3 collect_raw_dataset.py --curobo







python3 test_mit_controller.py --firmware v183 \
    --torque-scale 1.05 \
    --coulomb-comp 0.05 --viscous-comp 0.05






calibrate_gravity.py --calibrate payload




python calibrate_gravity.py --calibrate payload \
  --gravity 1.196 -9.687 -0.981 \
  --hand-mass 0.75






[diag] Before enable — q=['-0.8690', '+1.5151', '-0.8116', '-1.0334','-1.0039', '-1.3905']  eef=['+0.2458', '-0.1886', '+0.2840', '-1.5200', '-0.5968', '-1.6568']
[diag] grav_comp model=piper_grav_comp_midas_hand.xml  gravity=[0.0, -9.81, 0.0]  output_scales=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
[diag] After MIT mode — q=['-0.8690', '+1.5151', '-0.8116', '-1.0334', '-1.0039', '-1.3905']  eef=['+0.2458', '-0.1886', '+0.2840', '-1.5200', '-0.5968', '-1.6568']
[piper_worker] cuRobo MIT mode enabled.
Wrist delta calibration: 1/50
[diag/MIT] q=['-0.869', '+1.515', '-0.812', '-1.033', '-1.004', '-1.390']  grav_τ=['+2.903', '-2.038', '+1.710', '-0.229', '-0.592', '+0.582']  buf_stale=True  target_eef=[]
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
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
[diag/MIT] q=['-0.872', '+1.532', '-0.812', '-0.977', '-0.958', '-1.358']  grav_τ=['+2.898', '-1.972', '+1.782', '-0.205', '-0.510', '+0.607']  buf_stale=True  target_eef=[]
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
[diag/MIT] q=['-0.872', '+1.532', '-0.812', '-0.977', '-0.958', '-1.358']  grav_τ=['+2.898', '-1.972', '+1.782', '-0.205', '-0.513', '+0.607']  buf_stale=True  target_eef=[]
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
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
[diag/MIT] q=['-0.872', '+1.532', '-0.812', '-0.977', '-0.958', '-1.358']  grav_τ=['+2.898', '-1.972', '+1.782', '-0.205', '-0.513', '+0.607']  buf_stale=True  target_eef=[]
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[diag/remote_rtt] 39.0ms  ok=True
[diag/solve] solve=39.0ms  skip=4/9 waypoints (33.3ms dropped)
[diag/cuRobo #1] Plan OK  tgt_eef=['+0.247', '-0.200', '+0.279', '-1.535', '-0.525', '-1.709']  waypoints=5  solve=39.0ms  skip=4  q_start=['-0.861', '+1.534', '-0.805', '-0.958', '-0.961', '-1.360']  q_end=['-0.806', '+1.546', '-0.768', '-0.858', '-0.977', '-1.375']
[diag/remote_rtt] 14.7ms  ok=True
[diag/cuRobo #2] Plan OK  tgt_eef=['+0.248', '-0.207', '+0.279', '-1.535', '-0.525', '-1.709']  waypoints=7  solve=14.8ms  skip=2  q_start=['-0.731', '+1.562', '-0.720', '-0.723', '-0.999', '-1.395']  q_end=['-0.620', '+1.592', '-0.648', '-0.525', '-1.033', '-1.425']
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=+0.1864)  J2(measured_dq=+0.0000,planned_end_dq=+0.0459)  J3(measured_dq=+0.0000,planned_end_dq=+0.1199)  J4(measured_dq=+0.0000,planned_end_dq=+0.3323)  J5(measured_dq=+0.0000,planned_end_dq=-0.0561)  J6(measured_dq=+0.0000,planned_end_dq=-0.0498)
[diag/remote_rtt] 13.9ms  ok=True
[diag/cuRobo #3] Plan OK  tgt_eef=['+0.248', '-0.210', '+0.279', '-1.535', '-0.525', '-1.709']  waypoints=7  solve=14.0ms  skip=2  q_start=['-0.818', '+1.560', '-0.784', '-0.867', '-0.969', '-1.367']  q_end=['-0.814', '+1.575', '-0.788', '-0.835', '-0.975', '-1.367']
[diag/joint] J1(measured_dq=+0.0176,planned_end_dq=-0.1938)  J2(measured_dq=+0.0084,planned_end_dq=-0.0166)  J3(measured_dq=+0.0087,planned_end_dq=-0.1399)  J4(measured_dq=+0.0241,planned_end_dq=-0.3099)  J5(measured_dq=+0.0032,planned_end_dq=+0.0584)  J6(measured_dq=+0.0000,planned_end_dq=+0.0581)
[diag/remote_rtt] 16.5ms  ok=True
[diag/cuRobo #4] Plan OK  tgt_eef=['+0.248', '-0.210', '+0.279', '-1.535', '-0.525', '-1.709']  waypoints=7  solve=16.7ms  skip=2  q_start=['-0.780', '+1.572', '-0.759', '-0.686', '-0.973', '-1.370']  q_end=['-0.795', '+1.580', '-0.779', '-0.713', '-0.975', '-1.365']
[diag/joint] J1(measured_dq=+0.0804,planned_end_dq=+0.0186)  J2(measured_dq=+0.0213,planned_end_dq=+0.0051)  J3(measured_dq=+0.0563,planned_end_dq=+0.0096)  J4(measured_dq=+0.2656,planned_end_dq=+0.1218)  J5(measured_dq=-0.0149,planned_end_dq=-0.0001)  J6(measured_dq=-0.0148,planned_end_dq=+0.0018)
[diag/remote_rtt] 26.6ms  ok=True
[diag/cuRobo #5] Plan OK  tgt_eef=['+0.249', '-0.217', '+0.279', '-1.535', '-0.525', '-1.709']  waypoints=6  solve=27.0ms  skip=3  q_start=['-0.808', '+1.580', '-0.789', '-0.822', '-0.971', '-1.342']  q_end=['-0.817', '+1.582', '-0.802', '-0.852', '-0.969', '-1.347']
[diag/joint] J1(measured_dq=-0.0118,planned_end_dq=-0.0219)  J2(measured_dq=+0.0119,planned_end_dq=+0.0022)  J3(measured_dq=-0.0165,planned_end_dq=-0.0233)  J4(measured_dq=-0.0809,planned_end_dq=-0.1382)  J6(measured_dq=+0.0246,planned_end_dq=+0.0183)
[diag/remote_rtt] 25.0ms  ok=True
[diag/joint] J1(measured_dq=-0.0022,planned_end_dq=+0.0258)  J2(measured_dq=+0.0190,planned_end_dq=+0.0098)  J3(measured_dq=-0.0082,planned_end_dq=+0.0140)  J4(measured_dq=-0.0050,planned_end_dq=+0.0518)  J5(measured_dq=+0.0026,planned_end_dq=+0.0109)  J6(measured_dq=+0.0049,planned_end_dq=-0.0320)
[diag/remote_rtt] 11.8ms  ok=True
[diag/joint] J1(measured_dq=-0.0111,planned_end_dq=-0.0248)  J3(measured_dq=-0.0070,planned_end_dq=-0.0179)  J4(measured_dq=-0.0126,planned_end_dq=-0.0152)  J5(measured_dq=+0.0263,planned_end_dq=+0.0172)  J6(measured_dq=+0.0142,planned_end_dq=+0.0137)
[diag/remote_rtt] 28.2ms  ok=True
[diag/joint] J3(measured_dq=-0.0111,planned_end_dq=-0.0133)  J4(measured_dq=+0.0442,planned_end_dq=+0.0388)  J5(measured_dq=+0.0286,planned_end_dq=+0.0203)
[diag/MIT] q=['-0.813', '+1.606', '-0.796', '-0.723', '-0.911', '-1.330']  grav_τ=['+3.151', '-1.629', '+1.920', '-0.222', '-0.260', '+0.649']  buf_stale=False  target_eef=['+0.246', '-0.230', '+0.277', '-1.535', '-0.525', '-1.709']
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=3.779  J2=1.007  J3=2.731  J4=10.167  J5=1.771  J6=1.791 rad/s
[diag/remote_rtt] 12.2ms  ok=True
[diag/joint] J4(measured_dq=+0.0183,planned_end_dq=+0.0030)
[diag/remote_rtt] 26.5ms  ok=True
[diag/joint] J3(measured_dq=-0.0100,planned_end_dq=-0.0058)
[diag/remote_rtt] 24.5ms  ok=True
[diag/joint] J1(measured_dq=-0.0069,planned_end_dq=+0.0136)  J4(measured_dq=+0.0021,planned_end_dq=+0.0206)  J5(measured_dq=+0.0061,planned_end_dq=+0.0149)
[diag/remote_rtt] 12.2ms  ok=True
[diag/joint] J5(measured_dq=+0.0144,planned_end_dq=+0.0128)
[diag/remote_rtt] 34.2ms  ok=True
[diag/joint] J4(measured_dq=+0.0114,planned_end_dq=+0.0074)  J5(measured_dq=+0.0130,planned_end_dq=+0.0071)
[diag/remote_rtt] 11.4ms  ok=True
[diag/remote_rtt] 24.8ms  ok=True
[diag/remote_rtt] 24.2ms  ok=True
[diag/remote_rtt] 11.5ms  ok=True
[diag/remote_rtt] 24.4ms  ok=True
[diag/remote_rtt] 12.3ms  ok=True
[diag/remote_rtt] 25.0ms  ok=True
[diag/joint] J5(measured_dq=+0.0109,planned_end_dq=+0.0090)
[diag/MIT] q=['-0.811', '+1.663', '-0.870', '-0.669', '-0.825', '-1.323']  grav_τ=['+3.315', '-1.596', '+1.911', '-0.156', '-0.176', '+0.647']  buf_stale=False  target_eef=['+0.227', '-0.278', '+0.263', '-1.535', '-0.525', '-1.709']
[diag/remote_rtt] 24.2ms  ok=True
[diag/joint] J4(measured_dq=+0.0075,planned_end_dq=+0.0417)  J6(measured_dq=+0.0009,planned_end_dq=+0.0197)
[diag/remote_rtt] 13.6ms  ok=True
[diag/remote_rtt] 25.9ms  ok=True
[diag/joint] J4(measured_dq=+0.0112,planned_end_dq=+0.0087)
[diag/remote_rtt] 12.0ms  ok=True
[diag/joint] J4(measured_dq=+0.0175,planned_end_dq=+0.0097)
[diag/remote_rtt] 24.2ms  ok=True
[diag/joint] J4(measured_dq=+0.0141,planned_end_dq=+0.0015)
[diag/remote_rtt] 12.5ms  ok=True
[diag/joint] J4(measured_dq=+0.0151,planned_end_dq=-0.0048)
[diag/remote_rtt] 24.3ms  ok=True
[diag/solve] solve=24.5ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 11.6ms  ok=True
[diag/remote_rtt] 12.8ms  ok=True
[diag/remote_rtt] 23.4ms  ok=True
[diag/remote_rtt] 11.6ms  ok=True
[diag/remote_rtt] 32.7ms  ok=True
[diag/remote_rtt] 11.5ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.190  J2=0.192  J3=0.151  J4=0.337  J5=0.373  J6=0.269 rad/s
[diag/remote_rtt] 14.5ms  ok=True
[diag/remote_rtt] 22.7ms  ok=True
[diag/remote_rtt] 12.5ms  ok=True
[diag/remote_rtt] 12.2ms  ok=True
[diag/remote_rtt] 11.5ms  ok=True
[diag/remote_rtt] 24.7ms  ok=True
[diag/remote_rtt] 14.7ms  ok=True
[diag/remote_rtt] 25.3ms  ok=True
[diag/remote_rtt] 11.7ms  ok=True
[diag/remote_rtt] 24.8ms  ok=True
Arm teleop enabled: False
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.000  J2=0.013  J3=0.006  J4=0.003  J5=0.065  J6=0.051 rad/s