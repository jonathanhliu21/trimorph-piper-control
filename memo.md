//environment setup
/home/grasswonder/midas_hand/midas-piper-control/setup_record.sh

python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py --task-camera-device /dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0



cd /home/grasswonder/midas_hand/midas-piper-control
source .venv/bin/activate
export DISPLAY=:0
cd data_collection
python3 collect_raw_dataset.py --curobo

cd /home/grasswonder/midas_hand/midas-piper-control/data_collection
conda activate piper_control
python3 /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py



b



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



-- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 14.6ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.6ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0107)
[diag/remote_rtt] 12.5ms  ok=True
[diag/solve] solve=26.0ms  skip=3/9 waypoints (25.0ms dropped)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.113  J2=0.247  J3=0.097  J4=0.321  J5=0.168  J6=0.311 rad/s
[diag/remote_rtt] 11.9ms  ok=True
[diag/remote_rtt] 26.5ms  ok=True
[diag/joint] J3(measured_dq=+0.0000,planned_end_dq=+0.0107)
[diag/remote_rtt] 25.0ms  ok=True
[diag/remote_rtt] 24.6ms  ok=True
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0116)  J6(measured_dq=-0.0016,planned_end_dq=-0.0127)
[diag/remote_rtt] 29.3ms  ok=True
[diag/solve] solve=13.1ms  skip=2/9 waypoints (16.7ms dropped)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.049  J2=0.227  J3=0.111  J4=0.168  J5=0.279  J6=0.419 rad/s
[diag/remote_rtt] 25.2ms  ok=True
[diag/remote_rtt] 24.1ms  ok=True
[diag/remote_rtt] 24.1ms  ok=True
[diag/remote_rtt] 12.0ms  ok=True
[diag/solve] solve=14.2ms  skip=2/9 waypoints (16.7ms dropped)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.000  J2=0.168  J3=0.096  J4=0.474  J5=0.247  J6=0.304 rad/s
[diag/remote_rtt] 26.2ms  ok=True
[diag/remote_rtt] 24.5ms  ok=True
[diag/remote_rtt] 22.1ms  ok=True
[diag/remote_rtt] 26.0ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.064  J2=0.126  J3=0.059  J4=0.139  J5=0.281  J6=0.304 rad/s
[diag/remote_rtt] 26.3ms  ok=True
[diag/solve] solve=26.4ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 24.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.7ms  ok=True
[diag/remote_rtt] 18.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 29.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.021  J2=0.148  J3=0.036  J4=0.379  J5=0.321  J6=0.342 rad/s
[diag/solve] solve=13.3ms  skip=2/9 waypoints (16.7ms dropped)
[diag/remote_rtt] 16.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0016,planned_end_dq=-0.0142)
[diag/remote_rtt] 12.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.7ms  ok=True
[diag/remote_rtt] 23.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0014,planned_end_dq=+0.0136)
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.088  J2=0.208  J3=0.135  J4=0.628  J5=0.438  J6=0.331 rad/s
[diag/remote_rtt] 28.0ms  ok=True
[diag/solve] solve=28.1ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 30.3ms  ok=True
[diag/remote_rtt] 26.6ms  ok=True
[diag/remote_rtt] 24.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0104)
[diag/remote_rtt] 29.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0022,planned_end_dq=-0.0100)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.085  J2=0.218  J3=0.147  J4=0.214  J5=0.289  J6=0.334 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=24.5ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 25.4ms  ok=True
[diag/remote_rtt] 27.2ms  ok=True
[diag/remote_rtt] 16.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 25.0ms  ok=True
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0104)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.073  J2=0.205  J3=0.095  J4=0.359  J5=0.325  J6=0.294 rad/s
[diag/remote_rtt] 24.3ms  ok=True
[diag/solve] solve=25.7ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 28.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0014,planned_end_dq=+0.0103)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0109)
[diag/remote_rtt] 30.0ms  ok=True
[diag/joint] J4(measured_dq=+0.0010,planned_end_dq=-0.0136)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 24.2ms  ok=True
[diag/remote_rtt] 24.6ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.083  J2=0.268  J3=0.098  J4=0.281  J5=0.294  J6=0.310 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 25.0ms  ok=True
[diag/solve] solve=25.1ms  skip=3/9 waypoints (25.0ms dropped)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0139)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0191)  J6(measured_dq=-0.0002,planned_end_dq=-0.0126)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=-0.0146)  J3(measured_dq=+0.0016,planned_end_dq=+0.0160)
[diag/remote_rtt] 26.5ms  ok=True
[diag/joint] J3(measured_dq=-0.0002,planned_end_dq=+0.0123)
[diag/joint] J2(measured_dq=-0.0024,planned_end_dq=+0.0115)  J3(measured_dq=+0.0049,planned_end_dq=-0.0101)
[diag/remote_rtt] 24.5ms  ok=True
[diag/remote_rtt] 29.6ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.114  J2=0.123  J3=0.117  J4=0.250  J5=0.296  J6=0.277 rad/s
[diag/remote_rtt] 24.0ms  ok=True
[diag/solve] solve=30.2ms  skip=4/9 waypoints (33.3ms dropped)
Captured retargeter neutral offsets: {'index_mcp_abad_joint': 0.019, 'index_mcp_pitch_joint': -0.278, 'index_pip_joint': -0.251, 'middle_mcp_abad_joint': 0.016, 'middle_mcp_pitch_joint': -0.161, 'middle_pip_joint': -0.146, 'ring_mcp_abad_joint': 0.015, 'ring_mcp_pitch_joint': -0.097, 'ring_pip_joint': -0.088, 'thumb_cmc_roll_joint': 1.098, 'thumb_cmc_side_joint': 0.202, 'thumb_mcp_joint': -0.279, 'thumb_dip_joint': -0.363}
[diag/remote_rtt] 24.2ms  ok=True
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0110)  J6(measured_dq=+0.0000,planned_end_dq=+0.0105)
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=-0.0103)
[diag/remote_rtt] 25.1ms  ok=True
[diag/joint] J2(measured_dq=+0.0024,planned_end_dq=-0.0107)  J3(measured_dq=+0.0000,planned_end_dq=+0.0159)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0110)
[diag/remote_rtt] 26.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0109)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 35.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=-0.0142)  J3(measured_dq=+0.0000,planned_end_dq=+0.0175)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0136)  J6(measured_dq=+0.0000,planned_end_dq=+0.0107)
[diag/joint] J6(measured_dq=-0.0019,planned_end_dq=-0.0101)
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.108  J2=0.283  J3=0.254  J4=0.267  J5=0.121  J6=0.351 rad/s
[diag/joint] J2(measured_dq=-0.0070,planned_end_dq=+0.0112)  J3(measured_dq=+0.0053,planned_end_dq=-0.0129)  J6(measured_dq=+0.0001,planned_end_dq=+0.0106)
[diag/remote_rtt] 24.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0140)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=11.4ms  skip=1/9 waypoints (8.3ms dropped)
[diag/remote_rtt] 25.7ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0077,planned_end_dq=+0.0147)  J6(measured_dq=+0.0033,planned_end_dq=+0.0105)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 23.0ms  ok=True
[diag/joint] J4(measured_dq=+0.0020,planned_end_dq=+0.0190)  J6(measured_dq=+0.0000,planned_end_dq=+0.0122)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0037,planned_end_dq=-0.0105)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 25.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0018,planned_end_dq=+0.0138)  J3(measured_dq=+0.0000,planned_end_dq=-0.0104)
[diag/remote_rtt] 25.8ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.137  J2=0.238  J3=0.208  J4=0.279  J5=0.455  J6=0.235 rad/s
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0141)  J6(measured_dq=+0.0000,planned_end_dq=+0.0113)
[diag/remote_rtt] 24.8ms  ok=True
[diag/solve] solve=25.7ms  skip=3/9 waypoints (25.0ms dropped)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 25.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 27.5ms  ok=True
[diag/joint] J4(measured_dq=+0.0024,planned_end_dq=+0.0120)
[diag/joint] J6(measured_dq=+0.0000,planned_end_dq=-0.0113)
[diag/remote_rtt] 23.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.089  J2=0.238  J3=0.099  J4=0.378  J5=0.265  J6=0.783 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=24.6ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 24.6ms  ok=True
[diag/remote_rtt] 28.5ms  ok=True
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0101)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 25.1ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.092  J2=0.155  J3=0.092  J4=0.105  J5=0.202  J6=0.277 rad/s
[diag/remote_rtt] 23.7ms  ok=True
[diag/remote_rtt] 25.7ms  ok=True
[diag/solve] solve=26.0ms  skip=3/9 waypoints (25.0ms dropped)
[diag/joint] J4(measured_dq=+0.0026,planned_end_dq=+0.0107)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.7ms  ok=True
[diag/joint] J4(measured_dq=+0.0017,planned_end_dq=+0.0113)
[diag/remote_rtt] 13.0ms  ok=True
[diag/joint] J6(measured_dq=+0.0000,planned_end_dq=+0.0105)
[diag/joint] J6(measured_dq=+0.0000,planned_end_dq=-0.0105)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0040,planned_end_dq=-0.0123)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0034,planned_end_dq=+0.0101)
[diag/remote_rtt] 12.5ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.124  J2=0.185  J3=0.043  J4=0.482  J5=0.350  J6=0.319 rad/s
[diag/remote_rtt] 16.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=29.2ms  skip=4/9 waypoints (33.3ms dropped)
[diag/remote_rtt] 12.7ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0017,planned_end_dq=-0.0115)  J6(measured_dq=+0.0000,planned_end_dq=-0.0119)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 29.7ms  ok=True
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0160)  J6(measured_dq=+0.0005,planned_end_dq=+0.0113)
[diag/remote_rtt] 24.2ms  ok=True
[diag/remote_rtt] 13.1ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.003  J2=0.217  J3=0.048  J4=0.184  J5=0.298  J6=0.137 rad/s
[diag/remote_rtt] 14.1ms  ok=True
[diag/joint] J4(measured_dq=+0.0019,planned_end_dq=+0.0115)
[diag/solve] solve=31.0ms  skip=4/9 waypoints (33.3ms dropped)
[diag/remote_rtt] 23.9ms  ok=True
[diag/remote_rtt] 24.7ms  ok=True
[diag/remote_rtt] 23.9ms  ok=True
[diag/joint] J4(measured_dq=+0.0020,planned_end_dq=+0.0100)
[diag/remote_rtt] 26.1ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.120  J2=0.117  J3=0.039  J4=0.599  J5=0.195  J6=0.507 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0120)
[diag/remote_rtt] 24.6ms  ok=True
[diag/joint] J6(measured_dq=-0.0025,planned_end_dq=-0.0106)
[diag/solve] solve=15.7ms  skip=2/9 waypoints (16.7ms dropped)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0021,planned_end_dq=-0.0139)  J3(measured_dq=+0.0018,planned_end_dq=+0.0214)
[diag/remote_rtt] 25.4ms  ok=True
[diag/joint] J2(measured_dq=-0.0048,planned_end_dq=+0.0134)  J3(measured_dq=+0.0110,planned_end_dq=-0.0070)
[diag/joint] J2(measured_dq=+0.0021,planned_end_dq=-0.0118)  J3(measured_dq=+0.0002,planned_end_dq=+0.0178)
[diag/remote_rtt] 26.4ms  ok=True
[diag/joint] J5(measured_dq=-0.0076,planned_end_dq=-0.0117)  J6(measured_dq=-0.0019,planned_end_dq=-0.0101)
[diag/remote_rtt] 24.5ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 24.3ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.034  J2=0.127  J3=0.160  J4=0.311  J5=0.272  J6=0.492 rad/s
[diag/joint] J3(measured_dq=+0.0027,planned_end_dq=+0.0108)
[diag/remote_rtt] 24.7ms  ok=True
[diag/joint] J3(measured_dq=+0.0090,planned_end_dq=+0.0119)  J6(measured_dq=-0.0022,planned_end_dq=-0.0102)
[diag/solve] solve=33.4ms  skip=4/9 waypoints (33.3ms dropped)
[diag/joint] J3(measured_dq=+0.0121,planned_end_dq=+0.0124)
[diag/remote_rtt] 24.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=+0.0108,planned_end_dq=+0.0112)  J4(measured_dq=+0.0012,planned_end_dq=+0.0116)  J6(measured_dq=-0.0019,planned_end_dq=+0.0109)
[diag/remote_rtt] 24.0ms  ok=True
[diag/joint] J3(measured_dq=+0.0108,planned_end_dq=+0.0120)
[diag/joint] J3(measured_dq=+0.0091,planned_end_dq=+0.0106)
[diag/joint] J4(measured_dq=+0.0017,planned_end_dq=+0.0103)
[diag/remote_rtt] 24.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=+0.0078,planned_end_dq=+0.0111)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.036  J2=0.293  J3=0.173  J4=0.193  J5=0.385  J6=0.254 rad/s
[diag/joint] J3(measured_dq=+0.0105,planned_end_dq=+0.0142)
[diag/remote_rtt] 13.3ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 24.2ms  ok=True
[diag/solve] solve=24.5ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 24.7ms  ok=True
[diag/remote_rtt] 24.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=-0.0040,planned_end_dq=-0.0132)  J3(measured_dq=+0.0061,planned_end_dq=+0.0177)  J5(measured_dq=-0.0046,planned_end_dq=-0.0137)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 24.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=+0.0114,planned_end_dq=+0.0086)
[diag/joint] J3(measured_dq=+0.0109,planned_end_dq=+0.0119)
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.148  J2=0.256  J3=0.244  J4=0.172  J5=0.311  J6=0.275 rad/s
[diag/remote_rtt] 12.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=24.3ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 13.4ms  ok=True
[diag/joint] J3(measured_dq=+0.0046,planned_end_dq=+0.0213)  J5(measured_dq=-0.0035,planned_end_dq=-0.0121)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=+0.0108,planned_end_dq=+0.0097)  J5(measured_dq=-0.0109,planned_end_dq=-0.0093)
[diag/joint] J3(measured_dq=+0.0124,planned_end_dq=+0.0089)
[diag/joint] J3(measured_dq=+0.0102,planned_end_dq=+0.0066)
[diag/remote_rtt] 35.4ms  ok=True
[diag/joint] J4(measured_dq=+0.0101,planned_end_dq=+0.0056)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=+0.0052,planned_end_dq=+0.0160)
[diag/remote_rtt] 13.8ms  ok=True
[diag/joint] J3(measured_dq=+0.0083,planned_end_dq=+0.0112)  J5(measured_dq=-0.0066,planned_end_dq=-0.0131)
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.145  J2=0.186  J3=0.124  J4=0.216  J5=0.490  J6=0.350 rad/s
[diag/remote_rtt] 13.1ms  ok=True
[diag/joint] J3(measured_dq=+0.0140,planned_end_dq=+0.0139)  J5(measured_dq=-0.0093,planned_end_dq=-0.0127)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0128)
[diag/remote_rtt] 13.7ms  ok=True
[diag/solve] solve=12.4ms  skip=1/9 waypoints (8.3ms dropped)
[diag/remote_rtt] 14.2ms  ok=True
[diag/remote_rtt] 24.0ms  ok=True
[diag/joint] J5(measured_dq=-0.0114,planned_end_dq=-0.0128)
[diag/remote_rtt] 24.3ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.150  J2=0.141  J3=0.157  J4=0.291  J5=0.285  J6=0.203 rad/s
[diag/remote_rtt] 12.7ms  ok=True
Returning to home pose... Press r again to cancel.
[diag/joint] J4(measured_dq=+0.0027,planned_end_dq=+0.0102)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J1(measured_dq=+0.0020,planned_end_dq=+0.0147)  J2(measured_dq=+0.0024,planned_end_dq=+0.0169)  J3(measured_dq=+0.0019,planned_end_dq=-0.0368)  J4(measured_dq=+0.0000,planned_end_dq=-0.0140)  J6(measured_dq=+0.0004,planned_end_dq=-0.0110)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0067,planned_end_dq=+0.0119)  J3(measured_dq=+0.0022,planned_end_dq=-0.0131)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0099,planned_end_dq=+0.0106)  J3(measured_dq=-0.0157,planned_end_dq=-0.0072)  J4(measured_dq=+0.0098,planned_end_dq=+0.0292)  J6(measured_dq=+0.0000,planned_end_dq=+0.0134)
[diag/remote_rtt] 18.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0121,planned_end_dq=+0.0020)  J3(measured_dq=-0.0129,planned_end_dq=-0.0081)  J4(measured_dq=+0.0065,planned_end_dq=-0.0101)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=12.9ms  skip=2/9 waypoints (16.7ms dropped)
[diag/joint] J4(measured_dq=-0.0000,planned_end_dq=-0.0170)  J6(measured_dq=+0.0000,planned_end_dq=-0.0116)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=-0.0110,planned_end_dq=-0.0054)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J3(measured_dq=-0.0104,planned_end_dq=-0.0069)  J6(measured_dq=-0.0101,planned_end_dq=-0.0075)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.6ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0123)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0020,planned_end_dq=-0.0153)  J6(measured_dq=-0.0000,planned_end_dq=-0.0107)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.4ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0154)  J6(measured_dq=+0.0000,planned_end_dq=+0.0121)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0131)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 16.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0130)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.099  J2=0.104  J3=0.101  J4=0.295  J5=0.251  J6=0.446 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0144)
[diag/remote_rtt] 12.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=12.8ms  skip=2/9 waypoints (16.7ms dropped)
[diag/remote_rtt] 21.7ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0029,planned_end_dq=+0.0157)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 13.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0169)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0109)  J6(measured_dq=+0.0000,planned_end_dq=+0.0139)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 13.7ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.066  J2=0.185  J3=0.092  J4=0.141  J5=0.242  J6=0.208 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=16.3ms  skip=2/9 waypoints (16.7ms dropped)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0009,planned_end_dq=-0.0142)
[diag/remote_rtt] 11.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0123)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0143)  J6(measured_dq=+0.0000,planned_end_dq=+0.0121)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0028,planned_end_dq=-0.0123)
[diag/remote_rtt] 11.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0232)  J6(measured_dq=-0.0023,planned_end_dq=+0.0127)
[diag/remote_rtt] 15.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0178)  J6(measured_dq=+0.0000,planned_end_dq=-0.0106)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.068  J2=0.261  J3=0.054  J4=0.539  J5=0.112  J6=0.449 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 16.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0125)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=12.2ms  skip=1/9 waypoints (8.3ms dropped)
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0115)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
Home mode cancelled.
[diag/remote_rtt] 12.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
Home mode cancelled; recalibrating wrist reference.
Wrist delta calibration: 1/50
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
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.062  J2=0.096  J3=0.044  J4=0.295  J5=0.323  J6=0.134 rad/s
Wrist delta calibration: 15/50
Wrist delta calibration: 16/50
Wrist delta calibration: 17/50
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
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
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.000 rad/s
Wrist delta calibration: 45/50
Wrist delta calibration: 46/50
Wrist delta calibration: 47/50
Wrist delta calibration: 48/50
Wrist delta calibration: 49/50
Wrist delta calibration complete.
[diag/remote_rtt] 16.1ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=16.6ms  skip=2/9 waypoints (16.7ms dropped)
[diag/joint] J2(measured_dq=+0.0110,planned_end_dq=+0.0160)  J4(measured_dq=+0.0477,planned_end_dq=+0.0620)  J6(measured_dq=+0.0255,planned_end_dq=+0.0333)
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=+0.0210)  J3(measured_dq=+0.0000,planned_end_dq=-0.0106)  J4(measured_dq=+0.0000,planned_end_dq=+0.0513)  J6(measured_dq=+0.0000,planned_end_dq=+0.0361)
[diag/joint] J3(measured_dq=-0.0047,planned_end_dq=-0.0128)  J4(measured_dq=+0.0116,planned_end_dq=+0.0053)
[diag/joint] J3(measured_dq=-0.0067,planned_end_dq=-0.0127)  J4(measured_dq=+0.0121,planned_end_dq=+0.0048)
[diag/joint] J4(measured_dq=+0.0149,planned_end_dq=+0.0101)
[diag/remote_rtt] 24.8ms  ok=True
[diag/joint] J3(measured_dq=-0.0127,planned_end_dq=-0.0051)
[diag/remote_rtt] 24.8ms  ok=True
[diag/remote_rtt] 14.4ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.084  J2=0.128  J3=0.144  J4=0.362  J5=0.302  J6=0.241 rad/s
[diag/remote_rtt] 15.6ms  ok=True
[diag/solve] solve=20.3ms  skip=2/9 waypoints (16.7ms dropped)
[diag/joint] J1(measured_dq=+0.0044,planned_end_dq=+0.0101)
[diag/remote_rtt] 13.3ms  ok=True
[diag/remote_rtt] 11.6ms  ok=True
[diag/remote_rtt] 20.3ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.087  J2=0.194  J3=0.134  J4=0.116  J5=0.258  J6=0.145 rad/s
Returning to home pose... Press r again to cancel.
[diag/remote_rtt] 24.5ms  ok=True
[diag/joint] J6(measured_dq=-0.0020,planned_end_dq=-0.0115)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=+0.0307)  J2(measured_dq=-0.0015,planned_end_dq=+0.0418)  J3(measured_dq=+0.0000,planned_end_dq=-0.0257)  J4(measured_dq=+0.0000,planned_end_dq=-0.0258)  J5(measured_dq=+0.0000,planned_end_dq=-0.0312)  J6(measured_dq=+0.0000,planned_end_dq=-0.0336)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J1(measured_dq=+0.0103,planned_end_dq=+0.0102)  J2(measured_dq=+0.0169,planned_end_dq=+0.0167)  J3(measured_dq=-0.0101,planned_end_dq=-0.0094)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J1(measured_dq=+0.0132,planned_end_dq=+0.0089)  J2(measured_dq=+0.0158,planned_end_dq=+0.0106)  J3(measured_dq=-0.0126,planned_end_dq=-0.0084)  J5(measured_dq=-0.0161,planned_end_dq=-0.0124)  J6(measured_dq=-0.0118,planned_end_dq=-0.0058)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=12.1ms  skip=1/9 waypoints (8.3ms dropped)
[diag/joint] J2(measured_dq=+0.0154,planned_end_dq=+0.0099)
[diag/remote_rtt] 12.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J1(measured_dq=+0.0106,planned_end_dq=+0.0073)  J2(measured_dq=+0.0146,planned_end_dq=+0.0095)  J3(measured_dq=-0.0112,planned_end_dq=-0.0080)  J6(measured_dq=-0.0135,planned_end_dq=-0.0116)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J2(measured_dq=+0.0116,planned_end_dq=+0.0098)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 13.6ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=-0.0021,planned_end_dq=+0.0123)  J6(measured_dq=+0.0003,planned_end_dq=+0.0106)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 13.2ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.140  J2=0.116  J3=0.112  J4=0.269  J5=0.141  J6=0.354 rad/s
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0129)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=12.3ms  skip=1/9 waypoints (8.3ms dropped)
[diag/joint] J4(measured_dq=-0.0022,planned_end_dq=+0.0112)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 13.0ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 11.9ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=-0.0168)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/joint] J4(measured_dq=+0.0000,planned_end_dq=+0.0148)
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/remote_rtt] 12.6ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
Home mode cancelled.
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
Home mode cancelled; recalibrating wrist reference.
Wrist delta calibration: 1/50
Wrist delta calibration: 2/50
Wrist delta calibration: 3/50
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.075  J2=0.145  J3=0.009  J4=0.127  J5=0.154  J6=0.279 rad/s
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
Wrist delta calibration: 17/50
[piper_worker] WARNING: cuRobo trajectory stale for >0.5s — replanning may have failed.
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
Wrist delta calibration: 33/50
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.000  J2=0.000  J3=0.000  J4=0.000  J5=0.000  J6=0.150 rad/s
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
[diag/remote_rtt] 15.8ms  ok=True
[curobo_worker] WARNING: joint velocity clamp engaged (cap=2.0 rad/s) -- plan output exceeded the safety ceiling and was capped.
[diag/solve] solve=16.3ms  skip=2/9 waypoints (16.7ms dropped)
[diag/joint] J2(measured_dq=+0.0106,planned_end_dq=+0.0151)  J4(measured_dq=+0.0444,planned_end_dq=+0.0524)  J6(measured_dq=+0.0306,planned_end_dq=+0.0356)
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=+0.0172)  J4(measured_dq=+0.0000,planned_end_dq=+0.0472)  J6(measured_dq=+0.0000,planned_end_dq=+0.0318)
[diag/joint] J4(measured_dq=+0.0111,planned_end_dq=+0.0000)  J6(measured_dq=+0.0092,planned_end_dq=+0.0113)
[diag/joint] J4(measured_dq=+0.0139,planned_end_dq=+0.0044)
[diag/joint] J1(measured_dq=-0.0043,planned_end_dq=-0.0149)  J4(measured_dq=+0.0068,planned_end_dq=-0.0101)  J6(measured_dq=+0.0100,planned_end_dq=+0.0126)
[diag/remote_rtt] 26.0ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.230  J2=0.233  J3=0.085  J4=0.705  J5=0.283  J6=0.474 rad/s
[diag/remote_rtt] 15.6ms  ok=True
[diag/remote_rtt] 12.5ms  ok=True
[diag/remote_rtt] 12.7ms  ok=True
[diag/solve] solve=16.7ms  skip=2/9 waypoints (16.7ms dropped)
[diag/remote_rtt] 12.8ms  ok=True
[diag/remote_rtt] 12.0ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.078  J2=0.193  J3=0.008  J4=0.049  J5=0.264  J6=0.181 rad/s
[diag/joint] J5(measured_dq=-0.0023,planned_end_dq=+0.0103)
[diag/remote_rtt] 24.4ms  ok=True
[diag/remote_rtt] 23.4ms  ok=True
[diag/joint] J2(measured_dq=+0.0000,planned_end_dq=-0.0132)  J5(measured_dq=+0.0000,planned_end_dq=+0.0174)
[diag/remote_rtt] 23.9ms  ok=True
[diag/solve] solve=24.7ms  skip=3/9 waypoints (25.0ms dropped)
[diag/joint] J4(measured_dq=+0.0020,planned_end_dq=+0.0102)  J5(measured_dq=+0.0000,planned_end_dq=-0.0120)
[diag/remote_rtt] 24.3ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.055  J2=0.145  J3=0.154  J4=0.327  J5=0.231  J6=0.302 rad/s
[diag/remote_rtt] 25.6ms  ok=True
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=-0.0230)  J3(measured_dq=-0.0024,planned_end_dq=-0.0208)  J5(measured_dq=+0.0000,planned_end_dq=+0.0262)
[diag/joint] J5(measured_dq=+0.0155,planned_end_dq=+0.0081)
[diag/joint] J1(measured_dq=-0.0094,planned_end_dq=+0.0103)  J3(measured_dq=-0.0136,planned_end_dq=+0.0003)  J5(measured_dq=+0.0093,planned_end_dq=-0.0188)
[diag/remote_rtt] 24.0ms  ok=True
[diag/joint] J1(measured_dq=+0.0000,planned_end_dq=-0.0179)  J3(measured_dq=-0.0049,planned_end_dq=-0.0179)  J5(measured_dq=+0.0012,planned_end_dq=+0.0180)
[diag/joint] J1(measured_dq=-0.0042,planned_end_dq=+0.0145)  J5(measured_dq=+0.0167,planned_end_dq=+0.0001)
[diag/remote_rtt] 24.2ms  ok=True
[diag/joint] J3(measured_dq=-0.0105,planned_end_dq=-0.0056)
[diag/solve] solve=24.5ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 24.5ms  ok=True
[diag/remote_rtt] 23.8ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (30 samples): J1=0.179  J2=0.163  J3=0.087  J4=0.172  J5=0.197  J6=0.270 rad/s
[diag/remote_rtt] 11.9ms  ok=True
[diag/remote_rtt] 25.9ms  ok=True
[diag/remote_rtt] 24.7ms  ok=True
[diag/solve] solve=25.0ms  skip=3/9 waypoints (25.0ms dropped)
[diag/remote_rtt] 24.8ms  ok=True
[diag/remote_rtt] 24.8ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (28 samples): J1=0.207  J2=0.180  J3=0.143  J4=0.401  J5=0.222  J6=0.219 rad/s
[diag/remote_rtt] 23.8ms  ok=True
[diag/remote_rtt] 12.1ms  ok=True
[diag/remote_rtt] 24.6ms  ok=True
[diag/solve] solve=13.2ms  skip=2/9 waypoints (16.7ms dropped)
[diag/remote_rtt] 29.8ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (31 samples): J1=0.095  J2=0.158  J3=0.132  J4=0.358  J5=0.369  J6=0.261 rad/s
[diag/joint] J4(measured_dq=+0.0027,planned_end_dq=+0.0125)
[diag/remote_rtt] 24.8ms  ok=True
[diag/remote_rtt] 25.1ms  ok=True
[diag/remote_rtt] 12.1ms  ok=True
[diag/joint] J2(measured_dq=+0.0025,planned_end_dq=-0.0113)
[diag/solve] solve=30.7ms  skip=4/9 waypoints (33.3ms dropped)
[diag/remote_rtt] 12.8ms  ok=True
[diag/joint] J2(measured_dq=-0.0022,planned_end_dq=+0.0118)
[diag/remote_rtt] 13.0ms  ok=True
[diag/jitter] measured |v| peak-to-peak over 250ms (29 samples): J1=0.074  J2=0.136  J3=0.040  J4=0.159  J5=0.419  J6=0.204 rad/s
[diag/remote_rtt] 16.5ms  ok=True
[diag/remote_rtt] 26.3ms  ok=True
[diag/remote_rtt] 35.3ms  ok=True
[diag/solve] solve=28.8ms  skip=3/9 waypoints (25.0ms dropped)