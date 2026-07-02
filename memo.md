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








[curobo_worker] Connecting to remote solver 192.168.10.1:5556 ...
[curobo_worker] Connected to remote solver.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.127', '-0.261', '+0.269', '-1.664', '-0.659', '-1.731']  q_current=['-1.069', '+1.536', '-0.845', '-1.170', '-0.944', '-1.361']
[curobo_worker] remote solver connection error after 6.6ms: [Errno 104] Connection reset by peer -- reconnecting
[curobo_worker] Connecting to remote solver 192.168.10.1:5556 ...
[curobo_worker] Connected to remote solver.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.127', '-0.259', '+0.266', '-1.664', '-0.659', '-1.731']  q_current=['-1.069', '+1.536', '-0.845', '-1.170', '-0.944', '-1.361']
[curobo_worker] remote solver connection error after 0.7ms: [Errno 104] Connection reset by peer -- reconnecting
[curobo_worker] Connecting to remote solver 192.168.10.1:5556 ...
[curobo_worker] Connected to remote solver.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.127', '-0.257', '+0.263', '-1.664', '-0.659', '-1.731']  q_current=['-1.069', '+1.536', '-0.845', '-1.170', '-0.944', '-1.361']
[curobo_worker] remote solver connection error after 0.7ms: [Errno 104] Connection reset by peer -- reconnecting
[curobo_worker] Connecting to remote solver 192.168.10.1:5556 ...
[curobo_worker] Connected to remote solver.
[diag/cuRobo] Plan FAILED  tgt_eef=['+0.127', '-0.254', '+0.257', '-1.664', '-0.659', '-1.731']  q_current=['-1.069', '+1.536', '-0.845', '-1.170', '-0.944', '-1.361']