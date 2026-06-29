(.venv) (base) grasswonder@grasswonder:~/midas_hand/midas-piper-control$ python /home/grasswonder/midas_hand/midas-piper-control/data_collection/collect_raw_dataset.py
Foot pedal input: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
Press c in a display window to calibrate teleop neutral. Press q to quit.
Starting realsense worker...
Warning: RealSense reports USB 2.x; use USB 3 for reliable 640x480@30.
RealSense stream: 640x480 rgb8 @30fps serial=317422074281
Starting piper worker...
Piper payload set to 'half': no ack
Piper connected: False receive FPS: 0.0
[piper_worker] cuRobo MIT mode enabled.
Starting midas worker...
Delta wrist teleop loaded: config=/home/grasswonder/midas_hand/midas-piper-control/data_collection/config.yaml orientation_enabled=False scale_factor=0.3
Midas hand connected on /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FJXX-if00-port0; motor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Traceback (most recent call last):
  File "/home/grasswonder/midas_hand/pyAgxArm/pyAgxArm/protocols/can_protocol/comms/can_comm.py", line 208, in send
    self.send_bus.send(msg, timeout)
  File "/home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/can/thread_safe_bus.py", line 76, in send
    return self.__wrapped__.send(msg=msg, timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grasswonder/midas_hand/midas-piper-control/.venv/lib/python3.12/site-packages/can/interfaces/socketcan/socketcan.py", line 881, in send
    raise can.CanOperationError("Transmit buffer full")
can.exceptions.CanOperationError: Transmit buffer full

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/grasswonder/midas_hand/pyAgxArm/pyAgxArm/protocols/can_protocol/drivers/core/arm_driver_abstract.py", line 85, in _send_msg
    comm.send(data)
  File "/home/grasswonder/midas_hand/pyAgxArm/pyAgxArm/protocols/can_protocol/comms/can_comm.py", line 211, in send
    raise can.CanOperationError(
can.exceptions.CanOperationError: Failed to send CAN message on interface='socketcan', channel='can0': Transmit buffer full

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/utils.py", line 60, in worker_wrapper
    target(*args, init_done=init_done)
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/runner.py", line 44, in _piper
    run_piper_worker(cfg, sh, rec, trajectory_buffer, init_done=init_done)
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/piper_worker.py", line 172, in run_piper_worker
    _run_mit_step(api, grav_comp, shared, trajectory_buffer, config)
  File "/home/grasswonder/midas_hand/midas-piper-control/data_collection/collection/piper_worker.py", line 29, in _run_mit_step
    api.move_mit_torques([0.0] * 6, kp=0.0, kd=0.0, clamp=True)
  File "/home/grasswonder/midas_hand/midas-piper-control/scripts/piper_api/piper_api.py", line 370, in move_mit_torques
    self.move_mit_torque(
  File "/home/grasswonder/midas_hand/midas-piper-control/scripts/piper_api/piper_api.py", line 341, in move_mit_torque
    self.robot.move_mit(
  File "/home/grasswonder/midas_hand/pyAgxArm/pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py", line 1294, in move_mit
    self._send_msg(msg)
  File "/home/grasswonder/midas_hand/pyAgxArm/pyAgxArm/protocols/can_protocol/drivers/core/arm_driver_abstract.py", line 87, in _send_msg
    raise RuntimeError(
RuntimeError: Failed to send ArmMsgJointMitCtrl1 on channel 'can0': Failed to send CAN message on interface='socketcan', channel='can0': Transmit buffer full
Error: piper worker failed: Failed to send ArmMsgJointMitCtrl1 on channel 'can0': Failed to send CAN message on interface='socketcan', channel='can0': Transmit buffer full