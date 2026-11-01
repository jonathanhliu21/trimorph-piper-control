python scripts/teleop/quest_teleop.py --calibration $CAL --quest-port 5005 --with-orientation  --max-offset-mm 400 --home-joints-deg -17.8 112.3 -73.9 45.0 -39.4 25.0 --wrist-singularity-deg 3 --max-tracking-error-mm 150


python3 data_collection/collect_hand_only.py --dataset-root data_trimorph_quest   --hand-config config/xl330_test_config.yaml   --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0   --use-quest --quest-jump-threshold 0.05   --piper-start-joints-deg -17.8 112.3 -73.9 45.0 -39.4 25.0


python3 data_collection/postprocess_dataset.py data_trimorph_quest --episode 6 --calibration-json $CAL --no-camera


P=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

python3 data_collection/replay_episode.py 6 --dataset-root data_trimorph_quest --arm-diag \
  --hand-config config/xl330_test_config.yaml --hand-port $P --playback-speed 0.5



  python3 data_collection/postprocess_dataset.py data_trimorph_quest --episode 9 \
  --calibration-json $CAL --no-camera --pose-source mocap --mocap-topic /vicon/Yori/Yori --absolute

python3 data_collection/replay_episode.py 9 --dataset-root data_trimorph_quest --arm-diag \
  --hand-config config/xl330_test_config.yaml --hand-port $P --playback-speed 0.5