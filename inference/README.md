# Trimorph policy deployment

This ports the MIDAS predictor/protocol/executor into this repository. The GPU
machine runs the shared diffusion-policy predictor. The robot machine runs the
Trimorph/Piper executor and the same OAK camera backend/settings as collection.
It does not import the MIDAS controller repository.

## Install

On the robot machine, in its control environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../trimorph_hand_api
python -c 'import midas_hand_api; print(midas_hand_api.__file__)'
```

The Trimorph fork retains the `midas-hand` distribution name, `midas_hand_api`
module, and `MidasHand` class. The executor defaults to the sibling
`../trimorph_hand_api` source directory and checks the imported module location.
Use `--hand-api-repo` if it lives elsewhere. Keep the GPU training environment
separate from the hardware dependencies.

On the GPU machine, install the dependencies for your `diffusion-policies`
checkout. Run from this repository's root (the checkout can live on each machine
at a different path):

```bash
python inference/predictor/predictor.py \
  --policy-repo /path/to/diffusion-policies \
  --checkpoint /path/to/trimi/checkpoints/latest.ckpt \
  --device cuda:0
```

No stale machine-specific checkpoint is selected automatically. The predictor
loads the checkpoint's normalizer and sampling settings. To experiment with
fewer denoising passes, pass `--num-inference-steps 8` and evaluate policy quality.

## Test the connection without hardware

With the predictor running:

```bash
python inference/executor/executor.py \
  --predictor GPU_HOST --fake --max-cycles 20
```

Fake mode uses synthetic images/state and does not connect to the hand, Piper,
or camera. The real GPU model still runs. Unit tests need no GPU or devices:

```bash
python -m unittest discover -s tests -p 'test_inference.py'
```

## Execute on Trimorph and Piper

```bash
python inference/executor/executor.py \
  --predictor GPU_HOST \
  --hand-api-repo /path/to/trimorph_hand_api \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/YOUR_TRIMORPH_HAND \
  --piper-channel can0 --hz 15
```

This enables and commands the actual hand and arm. There is no automatic home
motion: start from the intended task pose. Hand setup follows the repository's
replay path: mode 3 position control, hold measured position before enabling
torque, configured velocity/acceleration profiles, then nonblocking position
writes. It does not start the collection force-feedback workers. On exit it
shuts down the hand and disconnects Piper; it does not retract/home the arm.

Piper firmware/interface/channel/installation/payload and OAK image size, flip,
brightness, and anti-banding defaults come from
`data_collection/collection/config.py`. Override them through `--help` options
when collection used different settings. The hand API uses the YAML baudrate,
calibration, gains, and limits. It verifies the connected motor models.

## Coordinate contract

- State and absolute actions are `[XYZ_m, rotvec_rad, hand_rad]`, shape `[19]`.
- Hand elements are exactly motor IDs `0..12`, in leader coordinates. Use the
  same signs, home offsets, and zero calibration as collection. There is no
  follower scaling, thumb retargeting, or passive-joint/16-DOF expansion.
- `read_pos()` supplies calibrated hand positions. `set_positions()` applies
  hardware signs/home offsets once and clips to configured joint limits.
- Piper feedback Euler xyz angles are converted to rotation vectors before
  inference; predicted rotation vectors are converted back to Euler xyz for
  Piper commands. They must never be passed through as interchangeable angles.
- Camera frames are flipped like collection and converted from OAK BGR to RGB.
- Quest-derived training poses should be processed with the actual
  `--piper-start-pose` so orientation-dependent frame conversion agrees with
  execution. The inference protocol does not carry pose-frame metadata.
- `--hz` must match the synchronized training sample rate after downsampling.
  A 30 FPS dataset with stride 2 uses 15 Hz, not the raw hand measurement rate.

## Timing and limitations

This port retains synchronous request/reply execution. It captures an
observation, waits for the predictor, and then executes the returned chunk.
Inference therefore remains a pause between chunks. It does **not** solve that
pause by speeding up playback or appending stale predictions to a queue.

Each cycle logs observation time, full predictor `infer` time, request/reply
`rtt`, execution time, and action deadline overruns. `rtt` includes prediction;
do not add the two values. `observe + rtt` is the approximate inter-chunk overhead
on top of the scheduled action interval. Predictor timing covers CPU output
conversion as well as model execution. Socket and camera timeouts bound waits;
invalid/nonfinite outputs or mismatched sequence numbers abort execution.

Continuous execution requires a separate asynchronous controller with
observation timestamps, latency compensation, and correct relative-action
anchoring. No hardware performance or policy success is established by unit
tests or the synthetic link test.
