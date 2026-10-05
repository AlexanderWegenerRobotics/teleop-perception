# teleop-perception

Object pose estimation for the teleop twin from ArUco tags. The same pipeline runs in sim (frames from the simulator's shared memory) and on hardware (RealSense), selected by `mode` in `config/config.yaml`. Poses are published over UDP; in sim every pose is also logged against MuJoCo ground truth.

## Install

```
conda activate teleop
pip install -e .[sim]        # add [real] on the machine with the RealSense cameras
```

## Configs

`config/config.yaml` is the master. It picks `mode` (sim | real), the board file and the camera file per mode, and points to one file per topic:

| file | content |
|---|---|
| `boards/<board>.yaml` | tag dictionary, tag layouts per object type (pose on the object, size), object instances with their tag ids |
| `cameras/sim.yaml`, `cameras/real.yaml` | camera list: frame source, intrinsics file, extrinsics file, nominal pose (sim) |
| `intrinsics/*.yaml` | `fx fy cx cy dist`, or `fovy_deg` for sim cameras; `intrinsics: device` reads the RealSense factory values |
| `detection.yaml` | detector, per-object PnP gates, multi-camera fusion, still-averaging |
| `reference.yaml` | fixed table tags for the drift check |
| `calibration.yaml` | gripper tag, arm channel, calibration motion |
| `network.yaml` | pose publisher, ground-truth listener (sim) |

Object frames are the MuJoCo body frames, so estimates compare directly to `getFreeBodyPose`. Tag poses use `pos` plus `face` (outward normal) and `up`.

## Workflow

```
python -m teleop_perception generate     # tagged models, tagged task, perception sim + pipeline configs in the simulator
python -m teleop_perception simcheck     # offline render in MuJoCo, error vs ground truth (--tag-shift-mm, --noise, --blur)
python -m teleop_perception calibrate    # arm moves the gripper tag through poses, writes calib/extrinsics_* and calib/reference_*
python -m teleop_perception run          # live estimation, publish, log (--show for overlays)
python -m teleop_perception evaluate logs/<run>
```

`generate` writes perception variants of the tabletop sim, pipeline and robot configs next to their base configs (`sim.base_*` in `config.yaml`): tagged task, tagged hand with the cap, reference tags, perception cameras with their shared-memory streams, and `scene_objects` switched on so the avatar sends ground truth. Point the simulator's `config.yaml` at them.

The gripper tag sits on a slide-on cap for the hand's end opposite the cable connector. `generate` builds it from the hand meshes (`calibration.yaml -> gripper_tag.mount`), uses it in the sim hand and writes `hardware/gripper_tag_mount_print.stl` (tag face on the bed). The tag's exact position on the cap does not matter, calibration solves it; the cap only has to sit still during the run.

`calibrate` drives the arm through the avatar's absolute channel (POLICY authority) and releases it to HOLD at the end. Run it only when a camera or the reference tags moved; `run` checks the stored extrinsics against the reference tags and warns when they disagree.

## Output

One msgpack datagram per update to `network.publish`:

```
{seq, timestamp_ns (frame capture), sent_ns,
 objects: [{name, position [x y z], quaternion [w x y z], n_tags, n_cams, rms_px, static, ambiguous}]}
```

Logs go to `logs/<stamp>_<mode>_<board>/`: `poses.csv` (per camera and fused, with ground truth and error in sim), `cameras.csv`, `meta.yaml`.
