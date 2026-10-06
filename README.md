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
| `cameras/sim.yaml`, `cameras/real.yaml` | camera list: frame source, intrinsics file, calibrated intrinsics file, extrinsics file, nominal pose (sim) |
| `intrinsics/*.yaml` | `fx fy cx cy dist`, or `fovy_deg` for sim cameras; `intrinsics: device` reads the RealSense factory values |
| `detection.yaml` | detector, per-object PnP gates, multi-camera fusion, still-averaging |
| `reference.yaml` | fixed table tags for the drift check |
| `calibration.yaml` | target on the hand (ChArUco board or single tag), its printed mount, intrinsics estimation, arm channel, motion, coverage planning |
| `network.yaml` | pose publisher, ground-truth listener (sim) |

Object frames are the MuJoCo body frames, so estimates compare directly to `getFreeBodyPose`. Tag poses use `pos` plus `face` (outward normal) and `up`.

## Workflow

```
python -m teleop_perception generate     # tagged models, tagged task, perception sim + pipeline configs in the simulator
python -m teleop_perception simcheck     # offline render in MuJoCo, error vs ground truth (--tag-shift-mm, --noise, --blur)
python -m teleop_perception calibrate    # arm moves the target through poses, writes calib/extrinsics_*, calib/intrinsics_* (ChArUco) and calib/reference_*
python -m teleop_perception calibrate --role operator   # same for the operator camera, plus the VR overlay viewpoint
python -m teleop_perception run          # live estimation, publish, log (--show for overlays)
python -m teleop_perception evaluate logs/<run>
```

`generate` writes perception variants of the tabletop sim, pipeline and robot configs next to their base configs (`sim.base_*` in `config.yaml`): tagged task, tagged hand with the cap, reference tags, perception cameras with their shared-memory streams, and `scene_objects` switched on so the avatar sends ground truth. Point the simulator's `config.yaml` at them.

## Calibration target

`calibration.yaml -> target` selects what is mounted on the hand during calibration:

- `charuco`: a ChArUco board (default 7x5 squares of 24 mm). Gives extrinsics and intrinsics.
- `gripper_tag`: a single 40 mm tag. Extrinsics only.

Both sit on a slide-on cap for the hand's end opposite the cable connector. `generate` builds the cap from the hand meshes (`mount`) and uses it in the sim hand. It writes `hardware/<target>_mount_print.stl` (target face on the bed) and, for the board, `hardware/charuco_board_print.pdf` to print at 100 %. Measure one printed square with calipers and set `charuco.square` if it differs. The target's exact pose on the cap does not matter (calibration solves it); the cap only has to sit still during the run and comes off afterwards. On hardware, set the extra mass in the Franka end-effector load while it is mounted.

With the board, `calibrate` runs in two stages. First random poses around the start, solved for each camera's pose. Then, per camera, poses that put the board over a grid of image cells at several distances and tilts (`coverage`), so the image edges are covered. Finally one joint solve per camera: camera pose, board-on-hand pose, focal lengths, principal point and `intrinsics.dist_terms` distortion terms (2 = k1 k2; 4 and 5 need more image coverage than the arm usually reaches). The report prints the 1-sigma of fx fy cx cy, the image coverage and the difference to the base model (sim: the ground truth from `fovy`; real: factory values). The calibrated intrinsics go to `intrinsics_calibrated` and replace the base intrinsics in every later run.

`calibrate` drives the arm through the avatar's absolute channel (POLICY authority) and releases it to HOLD at the end. Run it only when a camera or the reference tags moved; `run` checks the stored extrinsics against the reference tags and warns when they disagree.

## Operator camera (VR ghost overlay)

Cameras with `role: operator` in the camera file are not used for pose estimation. `calibrate --role operator` calibrates them in a separate run with the same board, using the `operator` overrides in `calibration.yaml` (start position, yaw search, coverage distances and box) since that camera sits elsewhere. Besides extrinsics and intrinsics it writes `calib/overlay_<mode>_<camera>.json` (static viewpoint pos/look_at/up, horizontal FOV, render size) and updates `operator.overlay_file` in the VR interface in place when that path exists; on a remote setup copy the overlay file over. The overlay is an ideal pinhole, so on hardware the video has to be undistorted to it.

In sim the report also prints the pixel error over the workspace: calibrated camera vs truth, overlay vs undistorted video, and overlay vs raw video.

## Sim ground truth for intrinsics

An intrinsics file may combine `fovy_deg` with `dist`. With `apply_distortion: true` on a shared-memory source the reader warps every sim frame with that distortion, so the sim behaves like a real lens with known parameters and the calibration report compares against them (`desk_cam` uses k1 0.06, k2 -0.03). Inject only k1/k2 when `dist_terms` is 2.

## Output

One msgpack datagram per update to `network.publish`:

```
{seq, timestamp_ns (frame capture), sent_ns,
 objects: [{name, position [x y z], quaternion [w x y z], n_tags, n_cams, rms_px, static, ambiguous}]}
```

Logs go to `logs/<stamp>_<mode>_<board>/`: `poses.csv` (per camera and fused, with ground truth and error in sim), `cameras.csv`, `meta.yaml`.
