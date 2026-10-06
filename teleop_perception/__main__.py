import argparse
import os
import sys

from .config import load_config

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")


def main():
    """Command line entry: run, generate, simcheck, calibrate, evaluate."""
    ap = argparse.ArgumentParser(prog="teleop_perception")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=["sim", "real"], default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="live pose estimation")
    r.add_argument("--show", action="store_true")

    sub.add_parser("generate", help="write tagged MuJoCo models and perception configs into the simulator")

    s = sub.add_parser("simcheck", help="render tagged objects in MuJoCo and compare against ground truth")
    s.add_argument("--scenes", type=int, default=5)
    s.add_argument("--jitter", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag-shift-mm", type=float, default=0.0)
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--blur", type=float, default=0.0)
    s.add_argument("--save", default=None)

    k = sub.add_parser("calibrate", help="camera calibration with the target on the hand (extrinsics, intrinsics with the ChArUco board)")
    k.add_argument("--samples", default=None, help="re-solve from a saved calib/samples_*.npz without moving the arm")
    k.add_argument("--role", choices=["perception", "operator"], default="perception",
                   help="which cameras: the pose-estimation cameras or the operator (VR video) camera")

    e = sub.add_parser("evaluate", help="error summary of a logged run")
    e.add_argument("run_dir")

    a = ap.parse_args()
    if a.cmd == "evaluate":
        from .evaluate import summarize
        summarize(a.run_dir)
        return
    cfg = load_config(a.config, a.mode)
    if a.cmd == "run":
        from .runner import Runner
        Runner(cfg, show=a.show).run()
    elif a.cmd == "generate":
        from .mjcf import generate
        for p in generate(cfg):
            print(p)
    elif a.cmd == "simcheck":
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
            os.environ["MUJOCO_GL"] = "egl"
        from .simcheck import run
        run(cfg, a.scenes, a.jitter, a.seed, a.tag_shift_mm, a.noise, a.blur, a.save)
    elif a.cmd == "calibrate":
        from .calibrate import calibrate
        calibrate(cfg, a.samples, a.role)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
