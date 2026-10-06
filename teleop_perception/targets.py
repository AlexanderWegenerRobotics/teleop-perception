import cv2
import numpy as np

from .detection import TagDetector

PX_PER_SQUARE = 128


class TagTarget:
    """Single gripper tag: four corners, extrinsics only."""

    kind = "gripper_tag"
    supports_intrinsics = False

    def __init__(self, spec, dictionary, detector_params):
        self.id = int(spec["id"])
        self.size = float(spec["size"])
        self.dictionary = dictionary
        self.detector = TagDetector(dictionary, detector_params)
        self.min_points = 4
        h = self.size / 2.0
        self.points = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
        self.extent = (self.size, self.size)

    def detect(self, gray):
        """Image points in target point order, NaN where not seen."""
        out = np.full((len(self.points), 2), np.nan)
        d = self.detector.detect(gray)
        if self.id in d:
            out[:] = d[self.id]
        return out


class CharucoTarget:
    """ChArUco board: inner chessboard corners, intrinsics and extrinsics."""

    kind = "charuco"
    supports_intrinsics = True

    def __init__(self, spec):
        self.dictionary = spec["dictionary"]
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        nx, ny = (int(v) for v in spec["squares"])
        self.square = float(spec["square"])
        self.marker = float(spec["marker"])
        self.margin = float(spec.get("margin", self.square / 2.0))
        ids = np.arange(int(spec.get("first_id", 0)), int(spec.get("first_id", 0)) + (nx * ny) // 2)
        self.board = cv2.aruco.CharucoBoard((nx, ny), self.square, self.marker, d, ids)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.min_points = int(spec.get("min_corners", 6))
        self.size_xy = (nx * self.square, ny * self.square)
        self.extent = (self.size_xy[0] + 2 * self.margin, self.size_xy[1] + 2 * self.margin)
        P = self.board.getChessboardCorners().astype(float)
        self.points = np.c_[P[:, 0] - self.size_xy[0] / 2.0, self.size_xy[1] / 2.0 - P[:, 1], np.zeros(len(P))]
        self.offset = self._corner_offset()

    def _corner_offset(self, px=24, s=8):
        """Correction for this OpenCV build's corner convention, measured on an exactly downsampled board image."""
        nx, ny = self.board.getChessboardSize()
        img = self.board.generateImage((nx * px * s, ny * px * s), marginSize=0, borderBits=1)
        img = cv2.copyMakeBorder(img, px * s, px * s, px * s, px * s, cv2.BORDER_CONSTANT, value=255)
        low = cv2.GaussianBlur(cv2.resize(img, None, fx=1.0 / s, fy=1.0 / s, interpolation=cv2.INTER_AREA), (0, 0), 0.6)
        cc, ci, _, _ = self.detector.detectBoard(low)
        if ci is None or len(ci) < 4:
            return np.zeros(2)
        P = self.board.getChessboardCorners()[np.ravel(ci).astype(int), :2] / self.square * px + px - 0.5
        return -np.mean(cc.reshape(-1, 2) - P, axis=0)

    def detect(self, gray):
        """Image points in target point order, NaN where not seen."""
        out = np.full((len(self.points), 2), np.nan)
        cc, ci, _, _ = self.detector.detectBoard(gray)
        if ci is not None and len(ci) > 0:
            out[np.ravel(ci).astype(int)] = cc.reshape(-1, 2) + self.offset
        return out

    def image(self, px_per_square=PX_PER_SQUARE):
        """Board image including the white margin, top row first."""
        nx, ny = self.board.getChessboardSize()
        img = self.board.generateImage((nx * px_per_square, ny * px_per_square), marginSize=0, borderBits=1)
        m = int(round(self.margin / self.square * px_per_square))
        return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)


def make_target(cfg):
    """Calibration target selected in calibration.yaml."""
    c = cfg.calibration
    kind = c.get("target", "gripper_tag")
    if kind == "charuco":
        return CharucoTarget(c["charuco"])
    return TagTarget(c["gripper_tag"], cfg.board.dictionary, cfg.detection.get("detector", {}))
