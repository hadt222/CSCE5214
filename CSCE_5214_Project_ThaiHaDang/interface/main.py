import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from collections import defaultdict
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter
from typing import Any, Tuple, List
from ultralytics import YOLO

# Constants
TICKSIZE = 12
FONT_COLOR = "#4A4B52"
BACKGROUND_COLOR = "#FFFCFA"
SOURCE_VIDEO = "vehicles.mp4"
MPS_TO_KPH = 3.6

# Matplotlib theme configuration
MATPLOTLIB_THEME = {
    "axes.labelcolor": FONT_COLOR,
    "axes.labelsize": TICKSIZE,
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.titlesize": 16,
    "axes.grid": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.labelsize": TICKSIZE,
    "xtick.color": FONT_COLOR,
    "ytick.labelsize": TICKSIZE,
    "ytick.color": FONT_COLOR,
    "figure.facecolor": BACKGROUND_COLOR,
    "figure.edgecolor": BACKGROUND_COLOR,
    "figure.titlesize": 16,
    "figure.dpi": 72,
    "text.color": FONT_COLOR,
    "font.size": TICKSIZE,
    "font.family": "Serif",
}
plt.rcParams.update(MATPLOTLIB_THEME)

def imshow(img: NDArray, figsize: Tuple[int, int] = (11, 7)) -> None:
    """Display an image using matplotlib."""
    plt.figure(figsize=figsize, tight_layout=True)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

class PointMarker:
    """Mark points on an image using Left Mouse Button click."""
    
    def __init__(self, window: str = "Image") -> None:
        self._window = window
        self._points: List[Tuple[int, int]] = []

    @property
    def points(self) -> List[Tuple[int, int]]:
        return self._points

    def mark(self, image: NDArray, inplace: bool = False) -> List[Tuple[int, int]]:
        """Mark points on the image through mouse clicks."""
        img = image if inplace else image.copy()
        cv.namedWindow(self._window, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self._window, self._record_point, param=img)

        while True:
            cv.imshow(self._window, img)
            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()
        return self._points

    def _record_point(self, event: int, x: int, y: int, flags: int, image: Any | None) -> None:
        """Handle mouse click events to record points."""
        if event == cv.EVENT_LBUTTONDOWN:
            self._points.append((x, y))
            if image is not None:
                cv.drawMarker(image, (x, y), (0, 123, 255), cv.MARKER_CROSS, 20, 4, cv.LINE_AA)

class Cam2WorldMapper:
    """Maps points from image to world coordinates using perspective transform."""
    
    def __init__(self) -> None:
        self.M: NDArray | None = None

    def find_perspective_transform(self, image_pts: ArrayLike, world_pts: ArrayLike) -> NDArray:
        """Calculate perspective transform matrix."""
        image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        world_pts = np.asarray(world_pts, dtype=np.float32).reshape(-1, 1, 2)
        self.M = cv.getPerspectiveTransform(image_pts, world_pts)
        return self.M

    def map(self, image_pts: ArrayLike) -> NDArray:
        """Map image points to world coordinates."""
        if self.M is None:
            raise ValueError("Perspective transform not estimated")
        image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv.perspectiveTransform(image_pts, self.M).reshape(-1, 2)

class Speedometer:
    """Estimates speed of objects in world coordinates."""
    
    def __init__(self, mapper: Cam2WorldMapper, fps: int, unit: float = MPS_TO_KPH) -> None:
        self._mapper = mapper
        self._fps = fps
        self._unit = unit
        self._speeds: defaultdict[int, List[int]] = defaultdict(list)

    @property
    def speeds(self) -> defaultdict[int, List[int]]:
        return self._speeds

    def update_with_trace(self, idx: int, image_trace: NDArray) -> None:
        """Update speed estimates based on object trace."""
        if len(image_trace) > 1:
            world_trace = self._mapper(image_trace)
            dx, dy = np.median(np.abs(np.diff(world_trace, axis=0)), axis=0)
            ds = np.linalg.norm((dx, dy))
            self._speeds[idx].append(int(ds * self._fps * self._unit))

    def get_current_speed(self, idx: int) -> int:
        """Get the current speed for a given object ID."""
        return self._speeds[idx][-1] if self._speeds[idx] else 0

def setup_video_processing() -> Tuple[Cam2WorldMapper, Speedometer, sv.VideoInfo]:
    """Setup video processing components."""
    # Read first frame
    cap = cv.VideoCapture(SOURCE_VIDEO)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {SOURCE_VIDEO}")
    
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame from video")

    # Setup perspective transform
    image_pts = [(800, 410), (1125, 410), (1920, 850), (0, 850)]
    world_pts = [(0, 0), (32, 0), (32, 140), (0, 140)]
    
    mapper = Cam2WorldMapper()
    mapper.find_perspective_transform(image_pts, world_pts)

    # Setup video info and speedometer
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    speedometer = Speedometer(mapper, video_info.fps)

    # Display initial frame with annotations
    img = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    color1 = sv.Color.from_hex("#004080")
    color2 = sv.Color.from_hex("#f78923")
    poly = np.array(((800, 410), (1125, 410), (1920, 850), (0, 850)))
    
    img = sv.draw_filled_polygon(img, poly, color1, 0.5)
    img = sv.draw_polygon(img, poly, sv.Color.WHITE, 12)
    for point, label in [((800, 370), "A"), ((1125, 370), "B"), 
                        ((1880, 780), "C"), ((40, 780), "D")]:
        img = sv.draw_text(img, label, sv.Point(*point), color2, 2, 6)
    
    imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    
    return mapper, speedometer, video_info

def process_video(mapper: Cam2WorldMapper, speedometer: Speedometer, video_info: sv.VideoInfo) -> None:
    """Process video frames and annotate with tracking and speed information."""
    colors = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", 
              "#004080", "#003366", "#00264d")
    color_palette = sv.ColorPalette(list(map(sv.Color.from_hex, colors)))

    poly = np.array([(0, 410), (1920, 410), (1920, 900), (0, 900)])
    zone = sv.PolygonZone(poly, (sv.Position.TOP_CENTER, sv.Position.BOTTOM_CENTER))

    bbox_annotator = sv.BoxAnnotator(color=color_palette, thickness=2, 
                                   color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(color=color_palette, position=sv.Position.CENTER,
                                      thickness=2, trace_length=video_info.fps,
                                      color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.RichLabelAnnotator(color=color_palette, border_radius=2,
                                          font_size=16, color_lookup=sv.ColorLookup.TRACK,
                                          text_padding=6)

    yolo = YOLO("yolo11m.pt", task="detect")
    output_video = "vehicle_annotated.mp4"
    width, height = video_info.resolution_wh
    width, height = round(width / 32) * 32, round(height / 32) * 32
    classes = [2, 5, 7]  # Car, Bus, Truck
    conf = 0.4

    with sv.VideoSink(output_video, video_info) as sink:
        for frame in sv.get_video_frames_generator(SOURCE_VIDEO):
            result = yolo.track(
                frame, classes=classes, conf=conf, imgsz=(height, width),
                persist=True, verbose=False, tracker="bytetrack.yaml"
            )
            detection = sv.Detections.from_ultralytics(result[0])
            detection = detection[zone.trigger(detections=detection)]

            trace_ids = detection.tracker_id
            speeds, labels = [], []

            for trace_id in trace_ids or []:
                image_trace = trace_annotator.trace.get(trace_id)
                speedometer.update_with_trace(int(trace_id), image_trace)
                current_speed = speedometer.get_current_speed(int(trace_id))
                speeds.append(current_speed)
                labels.append(f"#{trace_id} {current_speed} km/h")

            frame = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
            frame = bbox_annotator.annotate(frame, detection)
            frame = trace_annotator.annotate(frame, detection)
            frame = label_annotator.annotate(frame, detection, labels=labels)
            sink.write_frame(frame)

def main() -> None:
    """Main function to run the vehicle speed tracking application."""
    try:
        mapper, speedometer, video_info = setup_video_processing()
        process_video(mapper, speedometer, video_info)
        print(f"Video processing completed. Output saved to vehicle_annotated.mp4")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()