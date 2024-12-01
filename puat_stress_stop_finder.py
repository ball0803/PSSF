import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, 
    QHBoxLayout, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, 
    QLineEdit, QMessageBox, QFormLayout, QSlider, QHeaderView
)
from PyQt5.QtGui import QPixmap, QImage,QIcon
from PyQt5.QtCore import Qt
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

def rgb_to_grayscale_gradient(
    image: np.ndarray, 
    gradient_points: List[Tuple[int, Tuple[int, int, int]]]
) -> np.ndarray:
    """
    Converts an RGB image to a grayscale image based on specified gradient points.
    """
    positions, colors = zip(*gradient_points)
    positions = np.array(positions)
    colors = np.array(colors, dtype=np.float32)

    grayscales = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    h, w, _ = image.shape
    grayscale_image = np.zeros((h, w), dtype=np.float32)

    def process_row(row_idx: int) -> np.ndarray:
        row = image[row_idx].astype(np.float32)
        distances = np.linalg.norm(row[:, None] - colors, axis=2)
        closest_idx = np.argsort(distances, axis=1)[:, :2]

        low_idx, high_idx = closest_idx[:, 0], closest_idx[:, 1]
        low_gray, high_gray = grayscales[low_idx], grayscales[high_idx]

        pixel_distances = distances[np.arange(len(distances)), low_idx]
        diff_distances = distances[np.arange(len(distances)), high_idx] - pixel_distances
        t = pixel_distances / (diff_distances + 1e-6)

        grayscale_row = (1 - t) * low_gray + t * high_gray
        return grayscale_row

    with ThreadPoolExecutor() as executor:
        rows_grayscale = list(executor.map(process_row, range(h)))

    for i in range(h):
        grayscale_image[i] = rows_grayscale[i]

    return np.clip(grayscale_image, 0, 255).astype(np.uint8)

def create_circular_mask(image_shape: Tuple[int, int], radius_factor: float = 0.5) -> np.ndarray:
    """
    Creates a circular mask for an image.
    """
    h, w = image_shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * radius_factor)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255

class GradientMaskApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PUAT AXLE Stress spot finder")
        self.setWindowIcon(QIcon("app_icon.ico"))
        self.setGeometry(100, 100, 600, 800)
        self.mask_radius: float = 0.4
        self.image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None

        self.init_ui()

    def init_ui(self) -> None:
        main_layout = QVBoxLayout()
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #000;")
        self.image_label.setMinimumSize(600, 400)
        main_layout.addWidget(self.image_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Contour", "Diameter (pixels)", "Diameter (real Y-axis)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        main_layout.addWidget(self.table)

        input_layout = QFormLayout()

        self.min_input = QLineEdit("10")
        self.min_input.setPlaceholderText("Min Diameter")
        min_label = QLabel("Minimum Diameter: ")

        self.max_input = QLineEdit("100")
        self.max_input.setPlaceholderText("Max Diameter")
        max_label = QLabel("Max Diameter: ")

        self.real_y_min_input = QLineEdit("0")
        self.real_y_min_input.setPlaceholderText("Real Y Min")
        real_min_label = QLabel("Real Y Min: ")

        self.real_y_max_input = QLineEdit("360")
        self.real_y_max_input.setPlaceholderText("Real Y Max")
        real_max_label = QLabel("Real Y Max: ")

        input_layout.addRow(min_label, self.min_input)
        input_layout.addRow(max_label, self.max_input)
        input_layout.addRow(real_min_label, self.real_y_min_input)
        input_layout.addRow(real_max_label, self.real_y_max_input)
        main_layout.addLayout(input_layout)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 100)
        self.radius_slider.setValue(int(self.mask_radius * 100))
        self.radius_slider.setTickInterval(1)
        self.radius_slider.setTickPosition(QSlider.TicksBelow)
        self.radius_slider.valueChanged.connect(self.update_radius_value)
        self.radius_label = QLabel(f"Mask Radius: {self.mask_radius:.2f}")
        main_layout.addWidget(self.radius_label)
        main_layout.addWidget(self.radius_slider)

        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.process_button)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_radius_value(self) -> None:
        self.mask_radius = self.radius_slider.value() / 100
        self.radius_label.setText(f"Mask Radius: {self.mask_radius:.2f}")
        if self.image is not None:
            self.process_image()

    def load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.image = cv.imread(file_name, cv.IMREAD_COLOR)
            self.show_image(self.image)

    def show_image(self, image: np.ndarray) -> None:
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def process_image(self) -> None:
        if self.image is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        try:
            min_diameter = float(self.min_input.text())
            max_diameter = float(self.max_input.text())
            real_y_min = float(self.real_y_min_input.text())
            real_y_max = float(self.real_y_max_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input values.")
            return

        gradient_points = [
            (0, (255, 255, 255)), (10, (192, 192, 192)), 
            (30, (0, 8, 255)), (50, (255, 255, 0)), (70, (255, 0, 0))
        ]
        gray = rgb_to_grayscale_gradient(cv.cvtColor(self.image, cv.COLOR_BGR2RGB), gradient_points)

        mask = create_circular_mask(gray.shape, radius_factor=self.mask_radius)
        isolated_image = gray.copy()
        isolated_image[mask == 0] = 255
        _, binary = cv.threshold(isolated_image, 220, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        h, _ = gray.shape

        self.processed_image = self.image.copy()
        self.table.setRowCount(0)
        drawn_contour = list()
        for idx, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area > 1:
                y_coords = [point[0][1] for point in contour]
                min_y, max_y = min(y_coords), max(y_coords)
                vertical_diameter = max_y - min_y

                if min_diameter <= vertical_diameter <= max_diameter:
                    drawn_contour.append(contour)
                    real_y_diameter = ((vertical_diameter / h) * (real_y_max - real_y_min)) + real_y_min
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QTableWidgetItem(str(idx+1)))
                    self.table.setItem(row_position, 1, QTableWidgetItem(str(vertical_diameter)))
                    self.table.setItem(row_position, 2, QTableWidgetItem(f"{real_y_diameter:.2f}"))
                    
                    moments = cv.moments(contour)
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                    else:
                        cx, cy = contour[0][0][0], contour[0][0][1]  # Default to the first point

                    cv.putText(self.processed_image, str(idx+1), (cx-20, cy-5), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        cv.drawContours(self.processed_image, drawn_contour, -1, (0, 255, 0), 2)
        # self.table.resizeColumnsToContents()
        self.show_image(self.processed_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GradientMaskApp()
    window.show()
    sys.exit(app.exec_())
