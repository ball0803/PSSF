import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
    QMessageBox,
    QFormLayout,
    QSlider,
    QHeaderView,
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor


def crop_to_roi(image: np.ndarray) -> np.ndarray:
    """
    Crop the image to the region of interest (ROI).
    Adjust the coordinates (x, y, w, h) based on the specific requirement.
    """
    # Define the ROI (x, y, width, height)
    x_start = 557  # Replace with the actual x start
    y_start = 570  # Replace with the actual y start
    crop_w = 790  # ความกว้างของภาพที่ต้องการครอป
    crop_h = 350  # ความสูงของภาพที่ต้องการครอป

    # Perform cropping
    cropped_image = image[y_start : y_start + crop_h, x_start : x_start + crop_w]
    return cropped_image


def load_image(self) -> None:
    file_name, _ = QFileDialog.getOpenFileName(
        self, "Open Image File", "", "Images (*.png *.jpg *.bmp)"
    )
    if file_name:
        self.image = cv.imread(file_name, cv.IMREAD_COLOR)

        # Automatically crop the image
        self.image = crop_to_roi(self.image)

        self.show_image(self.image)


def rgb_to_grayscale_gradient(
    image: np.ndarray, gradient_points: List[Tuple[int, Tuple[int, int, int]]]
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
        diff_distances = (
            distances[np.arange(len(distances)), high_idx] - pixel_distances
        )
        t = pixel_distances / (diff_distances + 1e-6)

        grayscale_row = (1 - t) * low_gray + t * high_gray
        return grayscale_row

    with ThreadPoolExecutor() as executor:
        rows_grayscale = list(executor.map(process_row, range(h)))

    for i in range(h):
        grayscale_image[i] = rows_grayscale[i]

    return np.clip(grayscale_image, 0, 255).astype(np.uint8)


def create_circular_mask(
    image_shape: Tuple[int, int], radius_factor: float = 0.5
) -> np.ndarray:
    """
    Creates a circular mask for an image.
    """
    h, w = image_shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * radius_factor)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255


from PyQt5.QtWidgets import QComboBox  # นำเข้า QComboBox

from PyQt5.QtCore import QTimer  # นำเข้า QTimer
import datetime  # นำเข้า datetime สำหรับดึงวันและเวลา

import csv
from PyQt5.QtWidgets import QFileDialog

import os


class GradientMaskApp(QMainWindow):

    def save_to_csv(self):
        # สร้างโฟลเดอร์ 'results' หากยังไม่มี
        results_folder = "results"

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Open a "Save As" dialog to get the file name and path
        options = QFileDialog.Options()
        options |= (
            QFileDialog.DontUseNativeDialog
        )  # Optional: Use non-native dialog if needed
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            "results/inspection_results.csv",  # Default directory and file name
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )

        if not file_path:  # If the user cancels the dialog
            QMessageBox.warning(
                self, "Error", "No file name specified. Operation canceled."
            )
            return

        # Check if the directory exists and create it if needed
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        write_header = not os.path.exists(file_path)  # เขียน header ถ้าไฟล์ยังไม่มี

        try:
            # เปิดไฟล์ในโหมด append ('a') และเขียนข้อมูลใน QTableWidget ลงในไฟล์ CSV
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                # เขียน Header ถ้าเป็นการสร้างไฟล์ใหม่
                if write_header:
                    headers = [
                        "Axle Type",
                        "Axle Number",
                        "Memo",
                        "Date",
                        "Time",
                        *[
                            self.table.horizontalHeaderItem(i).text()
                            for i in range(self.table.columnCount())
                        ],
                    ]
                    writer.writerow(headers)

                # เพิ่มข้อมูล Axle Type, Axle Number, Memo, Date, และ Time
                axle_type = self.axle_type_combo.currentText()
                axle_number = self.axle_number_input.text()
                memo = self.memo_input.currentText()
                now = datetime.datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")

                # เขียนข้อมูลในแต่ละแถว
                for row in range(self.table.rowCount()):
                    row_data = [
                        axle_type,
                        axle_number,
                        memo,
                        current_date,
                        current_time,
                    ]
                    for col in range(self.table.columnCount()):
                        cell_data = (
                            self.table.item(row, col).text()
                            if self.table.item(row, col)
                            else ""
                        )
                        row_data.append(cell_data)
                    writer.writerow(row_data)

            QMessageBox.information(
                self, "Success", f"Data saved successfully to {file_path}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save data: {str(e)}")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PUAT AXLE Stress spot finder")
        self.setWindowIcon(QIcon("app_icon.ico"))
        self.setGeometry(100, 100, 600, 800)
        self.mask_radius: float = 0.4
        self.image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None

        self.init_ui()

        # ตั้งค่า QTimer เพื่ออัปเดตเวลาแบบเรียลไทม์
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_datetime)
        self.timer.start(1000)  # อัปเดตทุกๆ 1000 มิลลิวินาที (1 วินาที)

    def init_ui(self) -> None:
        main_layout = QVBoxLayout()

        self.datetime_label = QLabel("")
        self.datetime_label.setAlignment(Qt.AlignCenter)
        self.datetime_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.datetime_label)

        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #e0e0e0; border: 1px solid #000;"
        )
        self.image_label.setMinimumSize(600, 400)
        main_layout.addWidget(self.image_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                "Contour",
                "Diameter (pixels)",
                "Degree of Rotation (real Y-axis)",
                "Length of Defect (mm)",
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        main_layout.addWidget(self.table)

        input_layout = QFormLayout()

        # เพิ่ม ComboBox สำหรับเลือกชนิดของเพลา
        self.axle_type_combo = QComboBox()
        self.axle_type_combo.addItems(
            ["Select Axle Type", "Type A", "Type B", "Type C"]
        )
        self.axle_type_combo.currentTextChanged.connect(self.update_radius_by_axle_type)
        axle_label = QLabel("Axle Type:")
        input_layout.addRow(axle_label, self.axle_type_combo)

        # เพิ่มช่องสำหรับกรอก Axle Number
        self.axle_number_input = QLineEdit("")
        self.axle_number_input.setPlaceholderText("Enter Axle Number")
        axle_number_label = QLabel("Axle Number:")
        input_layout.addRow(axle_number_label, self.axle_number_input)

        # เพิ่มช่องสำหรับ memo
        self.memo_input = QComboBox()
        self.memo_input.addItems(["Select Memo", "Bearings", "No Bearings"])
        memo_label = QLabel("Memo:")
        input_layout.addRow(memo_label, self.memo_input)

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

        self.radius_input = QLineEdit("85")  # สมมติค่าเริ่มต้นเป็น 85 mm
        self.radius_input.setPlaceholderText("Radius (mm)")
        radius_label = QLabel("Radius (mm):")

        input_layout.addRow(radius_label, self.radius_input)

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

        self.update_datetime()  # อัปเดตวันและเวลาเมื่อเริ่มต้น

        self.save_button = QPushButton("Save as CSV")
        self.save_button.clicked.connect(self.save_to_csv)
        button_layout.addWidget(self.save_button)

    def update_datetime(self) -> None:
        """
        อัปเดตวันและเวลาใน QLabel
        """
        now = datetime.datetime.now()
        self.datetime_label.setText(
            f"Date: {now.strftime('%Y-%m-%d')}  Time: {now.strftime('%H:%M:%S')}"
        )

    def update_radius_by_axle_type(self) -> None:
        """
        อัปเดตค่า Radius ตามชนิดของเพลา
        """
        axle_type = self.axle_type_combo.currentText()
        if axle_type == "Type A":
            self.radius_input.setText("85")  # ตัวอย่างค่า Radius สำหรับ Type A
        elif axle_type == "Type B":
            self.radius_input.setText("100")  # ตัวอย่างค่า Radius สำหรับ Type B
        elif axle_type == "Type C":
            self.radius_input.setText("120")  # ตัวอย่างค่า Radius สำหรับ Type C
        else:
            self.radius_input.setText("")  # เคลียร์ค่าเมื่อไม่ได้เลือกชนิดเพลา

    def update_radius_value(self) -> None:
        self.mask_radius = self.radius_slider.value() / 100
        self.radius_label.setText(f"Mask Radius: {self.mask_radius:.2f}")
        if self.image is not None:
            self.process_image()

    def load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_name:
            self.image = cv.imread(file_name, cv.IMREAD_COLOR)
            crop_image = crop_to_roi(self.image)
            if len(crop_image) == 0:
                QMessageBox.warning(
                    self, "Error", "Image too small can't crop any futher"
                )
            else:
                self.image = crop_image
            self.show_image(self.image)

    def show_image(self, image: np.ndarray) -> None:
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
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
            (0, (255, 255, 255)),
            (10, (192, 192, 192)),
            (30, (0, 8, 255)),
            (50, (255, 255, 0)),
            (70, (255, 0, 0)),
        ]
        gray = rgb_to_grayscale_gradient(
            cv.cvtColor(self.image, cv.COLOR_BGR2RGB), gradient_points
        )

        mask = create_circular_mask(gray.shape, radius_factor=self.mask_radius)
        isolated_image = gray.copy()
        isolated_image[mask == 0] = 255
        _, binary = cv.threshold(isolated_image, 220, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # **เพิ่มการตรวจสอบเมื่อไม่พบ Contours**
        if not contours:  # ตรวจสอบว่ามี Contours หรือไม่
            QMessageBox.information(
                self, "No Contours Found", "No defects were detected in the image."
            )
            return

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
                    degree_conversion_factor = 0.7  # ตัวแปลงหน่วยจากแกน Y เป็นองศา
                    real_y_diameter = (
                        ((vertical_diameter / h) * (real_y_max - real_y_min))
                        + real_y_min
                    ) * degree_conversion_factor

                    # คำนวณ Length of Defect
                    try:
                        radius = float(self.radius_input.text())  # อ่านค่ารัศมี
                    except ValueError:
                        QMessageBox.warning(self, "Error", "Invalid radius value.")
                        return
                    length_of_defect = (2 * np.pi * radius * real_y_diameter) / 360

                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QTableWidgetItem(str(idx + 1)))
                    self.table.setItem(
                        row_position, 1, QTableWidgetItem(str(vertical_diameter))
                    )
                    self.table.setItem(
                        row_position, 2, QTableWidgetItem(f"{real_y_diameter:.2f}")
                    )
                    self.table.setItem(
                        row_position, 3, QTableWidgetItem(f"{length_of_defect:.2f}")
                    )

                    moments = cv.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                    else:
                        cx, cy = (
                            contour[0][0][0],
                            contour[0][0][1],
                        )  # Default to the first point

                    cv.putText(
                        self.processed_image,
                        str(idx + 1),
                        (cx - 20, cy - 5),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv.LINE_AA,
                    )

        cv.drawContours(self.processed_image, drawn_contour, -1, (0, 255, 0), 2)
        # self.table.resizeColumnsToContents()
        self.show_image(self.processed_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GradientMaskApp()
    window.show()
    sys.exit(app.exec_())
