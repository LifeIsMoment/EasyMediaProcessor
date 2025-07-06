import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QListWidget, QProgressBar, QMessageBox,
                              QGroupBox, QComboBox, QSpinBox, QCheckBox,
                              QTextEdit, QFileDialog, QSlider, QGridLayout)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# TensorFlow/Keras import (선택적)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class ObjectDetector(QThread):
    """객체 감지 스레드"""
    progress = Signal(int)
    result = Signal(str, np.ndarray)  # message, processed_image
    error = Signal(str)
    
    def __init__(self, image_path, detection_method='opencv', confidence_threshold=0.5):
        super().__init__()
        self.image_path = image_path
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.model_path = None
    
    def set_model_path(self, model_path):
        """모델 경로 설정"""
        self.model_path = model_path
    
    def run(self):
        try:
            if self.detection_method == 'opencv':
                self.detect_with_opencv()
            elif self.detection_method == 'tensorflow':
                self.detect_with_tensorflow()
            elif self.detection_method == 'yolo':
                self.detect_with_yolo()
        except Exception as e:
            self.error.emit(f"객체 감지 실패: {str(e)}")
    
    def detect_with_opencv(self):
        """OpenCV의 내장 Haar Cascade를 사용한 얼굴 감지"""
        self.progress.emit(10)
        
        # 이미지 로드
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        self.progress.emit(30)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade 분류기 로드 (얼굴 감지)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.progress.emit(50)
        
        # 얼굴 감지
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        self.progress.emit(80)
        
        # 감지된 얼굴에 사각형 그리기
        result_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        self.progress.emit(100)
        message = f"OpenCV 얼굴 감지 완료: {len(faces)}개 얼굴 발견"
        self.result.emit(message, result_image)
    
    def detect_with_tensorflow(self):
        """TensorFlow/Keras 모델을 사용한 객체 감지"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 설치되지 않았습니다.")
        
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError("TensorFlow 모델 파일을 찾을 수 없습니다.")
        
        self.progress.emit(10)
        
        # 모델 로드
        model = keras.models.load_model(self.model_path)
        self.progress.emit(30)
        
        # 이미지 전처리
        image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 모델 입력 크기에 맞게 리사이즈 (예: 224x224)
        input_size = 224
        resized = cv2.resize(image_rgb, (input_size, input_size))
        normalized = resized.astype(np.float32) / 255.0
        batch_input = np.expand_dims(normalized, axis=0)
        
        self.progress.emit(70)
        
        # 예측 수행
        predictions = model.predict(batch_input)
        self.progress.emit(90)
        
        # 결과 처리 (이것은 예시이며, 실제 모델에 따라 달라짐)
        result_image = image.copy()
        
        # 간단한 분류 결과 표시
        if predictions.shape[-1] > 1:  # 다중 클래스 분류
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # 클래스 이름 (예시)
            class_names = ['cat', 'dog', 'person', 'car', 'unknown']
            if predicted_class < len(class_names):
                class_name = class_names[predicted_class]
            else:
                class_name = f"Class_{predicted_class}"
            
            # 텍스트 표시
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.progress.emit(100)
        message = f"TensorFlow 모델 예측 완료"
        self.result.emit(message, result_image)
    
    def detect_with_yolo(self):
        """YOLO를 사용한 객체 감지 (OpenCV DNN 모듈 사용)"""
        self.progress.emit(10)
        
        # YOLO 설정 파일들이 있다고 가정
        weights_path = "yolo/yolov3.weights"
        config_path = "yolo/yolov3.cfg"
        names_path = "yolo/coco.names"
        
        # 파일 존재 확인
        if not all(Path(p).exists() for p in [weights_path, config_path, names_path]):
            # YOLO 파일이 없으면 OpenCV의 사전 훈련된 모델 사용
            self.detect_with_opencv_dnn()
            return
        
        # YOLO 네트워크 로드
        net = cv2.dnn.readNet(weights_path, config_path)
        
        # 클래스 이름 로드
        with open(names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        self.progress.emit(30)
        
        # 이미지 로드
        image = cv2.imread(self.image_path)
        height, width, channels = image.shape
        
        # 블롭 생성
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        
        self.progress.emit(60)
        
        # 추론 실행
        outs = net.forward(self.get_output_layers(net))
        
        self.progress.emit(80)
        
        # 결과 처리
        result_image = self.process_yolo_outputs(image, outs, classes)
        
        self.progress.emit(100)
        message = "YOLO 객체 감지 완료"
        self.result.emit(message, result_image)
    
    def detect_with_opencv_dnn(self):
        """OpenCV DNN 모듈을 사용한 간단한 객체 감지"""
        self.progress.emit(20)
        
        # MobileNet SSD 모델 사용 (가벼운 모델)
        try:
            # 이미지 로드
            image = cv2.imread(self.image_path)
            height, width = image.shape[:2]
            
            self.progress.emit(50)
            
            # 간단한 blob 감지 (원형 객체 감지)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # HoughCircles를 사용한 원형 객체 감지
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(height/8),
                param1=200,
                param2=100,
                minRadius=int(height/25),
                maxRadius=int(height/5)
            )
            
            result_image = image.copy()
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(result_image, (x, y), r, (0, 255, 0), 4)
                    cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(result_image, 'Circle', (x-50, y-r-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            self.progress.emit(100)
            message = f"OpenCV DNN 원형 객체 감지 완료: {len(circles) if circles is not None else 0}개 발견"
            self.result.emit(message, result_image)
            
        except Exception as e:
            # 폴백: 단순한 엣지 감지
            self.simple_edge_detection()
    
    def simple_edge_detection(self):
        """간단한 엣지 감지"""
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 엣지를 컬러 이미지에 오버레이
        result_image = image.copy()
        result_image[edges != 0] = [0, 255, 0]  # 녹색으로 엣지 표시
        
        self.progress.emit(100)
        message = "엣지 감지 완료"
        self.result.emit(message, result_image)
    
    def get_output_layers(self, net):
        """YOLO 출력 레이어 이름 가져오기"""
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    def process_yolo_outputs(self, image, outs, classes):
        """YOLO 출력 처리"""
        height, width = image.shape[:2]
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        result_image = image.copy()
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"{label} {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image


class AIDetectionDialog(QDialog):
    """AI 객체 감지 다이얼로그"""
    
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.image_path = image_path
        self.detector = None
        self.result_image = None
        
        self.setWindowTitle("AI 객체 감지")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 이미지 선택
        file_group = QGroupBox("이미지 파일")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("파일이 선택되지 않음")
        if self.image_path:
            self.file_label.setText(str(Path(self.image_path).name))
        
        self.select_file_btn = QPushButton("파일 선택")
        self.select_file_btn.clicked.connect(self.select_image_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_btn)
        layout.addWidget(file_group)
        
        # 감지 설정
        settings_group = QGroupBox("감지 설정")
        settings_layout = QGridLayout(settings_group)
        
        # 감지 방법 선택
        settings_layout.addWidget(QLabel("감지 방법:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "OpenCV 얼굴 감지",
            "OpenCV 원형 객체",
            "엣지 감지"
        ])
        if TENSORFLOW_AVAILABLE:
            self.method_combo.addItem("TensorFlow 모델")
        
        settings_layout.addWidget(self.method_combo, 0, 1)
        
        # 신뢰도 임계값
        settings_layout.addWidget(QLabel("신뢰도 임계값:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        settings_layout.addWidget(self.confidence_slider, 1, 1)
        
        self.confidence_label = QLabel("0.50")
        settings_layout.addWidget(self.confidence_label, 1, 2)
        
        # TensorFlow 모델 경로 (선택적)
        if TENSORFLOW_AVAILABLE:
            settings_layout.addWidget(QLabel("TF 모델:"), 2, 0)
            self.model_path_label = QLabel("선택되지 않음")
            settings_layout.addWidget(self.model_path_label, 2, 1)
            
            self.select_model_btn = QPushButton("모델 선택")
            self.select_model_btn.clicked.connect(self.select_model_file)
            settings_layout.addWidget(self.select_model_btn, 2, 2)
        
        layout.addWidget(settings_group)
        
        # 실행 버튼
        self.detect_btn = QPushButton("객체 감지 시작")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.detect_btn)
        
        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 결과 표시
        result_group = QGroupBox("감지 결과")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(100)
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        # 결과 이미지 저장 버튼
        self.save_result_btn = QPushButton("결과 이미지 저장")
        self.save_result_btn.clicked.connect(self.save_result_image)
        self.save_result_btn.setEnabled(False)
        result_layout.addWidget(self.save_result_btn)
        
        layout.addWidget(result_group)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def select_image_file(self):
        """이미지 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", "",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp);;모든 파일 (*)"
        )
        
        if file_path:
            self.image_path = file_path
            self.file_label.setText(Path(file_path).name)
    
    def select_model_file(self):
        """TensorFlow 모델 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "TensorFlow 모델 선택", "",
            "모델 파일 (*.h5 *.keras *.pb);;모든 파일 (*)"
        )
        
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(Path(file_path).name)
    
    def update_confidence_label(self, value):
        """신뢰도 라벨 업데이트"""
        self.confidence_label.setText(f"{value/100:.2f}")
    
    def start_detection(self):
        """객체 감지 시작"""
        if not self.image_path or not Path(self.image_path).exists():
            QMessageBox.warning(self, "경고", "유효한 이미지 파일을 선택하세요.")
            return
        
        # 감지 방법 결정
        method_text = self.method_combo.currentText()
        if "얼굴" in method_text:
            method = "opencv"
        elif "원형" in method_text:
            method = "yolo"  # 원형 객체는 opencv_dnn으로 처리
        elif "엣지" in method_text:
            method = "yolo"  # 엣지 감지도 opencv_dnn으로 처리
        elif "TensorFlow" in method_text:
            method = "tensorflow"
        else:
            method = "opencv"
        
        confidence = self.confidence_slider.value() / 100.0
        
        # 감지 시작
        self.detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        
        self.detector = ObjectDetector(self.image_path, method, confidence)
        
        # TensorFlow 모델 경로 설정
        if method == "tensorflow" and hasattr(self, 'model_path'):
            self.detector.set_model_path(self.model_path)
        
        self.detector.progress.connect(self.progress_bar.setValue)
        self.detector.result.connect(self.on_detection_finished)
        self.detector.error.connect(self.on_detection_error)
        self.detector.start()
    
    def on_detection_finished(self, message, result_image):
        """감지 완료"""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.result_text.append(message)
        self.result_image = result_image
        self.save_result_btn.setEnabled(True)
        
        # 간단한 통계 표시
        if result_image is not None:
            height, width = result_image.shape[:2]
            self.result_text.append(f"처리된 이미지 크기: {width}x{height}")
    
    def on_detection_error(self, error):
        """감지 오류"""
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.result_text.append(f"❌ 오류: {error}")
        QMessageBox.critical(self, "오류", error)
    
    def save_result_image(self):
        """결과 이미지 저장"""
        if self.result_image is None:
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "결과 이미지 저장", 
            f"detected_{Path(self.image_path).stem}.jpg",
            "이미지 파일 (*.jpg *.png *.bmp);;모든 파일 (*)"
        )
        
        if save_path:
            success = cv2.imwrite(save_path, self.result_image)
            if success:
                self.result_text.append(f"✅ 결과 이미지 저장됨: {save_path}")
                QMessageBox.information(self, "성공", f"결과 이미지가 저장되었습니다:\n{save_path}")
            else:
                self.result_text.append(f"❌ 이미지 저장 실패: {save_path}")
                QMessageBox.warning(self, "실패", "이미지 저장에 실패했습니다.")