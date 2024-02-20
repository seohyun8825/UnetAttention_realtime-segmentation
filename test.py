import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from collections import OrderedDict
import cv2  

# 이미지 로드 및 전처리
def preprocess_image(image_path, target_size=(512, 512)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    image = image / 255.0  # 정규화
    return image

# 예측 결과를 색상으로 변환
def visualize_prediction_with_colors(original_image, prediction, label_to_color):
    # 마스크에서 각 클래스에 대한 인덱스 추출하고 색상으로 변환
    prediction = np.argmax(prediction, axis=-1).squeeze()
    output_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_index, (class_name, color) in enumerate(label_to_color.items()):
        output_image[prediction == class_index] = color
    return output_image

# 세그먼테이션된 이미지를 레이블 이미지로 변환
def convert_segmented_to_label(segmented_image, label_to_color):
    label_image = np.zeros(segmented_image.shape[:2], dtype=np.uint8)
    for label, color in label_to_color.items():
        mask = np.all(segmented_image == np.array(color), axis=-1)
        label_image[mask] = list(label_to_color.keys()).index(label)
    return label_image

# 방향을 결정하는 함수
def calculate_direction(label_image, label_to_color, traversable_labels):
    height, width = label_image.shape
    center_line_x = width // 2
    center_overlap = np.isin(label_image[:, center_line_x], traversable_labels).sum()
    left_line_x = center_line_x - int(width * 0.15)
    right_line_x = center_line_x + int(width * 0.15)
    left_overlap = np.isin(label_image[:, left_line_x], traversable_labels).sum()
    right_overlap = np.isin(label_image[:, right_line_x], traversable_labels).sum()
    if center_overlap < height * 0.6:
        if left_overlap > right_overlap:
            direction = "Turn Left"
        else:
            direction = "Turn Right"
    else:
        direction = "Straight"
    return direction, center_overlap, left_overlap, right_overlap

# 이미지에 방향 시각화

def visualize_direction_on_images(original_image, segmented_image, direction, label_to_color, center_overlap, left_overlap, right_overlap):

    height, width, _ = original_image.shape
    center_line_x = width // 2
    left_line_x = center_line_x - int(width * 0.15)
    right_line_x = center_line_x + int(width * 0.15)


    line_points = [(center_line_x, 0, center_line_x, height),
                   (left_line_x, 0, left_line_x, height),
                   (right_line_x, 0, right_line_x, height)]


    color_center = (255, 0, 0)  # Blue
    color_side = (0, 255, 0) if direction == "Turn Left" else (0, 0, 255)  # Green for Left, Red for Right
    color_straight = (255, 255, 255)  # White for Straight


    for image in [original_image, segmented_image]:
        cv2.line(image, line_points[0][:2], line_points[0][2:], color_center if direction == "Straight" else color_straight, 2)
        cv2.line(image, line_points[1][:2], line_points[1][2:], color_side if direction == "Turn Left" else color_straight, 2)
        cv2.line(image, line_points[2][:2], line_points[2][2:], color_side if direction == "Turn Right" else color_straight, 2)

    # 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # cv2 이미지를 RGB로 변환하여 출력
    plt.title('Original Image with Direction')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image.astype(np.uint8))
    plt.title('Segmented Image with Direction')
    plt.axis('off')

    plt.show()

    print(f"Direction: {direction}")
    print(f"Center Overlap: {center_overlap}")
    print(f"Left Overlap: {left_overlap}")
    print(f"Right Overlap: {right_overlap}")




LABEL_TO_COLOR = OrderedDict({
    "background": [255, 255, 255],  # 흰색
    "high_vegetation": [0, 80, 40],  # 짙은 녹색
    "traversable_grass": [0, 255, 128],  # 밝은 녹색
    "smooth_trail": [153, 176, 178],  # 회색 경로
    "obstacle": [0, 0, 255],  # 빨간색 장애물
    "sky": [255, 88, 1],  # 파란색 하늘
    "rough_trial": [30, 76, 156],  # 갈색 거친 길
    "puddle": [128, 0, 255],
    "non_traversable_low_vegetation": [0, 160, 0]
})


# 모델 로드 및 예측
model_path = '/content/attention_unet_best.h5'
model = load_model(model_path)
image_path = '/content/drive/MyDrive/yamaha_v0/train/iid000127/rgb.jpg'
preprocessed_image = preprocess_image(image_path)
predictions = model.predict(preprocessed_image)

# 원본 이미지 로드
original_image = load_img(image_path)
original_image = img_to_array(original_image) / 255.0

# 예측 결과로부터 세그먼테이션된 이미지 생성
segmented_image = visualize_prediction_with_colors(original_image, predictions, LABEL_TO_COLOR)

# 세그먼테이션된 이미지를 레이블 이미지로 변환
label_image = convert_segmented_to_label(segmented_image, LABEL_TO_COLOR)

# 통행 가능한 레이블 인덱스 결정
traversable_labels = [list(LABEL_TO_COLOR.keys()).index(key) for key in ['smooth_trail', 'traversable_grass']]

# 방향 결정
direction, center_overlap, left_overlap, right_overlap = calculate_direction(label_image, LABEL_TO_COLOR, traversable_labels)


# 이미지에 방향 시각화
visualize_direction_on_images(original_image, segmented_image, direction, LABEL_TO_COLOR, center_overlap, left_overlap, right_overlap)
