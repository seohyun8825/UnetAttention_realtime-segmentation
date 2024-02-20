import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


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


def visualize_prediction_with_colors(original_image, prediction):
    # 마스크에서 각 클래스에 대한 인덱스 추출
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction.squeeze()

    # 출력 이미지 (RGB)
    output_image = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    for class_index, (class_name, color) in enumerate(LABEL_TO_COLOR.items()):
        output_image[prediction == class_index] = color

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[1].imshow(output_image)
    ax[1].set_title('Segmentation Mask with Colors')
    plt.show()

# 이미지 로드 및 전처리 함수
def preprocess_image(image_path, target_size=(512, 512)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # 모델 예측을 위해 배치 차원 추가
    image = image / 255.0  
    return image

def visualize_prediction(original_image, prediction, num_classes=8):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')


    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction.squeeze()

    ax[1].imshow(prediction, cmap='jet', interpolation='nearest')
    ax[1].set_title('Segmentation Mask')
    plt.show()



model_path = '/content/attention_unet_best.h5'
model = load_model(model_path)

# 이미지 로드 및 전처리
image_path = '/content/drive/MyDrive/yamaha_v0/valid/iid000842/rgb.jpg'
preprocessed_image = preprocess_image(image_path)

# 모델 예측
predictions = model.predict(preprocessed_image)

# 예측 결과 시각화
original_image = load_img(image_path)

visualize_prediction_with_colors(original_image, predictions)

