from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path, mask_path):
    def load_image(image_path, mask_path):
        image = load_img(image_path.numpy().decode("utf-8"), target_size=(IMG_HEIGHT, IMG_WIDTH))
        image = img_to_array(image) / 255.0

        # 마스크 로드 및 one-hot 인코딩
        mask = load_img(mask_path.numpy().decode("utf-8"), target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
        mask = img_to_array(mask).squeeze()  # 채널 차원 제거
        mask = np.round(mask * (NUM_CLASSES - 1) / 255.0)  # 마스크 스케일 조정
        mask = tf.one_hot(tf.cast(mask, tf.int32), NUM_CLASSES)  # One-hot 인코딩
        return image, mask

    [image, mask] = tf.py_function(load_image, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])
    return image, mask

def create_dataset(image_paths, mask_paths, batch_size):
    image_paths = tf.constant(image_paths)
    mask_paths = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def load_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array


# Create datasets
BATCH_SIZE = 16
train_dataset = create_dataset(train_images, train_masks, BATCH_SIZE)
val_dataset = create_dataset(val_images, val_masks, BATCH_SIZE)
