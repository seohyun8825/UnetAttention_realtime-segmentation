from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, multiply, add, GlobalAveragePooling2D, Dense, Reshape, Conv2DTranspose, BatchNormalization, SeparableConv2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation, Multiply, Add, GlobalAveragePooling2D, Dense, Reshape, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Multiply, Add, Conv2D, Lambda, Activation
from tensorflow.keras.applications import EfficientNetB0
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature



from tensorflow.keras.layers import SeparableConv2D

def ResidualConvBlock(input_tensor, num_filters, kernel_size=3):
    x = SeparableConv2D(num_filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(num_filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Residual Connection
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def SEBlock(input_tensor, ratio=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([input_tensor, se])
    return x

def DenseBlock(input_tensor, num_layers, growth_rate, dropout_rate=0.5):
    for i in range(num_layers):
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, (3, 3), padding='same')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        input_tensor = concatenate([input_tensor, x])
    return input_tensor

def AtrousSpatialPyramidPooling(input_tensor):

    filters = input_tensor.shape[-1]
    rate1 = Conv2D(filters, (1, 1), padding='same', activation='relu', dilation_rate=1)(input_tensor)
    rate2 = Conv2D(filters, (3, 3), padding='same', activation='relu', dilation_rate=6)(input_tensor)
    rate3 = Conv2D(filters, (3, 3), padding='same', activation='relu', dilation_rate=12)(input_tensor)
    rate4 = Conv2D(filters, (3, 3), padding='same', activation='relu', dilation_rate=18)(input_tensor)

    x = add([rate1, rate2, rate3, rate4])
    return x

from tensorflow.keras.layers import UpSampling2D, Concatenate

def decoder_block(input_tensor, skip_features, num_filters):
    input_attention = cbam_block(input_tensor)

    x = UpSampling2D((2, 2), interpolation='bilinear')(input_attention)

    # 스킵 커넥션에 attention 적용
    skip_attention = cbam_block(skip_features)
    x = Concatenate()([x, skip_attention])

    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 최종 출력에 다시 attention 적용
    x = cbam_block(x)

    return x


def build_enhanced_attention_unet(input_shape=(512, 512, 3), num_classes=8):
    inputs = Input(input_shape)

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = Model(inputs=base_model.input, outputs=layers)

    skips = down_stack(inputs)
    x = skips[-1]

    num_filters = [256, 128, 64, 32, 16]
    for i, num_filter in enumerate(num_filters):
        if i < len(skips) - 1:
            x = decoder_block(x, skips[-(i + 2)], num_filter)

    x = DenseBlock(x, num_layers=4, growth_rate=32)
    x = AtrousSpatialPyramidPooling(x)

    x = UpSampling2D(size=(2, 2))(x)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model




model = build_enhanced_attention_unet(input_shape=(512, 512, 3), num_classes=9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()