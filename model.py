import tensorflow as tf
import tf_slim as slim
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, Dense, BatchNormalization
from keras.models import Model
import inception_mdls


def build_model():
    input_layer = Input(shape=(112, 112, 3))  

    # Conv1 (7x7)
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Max Pool + Norm (3x3)
    pool_norm = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

    # InceptionV3 Layer
    inception_net = inception_mdls.inception_net(pool_norm)

    # Avg Pool (1x1)
    avg_pool = GlobalAveragePooling2D()(inception_net)

    # Fully Conn (1x1x128)
    fully_conn = Dense(128, activation='relu')(avg_pool)

    # L2 Normalization (1x1x128)
    l2_norm = BatchNormalization()(fully_conn)

    # Output
    output_layer = Dense(1, activation='relu')(l2_norm)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = build_model()

# Print model summary
model.summary()
