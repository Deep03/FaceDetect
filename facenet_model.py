from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, AveragePooling2D, GlobalAveragePooling2D, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from inception_mdls import inception_module
from triplet_loss import triplet_loss

def model(input_emd):
   # Conv1
   conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input_emd)

   # Max Pool + Norm
   max_pool_norm = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='max_pool_norm')(conv1)

   # Inception (2)
   inception_2 = inception_module(max_pool_norm, 64, 64, 192, 1, 1, 1, name='inception_2')


   # Norm + Max Pool
   norm_max_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='norm_max_pool')(inception_2)

   # Inception Layer derived from https://arxiv.org/pdf/1503.03832.pdf & https://arxiv.org/pdf/1409.4842.pdf
   inception_3a = inception_module(norm_max_pool, 64, 96, 128, 16, 32, 32, name='inception_3a')
   inception_3b =  inception_module(inception_3a, 64, 96, 128, 32, 64, 64, name='inception_3b')
   inception_3c =  inception_module(inception_3b, 128, 128, 256, 32, 64, 64, name='inception_3c')

   inception_4a =  inception_module(inception_3c, 256, 96, 192, 32, 64, 128, name='inception_4a')
   inception_4b =  inception_module(inception_4a, 224, 112, 224, 32, 64, 128, name='inception_4b')
   inception_4c =  inception_module(inception_4b, 192, 128, 256, 32, 64, 128, name='inception_4c')
   inception_4d =  inception_module(inception_4c, 160, 144, 288, 32, 64, 128, name='inception_4d')
   inception_4e =  inception_module(inception_4d, 160, 160, 256, 64, 128, 128, name='inception_4e')


   inception_5a =  inception_module(inception_4e, 384, 192, 384, 48, 128, 128, name='inception_5a')
   inception_5b =  inception_module(inception_5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

   avgpool = AveragePooling2D(pool_size = 3)(inception_5b)

   fully_conn = Dense(128, activation='relu', name='fully_conn')(avgpool)

   fully_conn = Dense(2000)(fully_conn)

   global_avg_pool = GlobalAveragePooling2D()(fully_conn)

   l2_norm = BatchNormalization()(global_avg_pool)

   # Update Needed: use layers.concatenate
   model = Model(inputs=input_emd, outputs=l2_norm, name="FaceNetModel")

   # lr = 10^-3 described in the paper
   # loss(TBC)=triplet_loss
   model.compile(Adam(learning_rate=0.001), loss=triplet_loss, metrics=['accuracy'])
   
   return model


input_layer = Input(shape=(112, 112, 3), name='input_layer')
facenet_model = model(input_layer)

ANCHOR = Input(shape=(112, 112, 3), name='anchor_emd')
POSITIVE =  Input(shape=(112, 112, 3), name='positive_emd')
NEGATIVE =  Input(shape=(112, 112, 3), name='negative_emd')

anchor_emd = facenet_model(ANCHOR)
positive_emd = facenet_model(POSITIVE)
negative_emd = facenet_model(NEGATIVE)
