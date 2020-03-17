from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from resnet18 import ResNet18

import sys

if len(sys.argv) != 3:
    print("Usage: python3 train.py <dataset> <model.h5")
    sys.exit(0)


DATASET_PATH  = sys.argv[1]
NUM_CLASSES = 3

IMAGE_SIZE = (224, 224)


BATCH_SIZE = 32

NUM_EPOCHS = 20

WEIGHTS_FINAL = sys.argv[2]

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/Train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/Test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

net_final = ResNet18((IMAGE_SIZE[0],IMAGE_SIZE[1], 3), NUM_CLASSES)


net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())

net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

net_final.save(WEIGHTS_FINAL)
