from tensorflow.contrib import lite
import sys

if len(sys.argv) != 3:
    print("Usage: python3 convert.py <model.h5> <tflite-model.tflite>")
    sys.exit(0)

m = sys.argv[1]
tf_m = sys.argv[2]
converter = lite.TFLiteConverter.from_keras_model_file(m)
tfmodel = converter.convert()
open (tf_m , "wb") .write(tfmodel)
