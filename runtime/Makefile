TENSORFLOW = /home/user/
INCLUDE = -I ${TENSORFLOW}/tensorflow/ -I ${TENSORFLOW}/tensorflow/tensorflow/lite/tools/make/downloads/absl/ -I ${TENSORFLOW}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/

LIB = libtensorflow-lite-xnnpack.a -lpthread -ldl -latomic \
    /home/user/tensorflow/tensorflow/lite/tools/make/downloads/XNNPACK/build/libXNNPACK.a \
    /home/user/tensorflow/tensorflow/lite/tools/make/downloads/XNNPACK/build/pthreadpool/libpthreadpool.a \
    /home/user/tensorflow/tensorflow/lite/tools/make/downloads/XNNPACK/build/cpuinfo/libcpuinfo.a \
    /home/user/tensorflow/tensorflow/lite/tools/make/downloads/XNNPACK/build/cpuinfo/libcpuinfo_internals.a \
    /home/user/tensorflow/tensorflow/lite/tools/make/downloads/XNNPACK/build/clog/libclog.a \
    `pkg-config opencv --cflags --libs`

all:
	$(CXX) $(INCLUDE) main.cc face_detector.cc $(LIB) -o main
