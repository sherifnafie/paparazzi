TFLITE_SRCS = $(wildcard kernel/*.cc) $(wildcard kernels/*.cc) $(wildcard kernels/internal/*.cc) \
              $(wildcard c/*.c) $(wildcard core/*.cc) $(wildcard core/api/*.cc)

TFLITE_OBJS = $(TFLITE_SRCS:.cc=.o)
TFLITE_OBJS := $(TFLITE_OBJS:.c=.o)

CXXFLAGS += -std=c++11 -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -I. -I$(PAPARAZZI_SRC)/sw/ext/tflite

all: libtensorflowlite.a

%.o: %.cc
    $(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

libtensorflowlite.a: $(TFLITE_OBJS)
    $(AR) rcs $@ $^

clean:
    rm -f $(TFLITE_OBJS) libtensorflowlite.a
