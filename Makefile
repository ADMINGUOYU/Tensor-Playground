CC = g++
CFLAGS = -std=c++11 -g -Wall
SRCDIR = ./Tensor
# Collect all .cpp files recursively (including main.cpp)
SRCS := main.cpp $(shell find $(SRCDIR) -type f -name '*.cpp' | sort)
# Generate object files and dependency files from source files
OBJS = $(SRCS:.cpp=.o)
# Generate dependency files from source files
DEPS = $(SRCS:.cpp=.d)

ifneq ($(OS), Windows_NT)
RM = -rm
ifeq ($(shell uname), Linux)
# Enable AddressSanitizer and UndefinedBehaviorSanitizer on Linux
CFLAGS += -fsanitize=address,undefined,leak
endif
else
RM = -del
endif

tensor: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

-include $(DEPS)

%.o: %.cpp
	$(CC) -MMD -MP -c -o $@ $< $(CFLAGS)

clean:
	$(RM) $(OBJS) $(DEPS) tensor