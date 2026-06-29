# Define separate compilers
CC = gcc
CXX = g++

# Set distinct standard flags for C and C++
CFLAGS = -std=c99 -g -Wall -march=native
CXXFLAGS = -std=c++11 -g -Wall -march=native
SRCDIR = ./Tensor

# Collect files by their specific extensions
# Collect all .c files recursively
C_SRCS := $(shell find $(SRCDIR) -type f -name '*.c' | sort)
# Collect all .cpp files recursively (including main.cpp)
CPP_SRCS := main.cpp $(shell find $(SRCDIR) -type f -name '*.cpp' | sort)

# Generate object files and dependency files from source files
OBJS = $(C_SRCS:.c=.o) $(CPP_SRCS:.cpp=.o)
# Generate dependency files from source files
DEPS = $(C_SRCS:.c=.d) $(CPP_SRCS:.cpp=.d)

ifneq ($(OS), Windows_NT)
RM = -rm
ifeq ($(shell uname), Linux)
# Enable AddressSanitizer and UndefinedBehaviorSanitizer on Linux
SANITIZERS = -fsanitize=address,undefined,leak
CFLAGS += $(SANITIZERS)
CXXFLAGS += $(SANITIZERS)
endif
else
RM = -del
endif

# Link everything together using the C++ linker (g++)
tensor: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)

-include $(DEPS)

# Strict rule for C files (Uses gcc with -std=c99)
%.o: %.c
	$(CC) -MMD -MP -c -o $@ $< $(CFLAGS)

# Strict rule for C++ files (Uses g++ with -std=c++11)
%.o: %.cpp
	$(CXX) -MMD -MP -c -o $@ $< $(CXXFLAGS)

clean:
	$(RM) $(OBJS) $(DEPS) tensor