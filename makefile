COMMON_FLAGS=-g -O3 -Wall -Wpedantic -I./include -Llib
CPPFLAGS=-std=c++17 $(COMMON_FLAGS)
CPP_SRCS=$(wildcard src/*.cpp)
ALL_OBJS=$(patsubst src/%.cpp, build/%.o, $(CPP_SRCS))
OPENGL=-lglfw -lGL -lGLEW -ldl -lX11
LIBS=-lpix_core -lnn

# compiles all objects, and creates executable files from ./src/main
.PHONY: _exec
_exec: $(patsubst src/main/%.cpp, bin/%, $(wildcard src/main/*.cpp))

# objects should not be removed automatically
.PRECIOUS: build/%.o

# links all C++ source files from ./src/main
bin/%: src/main/%.cpp
	# ----- C++ executable ----- #
	g++ $(CPPFLAGS) -o"$@" $^

# compiles a C++ source file from ./src
build/%.o: src/%.cpp
	# ----- C++ object ----- #
	g++ $(CPPFLAGS) $< -c -o"$@"

build/nn/%.o: src/nn/%.cpp
	# ----- Neural Network object ----- #
	g++ $(CPPFLAGS) -c -o"$@" $^

build/pix/%.o: src/pix/%.cpp
	# ----- Pix object ----- #
	g++ $(CPPFLAGS) -c -o"$@" $^

# Core functions for Pix
lib/libpix_core.a: $(patsubst src/pix/%.cpp,build/pix/%.o,$(wildcard src/pix/*.cpp))
	# ----- Pix Core static library ----- #
	rm -f $@
	ar -rs $@ $^

# Core functions for Pix
lib/libnn.a: $(patsubst src/nn/%.cpp,build/nn/%.o,$(wildcard src/nn/*.cpp))
	# ----- Neural Network static library ----- #
	rm -f $@
	ar -rs $@ $^

# Custom target for OpenGL-powered binaries
bin/nncli: lib/libpix_core.a lib/libnn.a src/main/nncli.cpp
	g++ $(CPPFLAGS) -o"$@" $^ -lpix_core -lnn $(OPENGL) -lpthread

bin/nntest: src/main/nntest.cpp
	g++ $(CPPFLAGS) -o"$@" $<

.PHONY: setup clean reset
setup: reset
	mkdir -p src/main build/nn build/pix

clean:
	rm -rf build
	mkdir build build/nn build/pix

reset:
	rm -rf bin lib build
	mkdir bin lib build build/nn build/pix
