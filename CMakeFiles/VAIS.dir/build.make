# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.4.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.4.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alfredoliardo/Xcode/VAIS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alfredoliardo/Xcode/VAIS

# Include any dependencies generated for this target.
include CMakeFiles/VAIS.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VAIS.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VAIS.dir/flags.make

CMakeFiles/VAIS.dir/main.cpp.o: CMakeFiles/VAIS.dir/flags.make
CMakeFiles/VAIS.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alfredoliardo/Xcode/VAIS/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VAIS.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VAIS.dir/main.cpp.o -c /Users/alfredoliardo/Xcode/VAIS/main.cpp

CMakeFiles/VAIS.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VAIS.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alfredoliardo/Xcode/VAIS/main.cpp > CMakeFiles/VAIS.dir/main.cpp.i

CMakeFiles/VAIS.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VAIS.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alfredoliardo/Xcode/VAIS/main.cpp -o CMakeFiles/VAIS.dir/main.cpp.s

CMakeFiles/VAIS.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/VAIS.dir/main.cpp.o.requires

CMakeFiles/VAIS.dir/main.cpp.o.provides: CMakeFiles/VAIS.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/VAIS.dir/build.make CMakeFiles/VAIS.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/VAIS.dir/main.cpp.o.provides

CMakeFiles/VAIS.dir/main.cpp.o.provides.build: CMakeFiles/VAIS.dir/main.cpp.o


# Object files for target VAIS
VAIS_OBJECTS = \
"CMakeFiles/VAIS.dir/main.cpp.o"

# External object files for target VAIS
VAIS_EXTERNAL_OBJECTS =

VAIS: CMakeFiles/VAIS.dir/main.cpp.o
VAIS: CMakeFiles/VAIS.dir/build.make
VAIS: /usr/local/lib/libopencv_videostab.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_superres.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_stitching.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_shape.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_photo.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_objdetect.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_calib3d.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_features2d.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_ml.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_highgui.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_videoio.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_imgcodecs.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_flann.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_video.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_imgproc.3.1.0.dylib
VAIS: /usr/local/lib/libopencv_core.3.1.0.dylib
VAIS: CMakeFiles/VAIS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/alfredoliardo/Xcode/VAIS/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VAIS"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VAIS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VAIS.dir/build: VAIS

.PHONY : CMakeFiles/VAIS.dir/build

CMakeFiles/VAIS.dir/requires: CMakeFiles/VAIS.dir/main.cpp.o.requires

.PHONY : CMakeFiles/VAIS.dir/requires

CMakeFiles/VAIS.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VAIS.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VAIS.dir/clean

CMakeFiles/VAIS.dir/depend:
	cd /Users/alfredoliardo/Xcode/VAIS && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alfredoliardo/Xcode/VAIS /Users/alfredoliardo/Xcode/VAIS /Users/alfredoliardo/Xcode/VAIS /Users/alfredoliardo/Xcode/VAIS /Users/alfredoliardo/Xcode/VAIS/CMakeFiles/VAIS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VAIS.dir/depend

