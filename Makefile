# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bobbi/proj/gym-battlesnake-pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bobbi/proj/gym-battlesnake-pytorch

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/bobbi/proj/gym-battlesnake-pytorch/CMakeFiles /home/bobbi/proj/gym-battlesnake-pytorch//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/bobbi/proj/gym-battlesnake-pytorch/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named gymbattlesnake

# Build rule for target.
gymbattlesnake: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 gymbattlesnake
.PHONY : gymbattlesnake

# fast build rule for target.
gymbattlesnake/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/build
.PHONY : gymbattlesnake/fast

gym_battlesnake_pytorch/src/gameinstance.o: gym_battlesnake_pytorch/src/gameinstance.cpp.o
.PHONY : gym_battlesnake_pytorch/src/gameinstance.o

# target to build an object file
gym_battlesnake_pytorch/src/gameinstance.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gameinstance.cpp.o
.PHONY : gym_battlesnake_pytorch/src/gameinstance.cpp.o

gym_battlesnake_pytorch/src/gameinstance.i: gym_battlesnake_pytorch/src/gameinstance.cpp.i
.PHONY : gym_battlesnake_pytorch/src/gameinstance.i

# target to preprocess a source file
gym_battlesnake_pytorch/src/gameinstance.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gameinstance.cpp.i
.PHONY : gym_battlesnake_pytorch/src/gameinstance.cpp.i

gym_battlesnake_pytorch/src/gameinstance.s: gym_battlesnake_pytorch/src/gameinstance.cpp.s
.PHONY : gym_battlesnake_pytorch/src/gameinstance.s

# target to generate assembly for a file
gym_battlesnake_pytorch/src/gameinstance.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gameinstance.cpp.s
.PHONY : gym_battlesnake_pytorch/src/gameinstance.cpp.s

gym_battlesnake_pytorch/src/gamewrapper.o: gym_battlesnake_pytorch/src/gamewrapper.cpp.o
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.o

# target to build an object file
gym_battlesnake_pytorch/src/gamewrapper.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gamewrapper.cpp.o
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.cpp.o

gym_battlesnake_pytorch/src/gamewrapper.i: gym_battlesnake_pytorch/src/gamewrapper.cpp.i
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.i

# target to preprocess a source file
gym_battlesnake_pytorch/src/gamewrapper.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gamewrapper.cpp.i
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.cpp.i

gym_battlesnake_pytorch/src/gamewrapper.s: gym_battlesnake_pytorch/src/gamewrapper.cpp.s
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.s

# target to generate assembly for a file
gym_battlesnake_pytorch/src/gamewrapper.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/gamewrapper.cpp.s
.PHONY : gym_battlesnake_pytorch/src/gamewrapper.cpp.s

gym_battlesnake_pytorch/src/threadpool.o: gym_battlesnake_pytorch/src/threadpool.cpp.o
.PHONY : gym_battlesnake_pytorch/src/threadpool.o

# target to build an object file
gym_battlesnake_pytorch/src/threadpool.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/threadpool.cpp.o
.PHONY : gym_battlesnake_pytorch/src/threadpool.cpp.o

gym_battlesnake_pytorch/src/threadpool.i: gym_battlesnake_pytorch/src/threadpool.cpp.i
.PHONY : gym_battlesnake_pytorch/src/threadpool.i

# target to preprocess a source file
gym_battlesnake_pytorch/src/threadpool.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/threadpool.cpp.i
.PHONY : gym_battlesnake_pytorch/src/threadpool.cpp.i

gym_battlesnake_pytorch/src/threadpool.s: gym_battlesnake_pytorch/src/threadpool.cpp.s
.PHONY : gym_battlesnake_pytorch/src/threadpool.s

# target to generate assembly for a file
gym_battlesnake_pytorch/src/threadpool.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/gymbattlesnake.dir/build.make CMakeFiles/gymbattlesnake.dir/gym_battlesnake_pytorch/src/threadpool.cpp.s
.PHONY : gym_battlesnake_pytorch/src/threadpool.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... gymbattlesnake"
	@echo "... gym_battlesnake_pytorch/src/gameinstance.o"
	@echo "... gym_battlesnake_pytorch/src/gameinstance.i"
	@echo "... gym_battlesnake_pytorch/src/gameinstance.s"
	@echo "... gym_battlesnake_pytorch/src/gamewrapper.o"
	@echo "... gym_battlesnake_pytorch/src/gamewrapper.i"
	@echo "... gym_battlesnake_pytorch/src/gamewrapper.s"
	@echo "... gym_battlesnake_pytorch/src/threadpool.o"
	@echo "... gym_battlesnake_pytorch/src/threadpool.i"
	@echo "... gym_battlesnake_pytorch/src/threadpool.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

