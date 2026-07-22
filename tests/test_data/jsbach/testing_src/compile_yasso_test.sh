#!/bin/bash

# Compilation script for test_yasso_call.f90
# This script compiles the test program with all necessary dependencies

echo "=========================================="
echo "Compiling YASSO test program"
echo "=========================================="

echo "Current directory: $(pwd)"
# Set compiler (you can change this to gfortran, ifort, etc.)
FC=${FC:-gfortran}

# Compiler flags
FFLAGS="-O2 -g -fbacktrace -fcheck=all -Wall -cpp"

# Create a temporary directory for compilation
TEMP_DIR="build_temp"
mkdir -p $TEMP_DIR

echo "Using Fortran compiler: $FC"
echo "Compiler flags: $FFLAGS"
echo ""

# Copy source files to temp directory and compile in order

echo "Step 1: Compiling stub modules..."

# Compile stub modules first
$FC $FFLAGS -c testing_src/mo_kind_stub.f90 -o $TEMP_DIR/mo_kind.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile mo_kind_stub.f90"
    exit 1
fi

echo "Compiled mo_kind_stub.f90 successfully."

$FC $FFLAGS -c testing_src/mo_jsb_control_stub.f90 -o $TEMP_DIR/mo_jsb_control_stub.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile mo_jsb_control_stub.f90"
    exit 1
fi

echo "Step 2: Compiling JSBACH modules..."

# Compile physical constants
$FC $FFLAGS -I$TEMP_DIR -c src/shared/mo_jsb_physical_constants.f90 -o $TEMP_DIR/mo_jsb_physical_constants.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile mo_jsb_physical_constants.f90"
    exit 1
fi

# Compile carbon constants
$FC $FFLAGS -I$TEMP_DIR -c src/carbon/mo_carbon_constants.f90 -o $TEMP_DIR/mo_carbon_constants.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile mo_carbon_constants.f90"
    exit 1
fi

# Compile carbon process module
$FC $FFLAGS -I$TEMP_DIR -c src/carbon/mo_carbon_process.f90 -o $TEMP_DIR/mo_carbon_process.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile mo_carbon_process.f90"
    exit 1
fi

echo "Step 3: Compiling test program..."

# Compile the test program
$FC $FFLAGS -I$TEMP_DIR -c testing_src/test_yasso_call.f90 -o $TEMP_DIR/test_yasso_call.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile test_yasso_call.f90"
    exit 1
fi

echo "Step 4: Linking..."

# Link everything together
$FC $FFLAGS -o testing_src/test_yasso_call \
    $TEMP_DIR/mo_kind.o \
    $TEMP_DIR/mo_jsb_control_stub.o \
    $TEMP_DIR/mo_jsb_physical_constants.o \
    $TEMP_DIR/mo_carbon_constants.o \
    $TEMP_DIR/mo_carbon_process.o \
    $TEMP_DIR/test_yasso_call.o

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to link the program"
    exit 1
fi

echo ""
echo "=========================================="
echo "Compilation successful!"
echo "=========================================="
echo "Executable created: test_yasso_call"
echo ""
echo "To run the test:"
echo "  ./test_yasso_call"
echo ""
echo "To clean up build files:"
echo "  rm -rf $TEMP_DIR"
echo "  rm test_yasso_call"
echo "=========================================="
