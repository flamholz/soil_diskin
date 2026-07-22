# YASSO Subroutine Test Guide

This guide provides instructions for compiling and testing the YASSO soil carbon model subroutine from the JSBACH framework.

## Files Overview

- `test_yasso_call.f90` - Main test program demonstrating YASSO usage
- `compile_yasso_test.sh` - Compilation script
- `run_yasso_test.sh` - Complete test script (compile + run)
- `mo_kind_stub.f90` - Stub for working precision module
- `mo_jsb_control_stub.f90` - Stub for JSBACH control and physical constants

## Prerequisites

### Required Software
- GNU Fortran compiler (gfortran) or Intel Fortran compiler (ifort)
- Unix-like environment (Linux, macOS, WSL on Windows)

### Installation on macOS
```bash
# Install gfortran via Homebrew
brew install gcc

# Or install via MacPorts
sudo port install gcc11 +gfortran
```

### Installation on Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install gfortran
```

## Quick Start

### Option 1: Complete Test Script (Recommended)
```bash
# Make script executable
chmod +x run_yasso_test.sh

# Run complete test (compile + execute + optional cleanup)
./run_yasso_test.sh
```

### Option 2: Manual Compilation and Execution
```bash
# Make compilation script executable
chmod +x compile_yasso_test.sh

# Compile the program
./compile_yasso_test.sh

# Run the test program
./test_yasso_call

# Clean up build files (optional)
rm -rf build_temp test_yasso_call mo_kind_stub.f90 mo_jsb_control_stub.f90
```

### Option 3: Manual Compilation (Advanced Users)
```bash
# Create build directory
mkdir -p build_temp

# Compile stub modules
gfortran -O2 -g -cpp -c mo_kind_stub.f90 -o build_temp/mo_kind.o
gfortran -O2 -g -cpp -c mo_jsb_control_stub.f90 -o build_temp/mo_jsb_control_stub.o

# Compile JSBACH modules
gfortran -O2 -g -cpp -Ibuild_temp -c src/shared/mo_jsb_physical_constants.f90 -o build_temp/mo_jsb_physical_constants.o
gfortran -O2 -g -cpp -Ibuild_temp -c src/carbon/mo_carbon_constants.f90 -o build_temp/mo_carbon_constants.o
gfortran -O2 -g -cpp -Ibuild_temp -c src/carbon/mo_carbon_process.f90 -o build_temp/mo_carbon_process.o

# Compile test program
gfortran -O2 -g -cpp -Ibuild_temp -c test_yasso_call.f90 -o build_temp/test_yasso_call.o

# Link everything
gfortran -O2 -g -o test_yasso_call \
    build_temp/mo_kind.o \
    build_temp/mo_jsb_control_stub.o \
    build_temp/mo_jsb_physical_constants.o \
    build_temp/mo_carbon_constants.o \
    build_temp/mo_carbon_process.o \
    build_temp/test_yasso_call.o

# Run the program
./test_yasso_call
```

## Understanding the Output

The test program demonstrates three scenarios:

### Example 1: Non-woody Litter Decomposition
- Shows typical leaf and fine root litter decomposition
- Higher water-soluble and acid-soluble fractions
- Faster decomposition rates

### Example 2: Woody Litter Decomposition  
- Demonstrates woody debris decomposition
- Higher non-soluble fractions (lignin, cellulose)
- Size-dependent decomposition effects
- Slower overall decomposition

### Example 3: Nitrogen-Limited Decomposition
- Shows how nitrogen availability affects decomposition
- Reduced decomposition rates under N limitation
- Only essential outputs when redFact_Nlimit is provided

### Key Output Variables
- **Carbon Pools [mol(C)/m²]**: Updated carbon stocks in 9 chemical pools
- **Soil Respiration [mol(C)/m²/day]**: CO₂ emissions from soil
- **Carbon Fluxes**: Movement between pools and to atmosphere

## Customizing the Test

You can modify `test_yasso_call.f90` to test different scenarios:

### Change Environmental Conditions
```fortran
Weather(1) = 20.0_wp + tmelt   ! Temperature in Kelvin
Weather(2) = 0.000002_wp       ! Precipitation in mm/s
```

### Modify Litter Quality
```fortran
! For more lignified material (slower decomposition)
Lit_coefV(1) = 0.30_wp   ! Acid soluble
Lit_coefV(2) = 0.10_wp   ! Water soluble  
Lit_coefV(3) = 0.20_wp   ! Ethanol soluble
Lit_coefV(4) = 0.38_wp   ! Non-soluble (lignin)
Lit_coefV(5) = 0.02_wp   ! Humus
```

### Test Different Litter Sizes
```fortran
WoodLitterSize = 10.0_wp  ! Large woody debris (10 cm)
WoodLitterSize = 0.5_wp   ! Small branches (0.5 cm)
WoodLitterSize = 0.0_wp   ! Non-woody litter
```

## Troubleshooting

### Compilation Errors
1. **"gfortran: command not found"**
   - Install gfortran using package manager
   - Or set FC environment variable: `export FC=ifort`

2. **"Cannot open module file"**
   - Ensure stub modules are compiled first
   - Check that -I flags point to correct directories

3. **"Preprocessor directive" warnings**
   - Normal for JSBACH code, can be ignored
   - Use `-cpp` flag to enable preprocessor

### Runtime Errors
1. **"Invalid argument" or NaN values**
   - Check that Lit_coefV coefficients sum to 1.0
   - Ensure positive values for carbon pools
   - Verify temperature is in Kelvin (not Celsius)

2. **"Segmentation fault"**
   - Compile with debug flags: `-g -fcheck=all`
   - Check array bounds and initialization

## Using YASSO in Your Own Code

To integrate the YASSO subroutine into your own Fortran program:

1. **Include required modules:**
   ```fortran
   USE mo_kind, ONLY: wp
   USE mo_carbon_process, ONLY: yasso
   USE mo_jsb_physical_constants, ONLY: tmelt
   ```

2. **Prepare input arrays:**
   ```fortran
   REAL(wp), DIMENSION(9)  :: Yasso_io_pools  ! Initial pools
   REAL(wp), DIMENSION(2)  :: Weather         ! T [K], precip [mm/s]
   REAL(wp), DIMENSION(5)  :: Lit_coefV       ! Litter coefficients
   REAL(wp), DIMENSION(18) :: Yasso_out       ! Output array
   ```

3. **Call the subroutine:**
   ```fortran
   CALL yasso(Yasso_io_pools, Weather, litter, Lit_coefV, &
              WoodLitterSize, Yasso_out, fract_aboveground, &
              NPP_2_rootExudates [, redFact_Nlimit])
   ```

## References

- Liski, J., et al. (2005). Carbon accumulation in Finland's forests 1922–2004
- Tuomi, M., et al. (2008). Heterotrophic soil respiration—Comparison of different models
- Tuomi, M., et al. (2009). Leaf litter decomposition—Estimates of global variability
- Tuomi, M., et al. (2011). Soil carbon model Yasso07 graphical user interface

## Support

For questions about:
- **YASSO model**: Contact Finnish Environment Institute (SYKE)
- **JSBACH framework**: Contact MPI for Meteorology or MPI for Biogeochemistry
- **This test code**: Check the documentation in the source files
