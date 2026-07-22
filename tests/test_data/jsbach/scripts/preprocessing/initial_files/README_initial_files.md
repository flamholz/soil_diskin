### README: ICON-Land initial file generation

We use a series of scripts

- to generate ICON-Land initial (ic) and boundary condition (bc) files,
- to generate consistent HD parameter files for internal HD (currently available
  for grid resolutions up to R02B06),
- to adapt the extpar data file read by the NWP atmosphere accordingly and
- to generate the HD receive mask file needed in setups with external HD.

Besides, we provide scripts to

- generate initial soil conditions from model output (`generate_ic_soil_from_output.sh`)
- generate initial conditions for the (internal) HD model from a restart file
  (`generate_hdstart_from_restart.sh`)


The master script `create_icon-land_ini_files.sh` makes use of the following scripts:

1. `generate_fractional_mask.sh`
   - Generate the fractional mask file needed for coupled atmo/ocean configurations

2. `extpar4jsbach_mpim_icon.sh`
   - run extpar to generate soil texture data as well as albedo, roughness
     length, forest fraction and LAI and vegetation fraction climatologies.

3. `jsbach4_ini_files_from_gauss.sh`
   - Remapping of Gaussian grid JSBACH3 initial files to the ICON grid
   - Remapping of additional soil parameters from a 0.5deg to the ICON grid
   - Get glacier, land sea mask and orographic data based on extpar data
   - Assigning the data to the different jsbach4 ic and bc files

4. `jsbach4_ini_files_from_extpar.sh`
   Generate ic/bc files containing the new expar data:
   - Lake mask replaced (-> bc_land_frac)
   - Albedo, roughness length, forest fraction, lai_clim and veg_fract
     replaced. Note: albedo_veg_vis, -veg_nir, -soil_vis and -soil_nir are
     still interpolated from Gaussian grid
   - Rooting depth (and maxmoist) replaced, additional variables:
     FR_SAND, FR_SILT, FR_CLAY, FR_OC, SUB_FR_SAND, SUB_FR_SILT,
     SUB_FR_CLAY and SUB_FR_OC

5. `adapt_extpar_file.sh`
   The extpar data file read by the NWP atmosphere contains several
   variables, that are also included in the bc_land files. For consistency,
   these variables are replaced by the respective bc_land file variables.
   We make sure, variable `cell_sea_land_mask` is **not** included in the
   adapted extpar file to avoid land sea mask correction at run time.

6. `calc_hd_receive_mask.sh`
   Coupled setups with external HD model need a mask file indicating ocean
   inflow cells. This mask depends on the fractional land sea mask.

7. `generate_hdpara_file.sh`
   Generate a HD parameter file for the defined atmosphere and ocean grid
   combination needed with internal HD.

**Note:**
This approach is meant to be preliminary. It documents the current process
of initial data generation. The aim is however, to generate all initial data
from extpar in the not so far future.

#### Usage ####
Currently, the generation of initial files for resolutions up to **R2B6** are
shown to work fine on the levante login node, while a resolution of R2B8
and up exceeds the node's memory.

To generate initial files at **R2B8+** resolution, you should submit a batch
job (`sbatch create_icon-land_ini_files.sh`) or make use of an interactive
node with large memory: Replace `ACCOUNT` with your DKRZ
project account and use
`salloc --x11 -p interactive -A ACCOUNT --mem=450GB`
to log in to an interactive node with sufficient memory before starting
`create_icon-land_ini_files.sh`.

Even higher resolutions (**R2B10+**) exceed the memory of the levante nodes.
We suggest to generate initial data for these resolutions on breeze4.
