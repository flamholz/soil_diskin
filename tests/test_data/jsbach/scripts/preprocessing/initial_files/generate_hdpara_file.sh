#!/bin/bash
#
# ICON-Land
#
# ---------------------------------------
# Copyright (C) 2013-2024, MPI-M, MPI-BGC
#
# Contact: icon-model.org
# Authors: AUTHORS.md
# See LICENSES/ for license information
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------
#
# Script to generate the HD parameter file for simulations with ICON-Land and
# internal HD.  It is based on the instructions given in
# https://gitlab.dkrz.de/jsbach/jsbach/-/wikis/Documentation/ICON-Land-input-data/Icon-Land-initial-file-generation/low-res-hdpara-file-generation
#
#------------------------------------------------------------------------------
set -e
prog=$(basename $0)

tmpdir=${work_dir}/hd_tmp
mkdir -p ${tmpdir}; rm -f ${tmpdir}/*.nc
cd ${tmpdir}

cdo="$cdo"

echo "=============================================================="
echo "===  Generation of the HD parameter file"
echo "=============================================================="

# Currently only working for low atmospheric model resolutions (R02B06 or lower)
if [[   $(echo ${atmRes} | cut -c2-3 ) -gt 2  \
     || $(echo ${atmRes} | cut -c5-6 ) -gt 6 ]]; then
  echo "$0: ERROR: "
  echo "$0: HD parameter file generation currently not possible for grid resolution greater then R02B06"
  echo " "
  exit 1
fi

# 1. Setup the repository with icon hdpara generation tools
git clone https://github.com/ThomasRiddick/DynamicHD.git icon_hdpara_generation_tools
cd icon_hdpara_generation_tools
git checkout icon_hd_tools_version_1.4b
cp -r ${scripts_dir}/dynamic_hd_tools/*  Dynamic_HD_bash_scripts/parameter_generation_scripts/

# 2. Generate the necessary mamba environment (can be slow; needs to be done only once)
if [[ $(mamba info -e | grep "dyhdenv_mamba") == "" ]]; then
  ./Dynamic_HD_bash_scripts/regenerate_conda_environment.sh
fi

# 3. Activate the mamba environment
source activate dyhdenv_mamba

# 4. Compile code and generate example run script configuration files
make

# 5. Further preparations

# Use the fractional land sea mask from the bc_land_frac file.
# Note: Due to the min- and max_fract definition, the fractional mask in the bc_land_frac
# file ('notsea') is generally not identical with the fractional mask in the fractional_mask
# file. The expected variable name for the HD parameter generation tools is 'cell_sea_land_mask'.
if [[ $(cdo showvar ${hd_fractional_mask} | grep notsea ) != "" ]]; then
  $cdo setvar,cell_sea_land_mask -selvar,notsea ${hd_fractional_mask} ${tmpdir}/fractional_mask.nc
  fractional_lsmask_filepath=${tmpdir}/fractional_mask.nc
else
  if [[ $(cdo showvar ${hd_fractional_mask} | grep cell_sea_land_mask ) != "" ]]; then
    fractional_lsmask_filepath=${hd_fractional_mask}
  else
    echo "$0: ERROR: ICON-Land bc file ${hd_fractional_mask} "\
              "does not exist or does not contain variable 'notsea' nor 'cell_sea_land_mask'."
    echo ""
    exit 1
  fi
fi

# Change format of AtmRes: e.g. R02B04 -> r2b4
icon_atmo_grid_res=$(echo ${atmRes} | tr -s 'RB' 'rb' | tr -d 0)

# 6. Create and edit a run script configuration file
cd run
sed     "s/icon_atmo_grid_id=.*/icon_atmo_grid_id=${atmGridID}/"          examples/r2b3_example.cfg \
  | sed "s/icon_ocean_grid_id=.*/icon_ocean_grid_id=${oceGridID}/"                                  \
  | sed "s/icon_atmo_grid_res=.*/icon_atmo_grid_res=${icon_atmo_grid_res}/"                         \
  | sed "s:icon_grid_filepath=.*:icon_grid_filepath=${icon_grid}:"                                  \
  | sed "s:fractional_lsmask_filepath=.*:fractional_lsmask_filepath=${fractional_lsmask_filepath}:" \
    > ${icon_atmo_grid_res}.cfg
~/.conda/envs/dyhdenv_mamba/bin/python ../utils/run_utilities/mkproject.py ${icon_atmo_grid_res}.cfg

# 7. Run the script (use an appropriate interactive node)
./${icon_atmo_grid_res}.run

# 8. Save the new hdpara file and clean up
hdpara_file=$(ls ../projects/${icon_atmo_grid_res}/output/hdpara_${icon_atmo_grid_res}_${atmGridID}_${oceGridID}_*.nc)
hdpara_file_new=$(echo ${hdpara_file##*/} | tr -s _ )  # remove path and extra '_' in case of empty oceGridID
cp ${hdpara_file} ${bc_file_dir}/${hdpara_file_new}

echo "----------------------------------------------------------------"
echo "$0: Generated HD parameter file:"
echo "     ${bc_file_dir}/${hdpara_file_new}"
echo "----------------------------------------------------------------"

cd ..
rm -rf ${tmpdir}

