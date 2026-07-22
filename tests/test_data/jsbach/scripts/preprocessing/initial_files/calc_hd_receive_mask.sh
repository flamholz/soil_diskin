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
# Script to calculate the hd_receive_mask for icoupled ICON simulations
# with external HD.
#
# Based on icon/externals/hd/util/calc_hd_receive_mask.ksh
#
#------------------------------------------------------------------------------
set -e
prog=$(basename $0)

tmpdir=${work_dir}/hd_tmp
mkdir -p ${tmpdir}; rm -f ${tmpdir}/*.nc
cd ${tmpdir}

cdo="$cdo"

echo "=============================================================="
echo "===  Generation of the HD receive mask"
echo "=============================================================="

if [[ ${coupled} != true ]]; then
  echo ""
  echo "$0: ERROR: HD recieve mask can only be calculated for coupled setups"
  echo ""
  exit 1
fi

# Create the file hd_receive.nc

# 1. Select mask with all HD land points
$cdo gtc,0. -selvar,FDIR ${hdpara05_file} ./mask_hd.nc

# 2. Select ICON points that have land fraction >=0.05
$cdo gec,0.05 -selvar,cell_sea_land_mask ${fractional_mask} ./iconcoupmask.nc

# 3. Interpolate a mask with all ICON land/lake points to the HD grid
$cdo -L -remapycon,mask_hd.nc ./iconcoupmask.nc ./icon_to_hd_05.nc

# 4. Rename mask variable and output file
$cdo setvar,hd_receive_mask icon_to_hd_05.nc ${hd_receive}

echo "-------------------------------------------------------------------"
echo "$prog: Generated HD receive file:"
echo "     ${hd_receive}"
echo "----------------------------------------------------------------"

cd ..
rm -rf ${tmpdir}
