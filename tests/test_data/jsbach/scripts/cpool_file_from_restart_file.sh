#!/bin/bash

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

set -e
##############################################################################################
### This program collects a list of variables e.g. for carbon pools from an ICON restartfile
### and packs them into a new file ###
###
### Note: Tested with nco version 5.0.1 and cdo version 2.4.2
###
# Todo: Include further variables needed in simulations with natural LCC and/or disturbances
##############################################################################################
which ncatted || {
. ${MODULESHOME}/init/bash
  module load nco
}

### User Interface
INFILE="restart_jsbach_DOM01.nc"
OUTFILE="ic_land_carbon.nc"

PROCESS="carbon"
VARNAME_LIST="c_green    c_woods     c_reserve
              c_acid_ag1 c_water_ag1 c_ethanol_ag1 c_nonsoluble_ag1
              c_acid_bg1 c_water_bg1 c_ethanol_bg1 c_nonsoluble_bg1 c_humus_1
              c_acid_ag2 c_water_ag2 c_ethanol_ag2 c_nonsoluble_ag2
              c_acid_bg2 c_water_bg2 c_ethanol_bg2 c_nonsoluble_bg2 c_humus_2"
TILE_LIST="box land veg pft01 pft02 pft03 pft04 pft05 pft06 pft07 pft08 pft09 pft10 pft11"

### Delete existing output files ?
if [ -f  ${OUTFILE} ];then
   echo "The output file ${OUTFILE} already exists. Shall I delete it?"
   echo "RETURN =File will be deleted."
   echo "XXX    =Every other input stops this script here."
   read y
   if [ ! ${y} ] ; then
      rm ${OUTFILE}
      echo "The existing file ${OUTFILE} was deleted."
      echo ""
   else
      exit
   fi
fi

### Generate variable list
varlist=""
for VARNAME in ${VARNAME_LIST} ; do
   for TILE in ${TILE_LIST} ; do
      varlist="${varlist},${PROCESS}_${VARNAME}_${TILE}"
   done
done

### Extract the variables from restart file
cdo selvar${varlist}    ${INFILE}   ${OUTFILE}.tmp1

ncks -C -x -v time  ${OUTFILE}.tmp1 ${OUTFILE}.tmp2  # Delete variable time
ncwa -O -a time     ${OUTFILE}.tmp2 ${OUTFILE}.tmp3  # Delete dimension time
ncrename -d cells,cell ${OUTFILE}.tmp3 ${OUTFILE}    # Rename dim. cells to cell

### Clean up
rm -f ${OUTFILE}.tmp?

### Finish
echo ""
echo "$0: Generation of ${OUTFILE} done"
echo ""
exit 0

