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
set -e
#
cdo="$cdo -f nc4"
ISLM=2     # Choice of land seamask variable: 1=slm, 2=cell_sea_land_mask

export LD_LIBRARY_PATH="/sw/spack-levante/netcdf-fortran-4.5.3-l2ulgp/lib":${LD_LIBRARY_PATH}
#Working Directory
#Defaults
workdir_default="/scratch/local1/m300468/icon_rdirs_r2b5_hydrosheds_plus_corr_no_sinks"
fortran_src_dir_default="$HOME/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran"
base_data_dir_default="/scratch/local1/m300468/data/ICONHDdata"
grid_file_default="gridfiles/icon_grid_0019_R02B05_G.nc"
lsmasks_file_default="lsmasks/maxlnd_lsm_019_0032.nc"
rdirs_file_default="rdirs/rdirs_019_R02B05_G_hs_with_pd_corr_inc_tc_no_ts_20200305_160921_lr.nc"
orography_file_default="orographies/orog_top1_remap_to_r2b5_0019_G_filled.nc"
bifurcated_rdirs_default="${workdir_default}/bifurcated_rdirs.nc"
bifurcated_noutflows_default="${workdir_default}/number_of_outflows.nc"
#Command Line Variables
workdir=${1:-${workdir_default}}
fortran_src_dir=${2:-${fortran_src_dir_default}}
base_data_dir=${3:-${base_data_dir_default}}
grid_file=${4:-${grid_file_default}}
lsmasks_file=${5:-${lsmasks_file_default}}
rdirs_file=${6:-${rdirs_file_default}}
orography_file=${7:-${orography_file_default}}
use_bifurcations=${8:-false}
if ${use_bifurcations}; then
  bifurcated_rdirs=${9:-${bifurcated_rdirs_default}}
  bifurcated_noutflows=${10:-${bifurcated_noutflows_default}}
else
  bifurcated_rdirs=""
  bifurcated_noutflows=""
fi

cd $workdir
# **** HD-Parameter-Dateien fuer ICON erstellen
echo 'HD Parameter erzeugen'
pwd > ddir.inp
DDIR=`cat ddir.inp`
#
cat > paragen.inp << EOF1
# Initialisierungsdatei fuer Programm PARAGEN:

# IPARA : Art der Parameterisierung (1 = Sausen-Analogie)
8
# ISLOPE: Use of inner slope (1) or normal slope (0) for Overland flow)
0
# IQUE  : Kommentarvariable ( 0 = Kein Kommentar )
1
# IGMEM : Baseflowspeicherinitialisierung Ja/Nein (1/0)
0
# IBASE : Baseflowparameterisierungsart (0:k=300 days, 1: 0 mit dx-Abh., 2,3=Beate)
0
# TDIRIN: Main Directory with input files and input subdirectories
$base_data_dir
# TDNARE: File with areas, distances and with neighbor_cell_index(3)
$grid_file
# ISLMVA: Choice of land sea mask variable:  1=slm, 2=cell_sea_land_mask
$ISLM
# TDNSLM: File name that includes land sea mask, e.g. slm
$lsmasks_file
# TDNFDI: File name of ICON river directions
$rdirs_file
# TDNOTO: Dateiname des gesmoothed Orographie-Arrays (Variable: cell elevation): oro_otto.nc
$orography_file
# TDNORO: Dateiname des globalen Orographie-Arrays (Variable: cell elevation)
$orography_file
# TDNSIG: Dateiname des globalen Orographie-Streuungs-Arrays
${CRES}/bc_land_sso_${CRES}_1976.nc
# TDNSLI: Dateiname der Inner Slope-Datei
${CRES}/bc_land_sso_${CRES}_1976.nc
# TDNLAK: Dateiname der Lake-Percentage-Datei: VarName=f_lakes
${CRES}/cwater_${CRES}.nc
# TDNWET: Dateiname der Swamp-Percentage-Datei: VarName=f_wet
${CRES}/wetlands_${CRES}.nc
# ILAMOD: Modell der Lake-Dependance (0=ohne Lakes,1=Charbonneau, 2=tanh)
2
# ISWMOD: Modell der Swamp-Dependance (0=ohne Swamps, 1=swamps+lakes, 2=tanh, 4=Ov.)
5
# VLA100: Flow-Velocity bei 100 % Lake-Percentage [m/s] (0.0003 bei tanh)
0.01
# VSW100: Flow-Velocity bei 100 % Swamp-Percentage [m/s]: 0.077 = 200 km/month
0.06
# PROARE: Areale Percentage, ab der Lake- oder Swamp-Percentage sich auswirkt
50.
# FK_LFK: Modifizierungsfaktor fuer k-Werte beim Overlandflow
1.
# FK_LFN: Modifizierungsfaktor fuer n-Werte beim Overlandflow
1.
# FK_RFK: Modifizierungsfaktor fuer k-Werte beim Riverflow
1.
# FK_RFN: Modifizierungsfaktor fuer n-Werte beim Riverflow
1.
# FK_GFK: Modifizierungsfaktor fuer k-Werte beim Baseflow
1.
# The End
EOF1
#
# Compile RDF program
rm paragen_icon_driver || true
#Compile ICON paragen driver
if [[ $(uname -s) == "Darwin" ]]; then
  netcdf_f="/usr/local/Cellar/netcdf-fortran/4.6.0"
  netcdf_f_lib="libnetcdff.dylib"
elif [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
  netcdf_f="/sw/spack-levante/netcdf-fortran-4.5.3-l2ulgp"
  netcdf_f_lib="libnetcdff.so"
else
  netcdf_f="/sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gccsys/lib"
fi

gfortran -o paragen_icon_driver ${fortran_src_dir}/mo_read_icon_trafo.f90 ${fortran_src_dir}/paragen_icon.f90 ${fortran_src_dir}/paragen_icon_driver.f90 -I${netcdf_f}/include/ -L${netcdf_f}/lib -lnetcdff
# Run paragen driver
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gccsys/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/rhel6-x64/netcdf/netcdf_fortran-4.4.4-gcc64/lib
./paragen_icon_driver
# *************  Conversion to NetCDF
cat > hd_partab.txt << EOF2
&parameter
 param        = 701
 name         = FDIR
 long_name    = "ICON index of Flow Destination"
 /
&parameter
 param        = 702
 name         = ALF_K
 long_name    = "HD model parameter Overland flow k"
 /
&parameter
 param        = 703
 name         = ALF_N
 long_name    = "HD model parameter Overland flow n"
 /
&parameter
 param        = 704
 name         = ARF_K
 long_name    = "HD model parameter Riverflow k"
 /
&parameter
 param        = 705
 name         = ARF_N
 long_name    = "HD model parameter Riverflow n"
 /
&parameter
 param        = 706
 name         = AGF_K
 long_name    = "HD model parameter Baseflow k"
 /
&parameter
 param        = 708
 name         = FLON
 long_name    = "Longitude of ICON gridbox"
 /
&parameter
 param        = 709
 name         = FLAT
 long_name    = "Latitude of ICON gridbox"
 /
EOF2
#
  EXP=1
  echo 'Create HD-ICON Parameter file for' $EXP
  cdo setpartabn,hd_partab.txt  fdir_icon.nc hd_para_icon_${EXP}_pre.nc
#
# Remove temporary files
  rm fdir_icon.nc
#
# *** Veronikas addon
#------------------------------------------------------------------------------
# modify hd_para
#  - generate hd model mask
#       ocean:            -1
#       ocean inflow:      0
#       land with outflow: 1
#       internal drainage: 2
#  - merge with hd initial file from St. Hagemann
#  - merge with upstream_cells.nc (generated with hd_upstream_cells.f90)
#
# Veronika Gayler, Sept. 2015
#------------------------------------------------------------------------------

#
  if ${use_bifurcations}; then
    ln -s ${base_data_dir}/${bifurcated_rdirs} bifurcated_next_cell_index_for_upstream_cell.nc
  fi
  cp hd_para_icon_${EXP}_pre.nc hdpara_icon.nc
  rm hd_up || true
  gfortran -o hd_up ${fortran_src_dir}/upstream_cell_icon.f90 -I${netcdf_f}/include/ -L${netcdf_f}/lib -lnetcdff
  if ${use_bifurcations}; then
    ./hd_up 1
  else
    ./hd_up
  fi
  $cdo setcode,715 upstream_cells.nc test.nc
  mv test.nc upstream_cells.nc
#
  DNGRID="${base_data_dir}/${grid_file}"
  if [[ $(cdo showvar ${DNGRID} | grep cell_index) != "" ]]; then
    $cdo selvar,cell_index ${DNGRID} cell_index.nc
  else
    # DWD grids typically use a cdi:ignore attribute for variable cell_index. This attribute needs
    # to be removed to extract the variable using CDOs.
    ncatted -O -a cdi,cell_index,d,c, ${DNGRID} ${grid_file}.tmp
    $cdo selvar,cell_index ${grid_file}.tmp cell_index.nc
    rm ${grid_file}.tmp
  fi
  $cdo -setgrid,${DNGRID} -selvar,FDIR hd_para_icon_${EXP}_pre.nc FDIR.nc

  $cdo -eqc,-5 FDIR.nc int_drainage_mask.nc
  $cdo -ifthenelse int_drainage_mask.nc cell_index.nc FDIR.nc new_FDIR.nc
  mv new_FDIR.nc FDIR.nc

  $cdo -gtc,0 FDIR.nc hdmask.tmp

  $cdo -ltc,-1 FDIR.nc inner_ocean_mask.nc
  $cdo -mulc,-1 inner_ocean_mask.nc inner_ocean_val.nc

  $cdo -eqc,-1 FDIR.nc outer_ocean_mask.nc
  $cdo -mulc,0 FDIR.nc outer_ocean_val.nc

  $cdo -eq FDIR.nc cell_index.nc int_drainage_mask.nc
  $cdo -mulc,2 int_drainage_mask.nc int_drainage_val.nc

  $cdo -ifthenelse int_drainage_mask.nc int_drainage_val.nc hdmask.tmp  hdmask.tmp1
  $cdo -ifthenelse inner_ocean_mask.nc  inner_ocean_val.nc  hdmask.tmp1 hdmask.tmp2
  $cdo -ifthenelse outer_ocean_mask.nc  outer_ocean_val.nc hdmask.tmp2 hdmask.tmp3
  $cdo setname,MASK -setcode,714 hdmask.tmp3 hdmask.nc

  hdpara="hdpara_icon"
  rm -f ${hdpara}.nc
  if ${use_bifurcations}; then
    bifurcated_noutflows_renamed="noutflows.nc"
    bifurcated_rdirs_renamed="bifurcated_rdirs.nc"
    $cdo chname,num_outflows,NSPLIT ${base_data_dir}/${bifurcated_noutflows} ${bifurcated_noutflows_renamed}
    $cdo chname,bifurcated_next_cell_index,BIFURCATED_FDIR ${base_data_dir}/${bifurcated_rdirs} ${bifurcated_rdirs_renamed}
  else
    bifurcated_noutflows_renamed=""
    bifurcated_rdirs_renamed=""
  fi
  $cdo merge hd_para_icon_${EXP}_pre.nc hdmask.nc upstream_cells.nc ${bifurcated_noutflows_renamed} ${bifurcated_rdirs_renamed} hd_para_icon_${EXP}_pre2.nc
# correct dimension name (gets lost with cdo merge)
  $cdo setgrid,$DNGRID hd_para_icon_${EXP}_pre2.nc ${hdpara}.nc
  #ncrename -d lev,nneigh ${hdpara}.nc
  ncatted -O -h -a Version,global,o,c,"HD-ICON Model Parameter Version ${VS}" ${hdpara}.nc
# clean up
  rm FDIR.nc
  rm inner_ocean_mask.nc inner_ocean_val.nc
  rm int_drainage_mask.nc int_drainage_val.nc
  rm outer_ocean_mask.nc outer_ocean_val.nc
  rm hdmask.tmp  hdmask.tmp1 hdmask.tmp2 hdmask.tmp3
  rm hdmask.nc
  rm -f noutflows.nc
  rm -f bifurcated_rdirs.nc
#
  rm hd_para_icon_${EXP}_pre.nc hd_para_icon_${EXP}_pre2.nc
  rm upstream_cells.nc cell_index.nc

#
exit
#
