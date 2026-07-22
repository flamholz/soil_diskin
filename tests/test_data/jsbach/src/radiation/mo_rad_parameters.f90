!> QUINCY radiation parameters
!>
!> ICON-Land
!>
!> ---------------------------------------
!> Copyright (C) 2013-2024, MPI-M, MPI-BGC
!>
!> Contact: icon-model.org
!> Authors: AUTHORS.md
!> See LICENSES/ for license information
!> SPDX-License-Identifier: BSD-3-Clause
!> ---------------------------------------
!>
MODULE mo_rad_parameters
#ifndef __NO_QUINCY__

  USE mo_kind,                ONLY: wp
  USE mo_jsb_impl_constants,  ONLY: def_parameters

  IMPLICIT NONE
  PUBLIC

  ! Parameters required for albedo and fapar calculation
  REAL(wp), SAVE :: &
    & kbl0_vis = def_parameters, &               !< extinction coefficient over black leaves
    & kdf0_vis = def_parameters, &               !< extinction coefficient for diffuse PAR (Spitters eq. 7)
    & kbl0_nir = def_parameters, &               !< extinction coefficient over black leaves in NIR range
    & kdf0_nir = def_parameters, &               !< extinction coefficient for diffuse NIR (Spitters eq. 7)
    & rho2sbeta = def_parameters, &              !< scaling factor of solar angle in reflection calculation
    & k_csf = def_parameters, &                  !< scaling factor in the crown-shape effect on clumping (Campbell & Norman, 1998, eq 15.35)
    & min_cos_zenith_angle_rad = def_parameters  !< minimum coszine zenith angle for radiation calculation

  REAL(wp), SAVE :: &
    & albedo_stem_vis    = def_parameters, &  !< Stem albedo (visible range)                                  [-]
    & albedo_stem_nir    = def_parameters, &  !< Stem albedo (nir range)                                      [-]
    & kbl_stem           = def_parameters     !< Stem absorption (1-transmissivity (vis,nir))                 [-]

  ! conversion factors
  REAL(wp), SAVE :: &
    & rad2ppfd = def_parameters             !< micro-mol /J

  REAL(wp), SAVE :: &
    & MinRadiation = def_parameters         !< Minimum amount of radiation for albedo calculations [W/m2]

  CHARACTER(len=*), PARAMETER, PRIVATE :: modname = 'mo_rad_parameters'

CONTAINS

  ! ======================================================================================================= !
  !> initialize parameters for the process: radiation
  !>
  SUBROUTINE init_rad_parameters
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':init_rad_parameters'

    ! Parameters required for albedo and fapar calculation
    kbl0_vis                  = 0.5_wp        !< Spitters 1986
    kdf0_vis                  = 0.8_wp        !< Spitters 1986
    kbl0_nir                  = 0.5_wp        !< Spitters 1986
    kdf0_nir                  = 0.8_wp        !< Spitters 1986
    rho2sbeta                 = 1.6_wp        !< Spitters 1986
    k_csf                     = 2.2_wp        !< Campbell & Norman 1998, eq 15.35
    min_cos_zenith_angle_rad  = 0.1736482_wp  !< corresponding to 10 deg solar elevation, below which the
                                              !! radiation equations are no longer valid
    albedo_stem_vis           = 0.16_wp       !< Belda et al. 2022, https://doi.org/10.5194/gmd-15-6709-2022
    albedo_stem_nir           = 0.39_wp       !< Belda et al. 2022, https://doi.org/10.5194/gmd-15-6709-2022
    kbl_stem                  = 0.999_wp      !< Belda et al. 2022, https://doi.org/10.5194/gmd-15-6709-2022
    ! conversion factors
    rad2ppfd                  = 4.6_wp        !< Monteith & Unsworth 1995
    ! parameters also used in jsbach
    MinRadiation              = 1.0e-09_wp
  END SUBROUTINE init_rad_parameters

#endif
END MODULE mo_rad_parameters
