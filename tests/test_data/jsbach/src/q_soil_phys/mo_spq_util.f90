!> QUINCY helper routines for soil-physics-quincy
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
!> For more information on the QUINCY model see: <https://doi.org/10.17871/quincy-model-2019>
!>
!>#### various helper routines for the soil-physics-quincy process
!>
MODULE mo_spq_util
#ifndef __NO_QUINCY__

  USE mo_kind,          ONLY: wp
  USE mo_exception,     ONLY: finish, message, message_text
  USE mo_jsb_control,   ONLY: debug_on

  IMPLICIT NONE

  PRIVATE
  PUBLIC :: calc_qmax_texture
  PUBLIC :: reset_spq_fluxes           ! called in the land model interface prior to all Tasks

  CHARACTER(len=*), PARAMETER :: modname = 'mo_spq_util'

CONTAINS

  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION calc_qmax_texture
  !!
  !! Output: sorption capacity for organic material
  !-----------------------------------------------------------------------------------------------------
  ELEMENTAL FUNCTION calc_qmax_texture(qmax_fine_particle, silt_sl, clay_sl) RESULT(qmax_texture)


    IMPLICIT NONE
    ! ---------------------------
    ! 0.1 InOut
    REAL(wp), INTENT(in) :: qmax_fine_particle    !< maximum sorption capacity of fine soil particle [mol / kg fine particle (silt or clay)]
    REAL(wp), INTENT(in) :: silt_sl               !< silt content of the mineral soil [kg silt / kg mineral soil]
    REAL(wp), INTENT(in) :: clay_sl               !< clay content of the mineral soil [kg clay / kg mineral soil]
    REAL(wp)             :: qmax_texture          !< maximum sorption capacity of the soil [mol / kg]
    ! ---------------------------
    ! 0.2 Local
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':calc_qmax_texture'

    ! maximum sorption capacity per kg mineral soil (mol/kg)
    qmax_texture = qmax_fine_particle * (silt_sl + clay_sl)

  END FUNCTION calc_qmax_texture


  !-----------------------------------------------------------------------------------------------------
  !> init/reset soil fluxes (with/to zero)
  !!
  !! called in the land model interface prior to all Tasks
  !-----------------------------------------------------------------------------------------------------
  SUBROUTINE reset_spq_fluxes(tile, options)

    USE mo_jsb_class,             ONLY: Get_model
    USE mo_jsb_tile_class,        ONLY: t_jsb_tile_abstract
    USE mo_jsb_task_class,        ONLY: t_jsb_task_options
    USE mo_jsb_model_class,       ONLY: t_jsb_model
    USE mo_jsb_grid_class,        ONLY: t_jsb_vgrid
    USE mo_jsb_grid,              ONLY: Get_vgrid
    USE mo_jsb_process_class,     ONLY: SPQ_, HYDRO_
    USE mo_jsb_math_constants,    ONLY: zero

    ! Use of process memories
    dsl4jsb_Use_memory(SPQ_)
    dsl4jsb_Use_memory(HYDRO_)

    IMPLICIT NONE

    dsl4jsb_Real2D_onChunk :: transpiration
    dsl4jsb_Real3D_onChunk :: w_soil_freeze_flux
    dsl4jsb_Real3D_onChunk :: w_soil_melt_flux
    ! ---------------------------
    ! 0.1 InOut
    CLASS(t_jsb_tile_abstract), INTENT(inout)     :: tile         !< one tile with data structure for one lct
    TYPE(t_jsb_task_options),   INTENT(in)        :: options      !< model options
    ! ---------------------------
    ! 0.2 Local
    TYPE(t_jsb_model),      POINTER       :: model
    TYPE(t_jsb_vgrid),      POINTER       :: vgrid_soil_sb        !< Vertical grid
    INTEGER                               :: nsoil_sb             !< number of soil layers
    INTEGER                               :: isoil, ic
    INTEGER                               :: iblk, ics, ice, nc
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':reset_spq_fluxes'
    ! ---------------------------
    ! 0.3 Declare Memory
    ! Declare process configuration and memory Pointers
    dsl4jsb_Def_memory(SPQ_)
    dsl4jsb_Def_memory(HYDRO_)
    ! Get local variables from options argument
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc
    ! ---------------------------
    ! 0.4 Process Activity, Debug Option
    IF (.NOT. tile%Is_process_calculated(SPQ_)) RETURN
    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')
    ! ---------------------------
    ! 0.5 Get Memory
    ! Get process memories
    dsl4jsb_Get_memory(SPQ_)
    dsl4jsb_Get_memory(HYDRO_)
    model  => Get_model(tile%owner_model_id)
    vgrid_soil_sb => Get_vgrid('soil_layer_sb')
    nsoil_sb      =  vgrid_soil_sb%n_levels
    ! --------------------------------------------------------------------------------------------------------
    dsl4jsb_Get_var2D_onChunk(HYDRO_, transpiration)
    dsl4jsb_Get_var3D_onChunk(SPQ_,   w_soil_freeze_flux)
    dsl4jsb_Get_var3D_onChunk(SPQ_,   w_soil_melt_flux)

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
    DO ic = 1, nc
      transpiration(ic) = zero
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1)
    DO isoil = 1, nsoil_sb
      DO ic = 1, nc
        w_soil_freeze_flux(ic,isoil) = zero
        w_soil_melt_flux(ic,isoil)   = zero
      END DO
    END DO
    !$ACC END PARALLEL LOOP

  END SUBROUTINE reset_spq_fluxes

#endif
END MODULE mo_spq_util
