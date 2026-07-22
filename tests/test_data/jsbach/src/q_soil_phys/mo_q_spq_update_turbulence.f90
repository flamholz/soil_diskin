!> QUINCY soil-turbulence calculation
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
!>#### calculate the task update_soil_turbulence, i.e., surface condition & turbulence, drag coefficient
!>
MODULE mo_q_spq_update_turbulence
#ifndef __NO_QUINCY__

  USE mo_kind,                  ONLY: wp
  USE mo_jsb_control,           ONLY: debug_on, jsbach_runs_standalone
  USE mo_exception,             ONLY: message

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: update_soil_turbulence

  CHARACTER(len=*), PARAMETER :: modname = 'mo_q_spq_update_turbulence'

CONTAINS

  ! ======================================================================================================= !
  !>update soil moisture and theta for the plant only model - NOTE update/improve docu
  !>
  ! two cases considered:
  !   a) dynamic water uptake calculation
  !   b) prescribed values from namelist
  SUBROUTINE update_soil_turbulence(tile, options)

    USE mo_jsb_class,              ONLY: Get_model
    USE mo_jsb_process_class,      ONLY: A2L_, SPQ_, SSE_, VEG_, TURB_
    USE mo_jsb_tile_class,         ONLY: t_jsb_tile_abstract
    USE mo_jsb_task_class,         ONLY: t_jsb_task_options
    USE mo_jsb_model_class,        ONLY: t_jsb_model
    USE mo_atmland_constants,      ONLY: min_wind, max_wind
    USE mo_spq_constants,          ONLY: height_wind, height_humidity, w_snow_min
    USE mo_phy_schemes,            ONLY: update_drag
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Use_memory(A2L_)
    dsl4jsb_Use_memory(SPQ_)
    dsl4jsb_Use_memory(SSE_)
    dsl4jsb_Use_memory(VEG_)
    dsl4jsb_Use_memory(TURB_)
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout)     :: tile         !< one tile with data structure for one lct
    TYPE(t_jsb_task_options),   INTENT(in)        :: options      !< model options
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model),      POINTER       :: model                 !< the model
    REAL(wp), DIMENSION(options%nc)       :: wind_air
    REAL(wp), DIMENSION(options%nc)       :: hlp1
    REAL(wp), DIMENSION(options%nc)       :: ddrag_srf ! Change of drag with time, not used by Quincy
    INTEGER                               :: iblk, ics, ice, nc, ic!< dimensions
    INTEGER                               :: model_scheme
    LOGICAL                               :: jsb_standalone
    REAL(wp)                              :: dtime               !< ...
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':update_soil_turbulence'
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_memory(A2L_)
    dsl4jsb_Def_memory(SPQ_)
    dsl4jsb_Def_memory(SSE_)
    dsl4jsb_Def_memory(VEG_)
    dsl4jsb_Def_memory(TURB_)
    ! ----------------------------------------------------------------------------------------------------- !
    ! A2L_
    dsl4jsb_Real2D_onChunk                :: t_air
    dsl4jsb_Real2D_onChunk                :: q_air
    dsl4jsb_Real2D_onChunk                :: wind_10m
    dsl4jsb_Real2D_onChunk                :: press_srf
    ! TURB_ 2D
    dsl4jsb_Real2D_onChunk                :: fact_q_air
    dsl4jsb_Real2D_onChunk                :: fact_qsat_srf
    dsl4jsb_Real2D_onChunk                :: rough_m
    dsl4jsb_Real2D_onChunk                :: rough_h
    dsl4jsb_Real2D_onChunk                :: drag_srf_stdalone
    dsl4jsb_Real2D_onChunk                :: pch_stdalone
    dsl4jsb_Real2D_onChunk                :: t_acoef_stdalone
    dsl4jsb_Real2D_onChunk                :: t_bcoef_stdalone
    dsl4jsb_Real2D_onChunk                :: q_acoef_stdalone
    dsl4jsb_Real2D_onChunk                :: q_bcoef_stdalone
    ! SPQ_ 3D
    dsl4jsb_Real3D_onChunk                :: w_snow_snl
    dsl4jsb_Real3D_onChunk                :: t_snow_snl
    ! SSE_ 3D
    dsl4jsb_Real3D_onChunk                :: t_soil_sl
    ! VEG_
    dsl4jsb_Real2D_onChunk                :: height
    ! ----------------------------------------------------------------------------------------------------- !
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc
    dtime   = options%dtime
    ! ----------------------------------------------------------------------------------------------------- !
    IF (.NOT. tile%Is_process_calculated(SPQ_)) RETURN
    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')
    ! ----------------------------------------------------------------------------------------------------- !
    model  => Get_model(tile%owner_model_id)
    model_scheme = model%config%model_scheme
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_memory(A2L_)
    dsl4jsb_Get_memory(SPQ_)
    dsl4jsb_Get_memory(SSE_)
    dsl4jsb_Get_memory(VEG_)
    dsl4jsb_Get_memory(TURB_)
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_var2D_onChunk(A2L_, t_air)                ! in
    dsl4jsb_Get_var2D_onChunk(A2L_, q_air)                ! in
    dsl4jsb_Get_var2D_onChunk(A2L_, wind_10m)             ! in
    dsl4jsb_Get_var2D_onChunk(A2L_, press_srf)            ! in
    ! ---------------------------
    dsl4jsb_Get_var2D_onChunk(TURB_, fact_q_air)          ! in
    dsl4jsb_Get_var2D_onChunk(TURB_, fact_qsat_srf)       ! in
    dsl4jsb_Get_var2D_onChunk(TURB_, rough_m)             ! in
    dsl4jsb_Get_var2D_onChunk(TURB_, rough_h)             ! in
    dsl4jsb_Get_var2D_onChunk(TURB_, drag_srf_stdalone)   ! out
    dsl4jsb_Get_var2D_onChunk(TURB_, pch_stdalone)        ! out
    dsl4jsb_Get_var2D_onChunk(TURB_, t_acoef_stdalone)    ! out
    dsl4jsb_Get_var2D_onChunk(TURB_, t_bcoef_stdalone)    ! out
    dsl4jsb_Get_var2D_onChunk(TURB_, q_acoef_stdalone)    ! out
    dsl4jsb_Get_var2D_onChunk(TURB_, q_bcoef_stdalone)    ! out
    ! ---------------------------
    dsl4jsb_Get_var3D_onChunk(SPQ_, t_snow_snl)           ! in
    dsl4jsb_Get_var3D_onChunk(SPQ_, w_snow_snl)           ! in
    ! ---------------------------
    dsl4jsb_Get_var3D_onChunk(SSE_, t_soil_sl)            ! in
    ! ---------------------------
    dsl4jsb_Get_var2D_onChunk(VEG_, height)               ! in
    ! ----------------------------------------------------------------------------------------------------- !


    !$ACC DATA CREATE(wind_air(:), ddrag_srf(:), hlp1(:))

    !> 0.9 init local variable
    !>
    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
    DO ic = 1, nc
      wind_air(ic) = MIN(MAX(wind_10m(ic), min_wind), max_wind)
    END DO
    !$ACC END PARALLEL LOOP

    !> 1.0 temporary solution for update_drag
    !>
    ! at a later point update_drag() may be called from SEB_
    ! only for ICON-Land standalone (i..e, not coupled to the atmosphere)
    jsb_standalone = jsbach_runs_standalone()
    IF (jsb_standalone) THEN

      ! calculate the quincy equivalent of t_srf_proc (t(:) surface temperature) in jsbach
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
      DO ic = 1, nc
        IF (w_snow_snl(ic,1) > w_snow_min) THEN
          hlp1(ic) = t_snow_snl(ic,1)
        ELSE
          hlp1(ic) = t_soil_sl(ic,1)
        END IF
      END DO
      !$ACC END PARALLEL LOOP

      ! calc drag
      CALL update_drag( &
        ! INTENT in
        & nc, dtime, &
        & model_scheme, &
        & t_air(:), press_srf(:), q_air(:), wind_air(:), &
        & hlp1(:), &          ! jsbach t(:) from SEB_ (surface temperature)
        & fact_q_air(:), &
        & fact_qsat_srf(:), &
        & rough_h(:), &
        & rough_m(:), &
        & height_wind, &      ! jsbach: forcing_options(tile%owner_model_id)%heightWind
        & height_humidity, &  ! jsbach: forcing_options(tile%owner_model_id)%heightHumidity
        & 0.50_wp, &          ! jsbach default value dsl4jsb_Config(SEB_)%coef_ril_tm1
        & 0.25_wp, &          ! jsbach default value dsl4jsb_Config(SEB_)%coef_ril_t
        & 0.25_wp, &          ! jsbach default value dsl4jsb_Config(SEB_)%coef_ril_tp1
        ! INTENT out
        & drag_srf_stdalone(:), ddrag_srf(:), t_acoef_stdalone(:), t_bcoef_stdalone(:), &
        & q_acoef_stdalone(:), q_bcoef_stdalone(:), pch_stdalone(:), &
        ! optional (INTENT(IN)) argument
        & veg_height = height) ! vegetation height (quincy specific)
    END IF

    !$ACC WAIT(1)
    !$ACC END DATA
  END SUBROUTINE update_soil_turbulence

#endif
END MODULE mo_q_spq_update_turbulence
