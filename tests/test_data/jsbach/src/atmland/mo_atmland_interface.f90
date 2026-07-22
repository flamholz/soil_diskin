!> Interface to run JSBACH / QUINCY for one time step
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
MODULE mo_atmland_interface
#ifndef __NO_JSBACH__

  USE mo_exception,       ONLY: finish
  USE mo_kind,            ONLY: wp

  USE mo_jsb_control,        ONLY: jsbach_runs_standalone, acc_stream
  USE mo_jsb_model_class,    ONLY: t_jsb_model, MODEL_QUINCY
  USE mo_jsb_class,          ONLY: Get_model
  USE mo_jsb_grid,           ONLY: Get_grid
  USE mo_jsb_grid_class,     ONLY: t_jsb_grid
  USE mo_jsb_tile_class,     ONLY: t_jsb_tile_abstract
  USE mo_jsb_lct_class,      ONLY: LAKE_TYPE
  !USE mo_jsb_config_class,   ONLY: t_jsb_config
  !USE mo_jsb_process_class,  ONLY: t_jsb_process
  USE mo_jsb_task_class,     ONLY: t_jsb_task_options

  dsl4jsb_Use_processes A2L_, SEB_, TURB_, HYDRO_, RAD_, CARBON_, VEG_
  dsl4jsb_Use_config(SEB_)
  dsl4jsb_Use_memory(A2L_)
  dsl4jsb_Use_memory(SEB_)
  dsl4jsb_Use_memory(TURB_)
  dsl4jsb_Use_memory(HYDRO_)
  dsl4jsb_Use_memory(RAD_)
  dsl4jsb_Use_memory(CARBON_)

#ifndef __NO_QUINCY__
  dsl4jsb_Use_memory(VEG_)
#endif

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: update_atm2land, update_land2atm

  CHARACTER(len=*), PARAMETER :: modname = 'mo_atmland_interface'

CONTAINS

  SUBROUTINE update_atm2land( &
    & tile, options,          &
    & t_air,                  &
    & q_air,                  &
    & press_air,              &
    & rain,                   &
    & snow,                   &
    & wind_air,               &
    & wind_10m,               &
    & lw_srf_down,            &
    & swvis_srf_down,         &
    & swnir_srf_down,         &
    & swpar_srf_down,         &
    & fract_par_diffuse,      &
    & dz_srf,                 &
    & press_srf,              &
    & rho_srf,                &
    & drag_srf,               &
    & t_acoef,                &
    & t_bcoef,                &
    & q_acoef,                &
    & q_bcoef,                &
    & pch,                    &
    & cos_zenith_angle,       &
    & CO2_air,                &
    & co2_dC13,               & ! for optional use with quincy
    & co2_dC14,               & ! for optional use with quincy
    ! For lakes:
    & DEBUG_VAR,              &
    & drag_wtr,               &
    & drag_ice,               &
    & t_acoef_wtr,            &
    & t_bcoef_wtr,            &
    & q_acoef_wtr,            &
    & q_bcoef_wtr,            &
    & t_acoef_ice,            &
    & t_bcoef_ice,            &
    & q_acoef_ice,            &
    & q_bcoef_ice             &
    & )

    USE mo_jsb_physical_constants, ONLY: molarMassDryAir, molarMassCO2
#ifndef __NO_QUINCY__
    USE mtime,                     ONLY: datetime
    USE mo_time_config,            ONLY: time_config
    USE mo_jsb_time,               ONLY: get_secs_of_day, get_year, is_newyear
    USE mo_iq_atm2land_process,    ONLY: update_local_time_and_daytime_counter, update_slow_sb_pool_accelerator_bookkeeping
    USE mo_isotope_util,           ONLY: calc_mixing_ratio_C13C12, calc_mixing_ratio_C14C
    USE mo_atmland_constants,      ONLY: def_co2_deltaC13, def_co2_deltaC14
#endif
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout)             :: tile
    TYPE(t_jsb_task_options),   INTENT(in)                :: options
    ! TODO This line doesn't work with intel on mistral and standalone JSBACH with ECHAM infrastructure ... why?
    ! REAL(wp), OPTIONAL, DIMENSION(options%nc), INTENT(in) ::                                                       &
    REAL(wp), OPTIONAL, DIMENSION(:), INTENT(in) ::                                                                &
      & t_air, q_air, press_air, rain, snow, wind_air, wind_10m, lw_srf_down, swvis_srf_down, swnir_srf_down, swpar_srf_down, &
      & fract_par_diffuse, dz_srf, press_srf, rho_srf, drag_srf, t_acoef, t_bcoef, q_acoef, q_bcoef, pch, cos_zenith_angle,   &
      & CO2_air, co2_dC13, co2_dC14,                                                                               &
      & DEBUG_VAR, drag_wtr, drag_ice,                                                                             &
      & t_acoef_wtr, t_bcoef_wtr, q_acoef_wtr, q_bcoef_wtr,                                                        &
      & t_acoef_ice, t_bcoef_ice, q_acoef_ice, q_bcoef_ice

    dsl4jsb_Def_memory(A2L_)

    INTEGER  :: iblk, ics, ice, nc, i
    REAL(wp) :: dtime
    TYPE(t_jsb_model), POINTER :: model
    TYPE(t_jsb_grid),  POINTER :: grid
    REAL(wp), POINTER :: lon(:)
    INTEGER :: model_scheme
    LOGICAL :: tile_contains_lake

#ifndef __NO_QUINCY__
    TYPE(datetime),    POINTER :: mtime_current !< elapsed simulation time
    INTEGER  :: global_seconds_day, current_year
    REAL(wp) :: co2_dC13_constant, co2_dC14_constant
    LOGICAL :: run_spinup_accelerator                    !< model configuration: if running with slow sb pool spin-up accelerator
    INTEGER :: sb_pool_spinup_accelerator_max_executions !< bookkeeping configurations for the spin-up accelerator: max number of executions
    INTEGER :: sb_pool_spinup_accelerator_frequency      !< bookkeeping configurations for the spin-up accelerator: frequency of executions
    INTEGER :: sb_pool_spinup_accelerator_start_year     !< bookkeeping configurations for the spin-up accelerator: start year of executions
#endif

    CHARACTER(len=*), PARAMETER :: routine = modname//':update_atm2land'

    dsl4jsb_Real2D_onChunk ::  &
      & t_air_ptr,             &
      & q_air_ptr,             &
      & press_air_ptr,         &
      & rain_ptr,              &
      & snow_ptr,              &
      & wind_air_ptr,          &
      & wind_10m_ptr,          &
      & lw_srf_down_ptr,       &
      & swvis_srf_down_ptr,    &
      & swnir_srf_down_ptr,    &
      & swpar_srf_down_ptr,    &
      & fract_par_diffuse_ptr, &
      & dz_srf_ptr,            &
      & press_srf_ptr,         &
      & rho_srf_ptr,           &
      & drag_srf_ptr,          &
      & t_acoef_ptr,           &
      & t_bcoef_ptr,           &
      & q_acoef_ptr,           &
      & q_bcoef_ptr,           &
      & pch_ptr,               &
      & cos_zenith_angle_ptr,  &
      & CO2_air_ptr,           &
      & CO2_air_mol_ptr,       &
      & DEBUG_VAR_ptr,         &
      & drag_wtr_ptr,          &
      & drag_ice_ptr,          &
      & t_acoef_wtr_ptr,       &
      & t_bcoef_wtr_ptr,       &
      & q_acoef_wtr_ptr,       &
      & q_bcoef_wtr_ptr,       &
      & t_acoef_ice_ptr,       &
      & t_bcoef_ice_ptr,       &
      & q_acoef_ice_ptr,       &
      & q_bcoef_ice_ptr

#ifndef __NO_QUINCY__
    dsl4jsb_Real2D_onChunk ::                 &
      & CO2_mixing_ratio_ptr,                 &
      & CO2_mixing_ratio_C13_ptr,             &
      & CO2_mixing_ratio_C14_ptr,             &
      & daytime_counter_ptr,                  &
      & daylength_prev_day_ptr,               &
      & local_time_day_seconds_ptr,           &
      & slow_sb_pool_accelerator_execute_ptr, &
      & slow_sb_pool_accelerator_execution_counter_ptr
#endif
    ! ----------------------------------------------------------------------------------------------------- !
    IF (ASSOCIATED(tile%parent)) CALL finish(TRIM(routine), 'Should only be called for the root tile')

    iblk  = options%iblk
    ics   = options%ics
    ice   = options%ice
    nc    = options%nc
    dtime = options%dtime

    model => Get_model(tile%owner_model_id)
    grid  => get_grid(model%grid_id)
    lon   => grid%lon(ics:ice, iblk)

    model_scheme = model%config%model_scheme

    IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__

      mtime_current => time_config%tc_current_date
      global_seconds_day = NINT(get_secs_of_day(mtime_current))  ! TODO time steps w/ fractional seconds not supported by quincy, yet!
      current_year = get_year(mtime_current)
      run_spinup_accelerator = model%config%flag_slow_sb_pool_spinup_accelerator
      sb_pool_spinup_accelerator_max_executions = model%config%slow_sb_pool_spinup_accelerator_max_executions
      sb_pool_spinup_accelerator_frequency = model%config%slow_sb_pool_spinup_accelerator_frequency
      sb_pool_spinup_accelerator_start_year = model%config%slow_sb_pool_spinup_accelerator_start_year
#endif
    END IF

    ! make this logical accessible in the OpenACC code directly
    tile_contains_lake = tile%contains_lake

    IF (nc /= SIZE(t_air,1)) CALL finish(TRIM(routine), 'Wrong dimensions')

    dsl4jsb_Get_memory(A2L_)
    IF (PRESENT(DEBUG_VAR)) THEN
      DEBUG_VAR_ptr         => dsl4jsb_var2D_onChunk(A2L_, DEBUG_VAR)
    END IF

    IF (PRESENT(t_air)) THEN
      t_air_ptr             => dsl4jsb_var2D_onChunk(A2L_, t_air)
    END IF
    IF (PRESENT(q_air)) THEN
      q_air_ptr             => dsl4jsb_var2D_onChunk(A2L_, q_air)
    END IF
    IF (PRESENT(press_air)) THEN
      press_air_ptr         => dsl4jsb_var2D_onChunk(A2L_, press_air)
    END IF
    IF(PRESENT(rain)) THEN
      rain_ptr              => dsl4jsb_var2D_onChunk(A2L_, rain)
    END IF
    IF (PRESENT(snow)) THEN
      snow_ptr              => dsl4jsb_var2D_onChunk(A2L_, snow)
    END IF
    IF (PRESENT(wind_air)) THEN
      wind_air_ptr          => dsl4jsb_var2D_onChunk(A2L_, wind_air)
    END IF
    IF (PRESENT(wind_10m)) THEN
      wind_10m_ptr          => dsl4jsb_var2D_onChunk(A2L_, wind_10m)
    END IF
    IF (PRESENT(lw_srf_down)) THEN
      lw_srf_down_ptr       => dsl4jsb_var2D_onChunk(A2L_, lw_srf_down)
    END IF
    IF (PRESENT(swvis_srf_down)) THEN
      swvis_srf_down_ptr    => dsl4jsb_var2D_onChunk(A2L_, swvis_srf_down)
    END IF
    IF(PRESENT(swnir_srf_down)) THEN
      swnir_srf_down_ptr    => dsl4jsb_var2D_onChunk(A2L_, swnir_srf_down)
    END IF
    IF (PRESENT(swpar_srf_down)) THEN
      swpar_srf_down_ptr    => dsl4jsb_var2D_onChunk(A2L_, swpar_srf_down)
    END IF
    IF (PRESENT(fract_par_diffuse)) THEN
      fract_par_diffuse_ptr => dsl4jsb_var2D_onChunk(A2L_, fract_par_diffuse)
    END IF
    IF (PRESENT(dz_srf)) THEN
      dz_srf_ptr            => dsl4jsb_var2D_onChunk(A2L_, dz_srf)
    END IF
    IF (PRESENT(press_srf)) THEN
      press_srf_ptr         => dsl4jsb_var2D_onChunk(A2L_, press_srf)
    END IF
    IF (PRESENT(rho_srf)) THEN
      rho_srf_ptr           => dsl4jsb_var2D_onChunk(A2L_, rho_srf)
    END IF
    IF(PRESENT(drag_srf)) THEN
      drag_srf_ptr          => dsl4jsb_var2D_onChunk(A2L_, drag_srf)
    END IF
    IF (PRESENT(t_acoef)) THEN
      t_acoef_ptr           => dsl4jsb_var2D_onChunk(A2L_, t_acoef)
    END IF
    IF (PRESENT(t_bcoef)) THEN
      t_bcoef_ptr           => dsl4jsb_var2D_onChunk(A2L_, t_bcoef)
    END IF
    IF (PRESENT(q_acoef)) THEN
      q_acoef_ptr           => dsl4jsb_var2D_onChunk(A2L_, q_acoef)
    END IF
    IF (PRESENT(q_bcoef)) THEN
      q_bcoef_ptr           => dsl4jsb_var2D_onChunk(A2L_, q_bcoef)
    END IF
    IF (PRESENT(pch)) THEN
      pch_ptr               => dsl4jsb_var2D_onChunk(A2L_, pch)
    END IF
    IF (PRESENT(cos_zenith_angle)) THEN
      cos_zenith_angle_ptr  => dsl4jsb_var2D_onChunk(A2L_, cos_zenith_angle)
    END IF
    IF (PRESENT(CO2_air)) THEN
      CO2_air_ptr           => dsl4jsb_var2D_onChunk(A2L_, CO2_air)
      CO2_air_mol_ptr       => dsl4jsb_var2D_onChunk(A2L_, CO2_air_mol)
#ifndef __NO_QUINCY__
      IF (model_scheme == MODEL_QUINCY) THEN
        CO2_mixing_ratio_ptr      => dsl4jsb_var2D_onChunk(A2L_, CO2_mixing_ratio)
        CO2_mixing_ratio_C13_ptr  => dsl4jsb_var2D_onChunk(A2L_, CO2_mixing_ratio_C13)
        CO2_mixing_ratio_C14_ptr  => dsl4jsb_var2D_onChunk(A2L_, CO2_mixing_ratio_C14)
      END IF
#endif
    END IF
    IF (tile_contains_lake) THEN
      IF (PRESENT(drag_wtr)) THEN
        drag_wtr_ptr          => dsl4jsb_var2D_onChunk(A2L_, drag_wtr)
      END IF
      IF (PRESENT(drag_ice)) THEN
        drag_ice_ptr          => dsl4jsb_var2D_onChunk(A2L_, drag_ice)
      END IF
      IF (PRESENT(t_acoef_wtr)) THEN
        t_acoef_wtr_ptr       => dsl4jsb_var2D_onChunk(A2L_, t_acoef_wtr)
      END IF
      IF (PRESENT(t_bcoef_wtr)) THEN
        t_bcoef_wtr_ptr       => dsl4jsb_var2D_onChunk(A2L_, t_bcoef_wtr)
      END IF
      IF (PRESENT(q_acoef_wtr)) THEN
        q_acoef_wtr_ptr       => dsl4jsb_var2D_onChunk(A2L_, q_acoef_wtr)
      END IF
      IF (PRESENT(q_bcoef_wtr)) THEN
        q_bcoef_wtr_ptr       => dsl4jsb_var2D_onChunk(A2L_, q_bcoef_wtr)
      END IF
      IF (PRESENT(t_acoef_ice)) THEN
        t_acoef_ice_ptr       => dsl4jsb_var2D_onChunk(A2L_, t_acoef_ice)
      END IF
      IF (PRESENT(t_bcoef_ice)) THEN
        t_bcoef_ice_ptr       => dsl4jsb_var2D_onChunk(A2L_, t_bcoef_ice)
      END IF
      IF (PRESENT(q_acoef_ice)) THEN
        q_acoef_ice_ptr       => dsl4jsb_var2D_onChunk(A2L_, q_acoef_ice)
      END IF
      IF (PRESENT(q_bcoef_ice)) THEN
        q_bcoef_ice_ptr       => dsl4jsb_var2D_onChunk(A2L_, q_bcoef_ice)
      END IF
    END IF

    IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__
      daytime_counter_ptr         => dsl4jsb_var2D_onChunk(A2L_, daytime_counter)
      daylength_prev_day_ptr      => dsl4jsb_var2D_onChunk(A2L_, daylength_prev_day)
      local_time_day_seconds_ptr  => dsl4jsb_var2D_onChunk(A2L_, local_time_day_seconds)
      IF (run_spinup_accelerator) THEN
        slow_sb_pool_accelerator_execute_ptr           => dsl4jsb_var2D_onChunk(A2L_, slow_sb_pool_accelerator_execute)
        slow_sb_pool_accelerator_execution_counter_ptr => dsl4jsb_var2D_onChunk(A2L_, slow_sb_pool_accelerator_execution_counter)
      END IF

      IF (PRESENT(co2_dC13) .OR. PRESENT(co2_dC14)) THEN
        CALL finish(TRIM(routine), 'As soon as 13C or 14C are provided by the atmosphere,' &
          & //' the appropriate conversion should be implemented below, depending on the passed form.')
      END IF
#endif
    END IF

    !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

    IF (PRESENT(DEBUG_VAR)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        DEBUG_VAR_ptr(i) = DEBUG_VAR(i)
      END DO
    END IF

    IF (PRESENT(t_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        t_air_ptr(i) = t_air(i)
      END DO
    END IF
    IF (PRESENT(q_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        q_air_ptr(i) = q_air(i)
      END DO
    END IF
    IF (PRESENT(press_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        press_air_ptr(i) = press_air(i)
      END DO
    END IF
    IF (PRESENT(rain)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        rain_ptr(i) = rain(i)
      END DO
    END IF
    IF (PRESENT(snow)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        snow_ptr(i) = snow(i)
      END DO
    END IF
    IF (PRESENT(wind_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        wind_air_ptr(i) = wind_air(i)
      END DO
    END IF
    IF (PRESENT(wind_10m)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        wind_10m_ptr(i) = wind_10m(i)
      END DO
    END IF
    IF (PRESENT(lw_srf_down)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        lw_srf_down_ptr(i) = lw_srf_down(i)
      END DO
    END IF
    IF (PRESENT(swvis_srf_down)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        swvis_srf_down_ptr(i) = swvis_srf_down(i)
      END DO
    END IF
    IF (PRESENT(swnir_srf_down)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        swnir_srf_down_ptr(i) = swnir_srf_down(i)
      END DO
    END IF
    IF (PRESENT(swpar_srf_down)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        swpar_srf_down_ptr(i) = swpar_srf_down(i)
      END DO
    END IF
    IF (PRESENT(fract_par_diffuse)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        fract_par_diffuse_ptr(i)  = fract_par_diffuse(i)
      END DO
    END IF
    IF (PRESENT(dz_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        dz_srf_ptr(i) = dz_srf(i)
      END DO
    END IF
    IF (PRESENT(press_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        press_srf_ptr(i) = press_srf(i)
      END DO
    END IF
    IF (PRESENT(drag_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        drag_srf_ptr(i) = drag_srf(i)
      END DO
    END IF
    IF (PRESENT(rho_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        rho_srf_ptr(i) = rho_srf(i)
      END DO
    END IF
    IF (PRESENT(t_acoef)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        t_acoef_ptr(i) = t_acoef(i)
      END DO
    END IF
    IF (PRESENT(t_bcoef)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        t_bcoef_ptr(i) = t_bcoef(i)
      END DO
    END IF
    IF (PRESENT(q_acoef)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        q_acoef_ptr(i) = q_acoef(i)
      END DO
    END IF
    IF (PRESENT(q_bcoef)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        q_bcoef_ptr(i) = q_bcoef(i)
      END DO
    END IF
    IF (PRESENT(pch)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        pch_ptr(i) = pch(i)
      END DO
    END IF
    IF (PRESENT(cos_zenith_angle)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        cos_zenith_angle_ptr(i) = cos_zenith_angle(i)
      END DO
    END IF
    IF (PRESENT(CO2_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        CO2_air_ptr(i) = CO2_air(i)
        ! Convert CO2 mass mixing ratio [kg/kg] to particle mixing ratio [mol/mol]
        CO2_air_mol_ptr(i) = CO2_air(i) * molarMassDryAir / molarMassCO2
      END DO
      IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          ! convert CO2 from "molar ratio (volume)" to "co2 mixing ratio ppmv"
          CO2_mixing_ratio_ptr(i) = CO2_air_mol_ptr(i) * 1000000._wp
        END DO

        IF (.NOT. PRESENT(co2_dC13)) THEN
          co2_dC13_constant = calc_mixing_ratio_C13C12(def_co2_deltaC13)
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO i=1,nc
            CO2_mixing_ratio_c13_ptr(i) = CO2_air(i) / (1._wp + 1._wp / co2_dC13_constant)
          END DO
        END IF

        IF (.NOT. PRESENT(co2_dC14)) THEN
          co2_dC14_constant = calc_mixing_ratio_C14C(def_co2_deltaC13,def_co2_deltaC14)
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO i=1,nc
            CO2_mixing_ratio_c14_ptr(i) = CO2_air(i) / (1._wp + 1._wp / co2_dC14_constant)
          END DO
        END IF
#endif
      END IF ! IF (model_scheme == MODEL_QUINCY) THEN
    END IF ! IF (PRESENT(CO2_air)) THEN

    !> lakes
    !>
    IF (tile_contains_lake) THEN
      IF (PRESENT(drag_wtr)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          drag_wtr_ptr(i) = drag_wtr(i)
        END DO
      END IF
      IF (PRESENT(drag_ice)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          drag_ice_ptr(i) = drag_ice(i)
        END DO
      END IF
      IF (PRESENT(t_acoef_wtr)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_acoef_wtr_ptr(i) = t_acoef_wtr(i)
        END DO
      END IF
      IF (PRESENT(t_bcoef_wtr)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_bcoef_wtr_ptr(i) = t_bcoef_wtr(i)
        END DO
      END IF
      IF (PRESENT(q_acoef_wtr)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          q_acoef_wtr_ptr(i) = q_acoef_wtr(i)
        END DO
      END IF
      IF (PRESENT(q_bcoef_wtr)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          q_bcoef_wtr_ptr(i) = q_bcoef_wtr(i)
        END DO
      END IF
      IF (PRESENT(t_acoef_ice)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_acoef_ice_ptr(i) = t_acoef_ice(i)
        END DO
      END IF
      IF (PRESENT(t_bcoef_ice)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_bcoef_ice_ptr(i) = t_bcoef_ice(i)
        END DO
      END IF
      IF (PRESENT(q_acoef_ice)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          q_acoef_ice_ptr(i) = q_acoef_ice(i)
        END DO
      END IF
      IF (PRESENT(q_bcoef_ice)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          q_bcoef_ice_ptr(i) = q_bcoef_ice(i)
        END DO
      END IF

    END IF

    !$ACC END PARALLEL

    IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__
      ! Update local time and the daytime counter
      CALL update_local_time_and_daytime_counter( &
        &     global_seconds_day, dtime, lon, swpar_srf_down, daytime_counter_ptr, daylength_prev_day_ptr, local_time_day_seconds_ptr)

      ! In case that we run with spin-up acceleration we need to update the accelerator bookkeeping
      IF (run_spinup_accelerator) THEN
        ! in most years the spin-up will not be accelerated
        slow_sb_pool_accelerator_execute_ptr(:) = 0.0_wp

        IF (is_newyear(mtime_current, dtime)) THEN
          CALL finish(TRIM(routine), 'This functionality has so far not been tested within a coupled run, but should be!')

          CALL update_slow_sb_pool_accelerator_bookkeeping( dtime, current_year, &
            & sb_pool_spinup_accelerator_max_executions, sb_pool_spinup_accelerator_frequency, sb_pool_spinup_accelerator_start_year, &
            & slow_sb_pool_accelerator_execution_counter_ptr, slow_sb_pool_accelerator_execute_ptr)
        END IF
      END IF
#endif
    END IF
  END SUBROUTINE update_atm2land

  SUBROUTINE update_land2atm(tile, options,                                          &
    & t_srf, t_srf_rad, t_eff_srf, qsat_srf, s_srf,                                  &
    & fact_q_air, fact_qsat_srf, evapopot,                                           &
    & evapotrans, latent_hflx, sensible_hflx, grnd_hflx, grnd_hcap,                  &
    & rough_h_srf, rough_m_srf, q_snocpymlt,                                         &
    & alb_vis_dir, alb_nir_dir, alb_vis_dif, alb_nir_dif,                            &
    & kh, km, kh_neutral, km_neutral, CO2_flux,                                      &
    & t_lwtr, t_lice, qsat_lwtr, qsat_lice, s_lwtr, s_lice,                          &
    & evapo_wtr, latent_hflx_wtr, sensible_hflx_wtr,                                 &
    & evapo_ice, latent_hflx_ice, sensible_hflx_ice,                                 &
    & ice_fract_lake,                                                                &
    & alb_vis_dir_wtr, alb_vis_dif_wtr, alb_nir_dir_wtr, alb_nir_dif_wtr,            &
    & albedo_lwtr, albedo_lice)

    USE mo_jsb_physical_constants, ONLY: molarMassCO2

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options
    REAL(wp), OPTIONAL, INTENT(out) ::                                               &
      & t_srf(:), t_srf_rad(:), t_eff_srf(:), qsat_srf(:), s_srf(:),                 &
      & fact_q_air(:), fact_qsat_srf(:), evapopot(:),                                &
      & evapotrans(:), latent_hflx(:), sensible_hflx(:), grnd_hflx(:), grnd_hcap(:), &
      & rough_h_srf(:), rough_m_srf(:), q_snocpymlt(:),                              &
      & t_lwtr(:), t_lice(:), qsat_lwtr(:), qsat_lice(:), s_lwtr(:), s_lice(:),      &
      & evapo_wtr(:), latent_hflx_wtr(:), sensible_hflx_wtr(:),                      &
      & evapo_ice(:), latent_hflx_ice(:), sensible_hflx_ice(:),                      &
      & ice_fract_lake(:),                                                           &
      & alb_vis_dir_wtr(:), alb_vis_dif_wtr(:), alb_nir_dir_wtr(:),                  &
      & alb_nir_dif_wtr(:), albedo_lwtr(:), albedo_lice(:),                          &
      & alb_vis_dir(:), alb_nir_dir(:), alb_vis_dif(:), alb_nir_dif(:),              &
      & CO2_flux(:), kh(:), km(:), kh_neutral(:), km_neutral(:)

    dsl4jsb_Def_config(SEB_)
    dsl4jsb_Def_memory(SEB_)
    dsl4jsb_Def_memory(TURB_)
    dsl4jsb_Def_memory(HYDRO_)
    dsl4jsb_Def_memory(RAD_)
    dsl4jsb_Def_memory(CARBON_)

#ifndef __NO_QUINCY__
    dsl4jsb_Def_memory(VEG_)
#endif

    TYPE(t_jsb_model), POINTER :: model
    CLASS(t_jsb_tile_abstract), POINTER :: land_tile
    dsl4jsb_Def_memory_tile(TURB_, land_tile)

    INTEGER  :: iblk, ics, ice, nc, i, j, it, nt
    LOGICAL  :: use_tmx, tile_is_carbon_active
    INTEGER  :: stream, original_stream
    INTEGER  :: model_scheme
    REAL(wp) :: fact_lake(options%nc), fact_land(options%nc)

    CHARACTER(len=*), PARAMETER :: routine = modname//':update_land2atm'

    dsl4jsb_Real2D_onChunk :: &
      & t_ptr, &                  ! tmx
      & t_filt_ptr, &
      & t_rad_ptr, &
      & t_eff_ptr, &
      & qsat_star_ptr, &
      & s_star_ptr, &
      & fact_q_air_ptr, &
      & fact_q_air_land_tile, &
      & fact_qsat_srf_ptr, &
      & fact_qsat_srf_land_tile, &
      & evapopot_ptr, &
      & evapotrans_ptr, &         ! tmx
      & latent_hflx_ptr, &        ! tmx
      & sensible_hflx_ptr, &      ! tmx
      & evapotrans_lnd_ptr, &
      & latent_hflx_lnd_ptr, &
      & sensible_hflx_lnd_ptr, &
      & forc_hflx_ptr, &
      & heat_cap_ptr, &
      & rough_h_ptr, &
      & rough_m_ptr, &
      & q_snocpymlt_ptr, &
      & alb_vis_lnd_ptr, &
      & alb_nir_lnd_ptr, &
      & kh_ptr, &
      & km_ptr, &
      & kh_neutral_ptr, &
      & km_neutral_ptr, &
      & co2flux_npp_2_atm_ta_ptr, &
      & co2flux_soilresp_2_atm_ta_ptr, &
      & co2flux_herb_2_atm_ta_ptr, &
      & co2flux_fire_all_2_atm_ta_ptr, &
      & t_lwtr_ptr, &
      & t_lice_ptr, &
      & qsat_lwtr_ptr, &
      & qsat_lice_ptr, &
      & s_lwtr_ptr, &
      & s_lice_ptr, &
      & evapo_wtr_ptr, &
      & latent_hflx_wtr_ptr, &
      & sensible_hflx_wtr_ptr, &
      & evapo_ice_ptr, &
      & latent_hflx_ice_ptr, &
      & fract_lice_ptr, &
      & albedo_lwtr_ptr, &
      & albedo_lice_ptr, &
      & sensible_hflx_ice_ptr, &
      & net_biosphere_production_ptr ! the JSBACH interface uses co2flux components (different unit&sign)

    ! avoid compiler warnings about dummy arguments not being used
    IF (PRESENT(alb_nir_dif_wtr)) CONTINUE
    IF (PRESENT(alb_nir_dir_wtr)) CONTINUE
    IF (PRESENT(alb_vis_dif_wtr)) CONTINUE
    IF (PRESENT(alb_vis_dir_wtr)) CONTINUE

    IF (ASSOCIATED(tile%parent)) CALL finish(TRIM(routine), 'Should only be called for the root tile')

    iblk = options%iblk
    ics  = options%ics
    ice  = options%ice
    nc   = options%nc

    IF (nc /= SIZE(t_srf,1)) CALL finish(TRIM(routine), 'Wrong dimensions')

    model => Get_model(tile%owner_model_id)
    model_scheme = model%config%model_scheme

    use_tmx = model%config%use_tmx
    IF (model_scheme == MODEL_QUINCY) THEN
      tile_is_carbon_active = .TRUE.
    ELSE
      tile_is_carbon_active = tile%Is_process_active(CARBON_)
    END IF

    dsl4jsb_Get_config(SEB_)
    dsl4jsb_Get_memory(SEB_)
    dsl4jsb_Get_memory(TURB_)
    dsl4jsb_Get_memory(HYDRO_)
    dsl4jsb_Get_memory(RAD_)

    IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__
      dsl4jsb_Get_memory(VEG_)
#endif
    ELSE IF (tile_is_carbon_active) THEN
      dsl4jsb_Get_memory(CARBON_)
    END IF

    !$ACC DATA CREATE(fact_land, fact_lake) ASYNC(acc_stream)

    IF (tile%contains_lake .AND. .NOT. use_tmx .AND. .NOT. jsbach_runs_standalone()) THEN

      ! With lakes, re-scale _lnd, _lice and _lwtr fluxes given back to the atmosphere so that they are relative to the
      ! jsbach grid box, not/only including lakes (JSBACH grid boxes do not include ocean fractions.). Re-scaling is
      ! necessary, as in the atmosphere fluxes are regarded as land-only (or lake-only) fluxes, and lake, land and ocean
      ! fractions are considered separately.
      ! Note that, if lakes are considered, the lake tile must be a direct child of the root tile.

      DO j=1,SIZE(tile%lcts)
        IF (tile%lcts(j)%id == LAKE_TYPE) EXIT         ! We get index j of the lake tile
      END DO

      !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream) PRESENT(tile%lcts)
      !$ACC LOOP GANG VECTOR
      DO i=1,nc
        fact_lake(i) = tile%lcts(j)%fract(ics+i-1,iblk)      ! Lake fraction
        fact_land(i) = 1._wp - fact_lake(i)            ! Land fraction
        IF (fact_lake(i) > 0._wp) THEN
          fact_lake(i) = 1._wp / fact_lake(i)          ! Lake factor: 1/lake_fraction
        ELSE
          fact_lake(i) = 1._wp
        END IF
        IF (fact_land(i) > 0._wp) THEN
          fact_land(i) = 1._wp / fact_land(i)          ! Land factor: 1/land_fraction
        ELSE
          fact_land(i) = 1._wp
        END IF
      END DO
      !$ACC END LOOP
      !$ACC END PARALLEL

    ELSE    ! without lakes

      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO i=1,nc
        fact_land(i) = 1._wp
        fact_lake(i) = 0._wp
      END DO
      !$ACC END PARALLEL LOOP

    END IF

    ! Exchange fields on the land tile
    ! --------------------------------

    ! For some variables the atmosphere expects values for the land tile (excluding lakes).
    ! Thus we define a pointer to the land tile.
    nt = tile%Get_no_of_children()
    DO it=1,nt
      IF (it == 1) THEN
        land_tile => tile%Get_first_child_tile()
      ELSE
        land_tile => land_tile%Get_next_sibling_tile()
      END IF
      ! Exit if the land tile was found
      IF (TRIM(land_tile%name) == 'land') EXIT
    END DO
    dsl4jsb_Get_memory_tile(TURB_, land_tile)

    t_eff_ptr        => dsl4jsb_var2D_onChunk(SEB_,   t_eff4)
    qsat_star_ptr    => dsl4jsb_var2D_onChunk(SEB_,   qsat_star)
    s_star_ptr       => dsl4jsb_var2D_onChunk(SEB_,   s_star)
    dsl4jsb_Get_var2d_onChunk_tile_name(TURB_, fact_q_air, land_tile)
    dsl4jsb_Get_var2d_onChunk_tile_name(TURB_, fact_qsat_srf, land_tile)
    evapopot_ptr     => dsl4jsb_var2D_onChunk(HYDRO_, evapopot) ! For offline only
    forc_hflx_ptr    => dsl4jsb_var2D_onChunk(SEB_,   forc_hflx)    ! TODO Not used
    heat_cap_ptr     => dsl4jsb_var2D_onChunk(SEB_,   heat_cap)
    rough_h_ptr      => dsl4jsb_var2D_onChunk(TURB_,  rough_h)
    rough_m_ptr      => dsl4jsb_var2D_onChunk(TURB_,  rough_m)
    q_snocpymlt_ptr  => dsl4jsb_var2D_onChunk(HYDRO_, q_snocpymlt)
    IF (use_tmx) THEN
      t_ptr                 => dsl4jsb_var2D_onChunk(SEB_,   t)
      t_rad_ptr             => dsl4jsb_var2D_onChunk(SEB_,   t_rad4)
      evapotrans_ptr        => dsl4jsb_var2D_onChunk(HYDRO_, evapotrans)
      latent_hflx_ptr       => dsl4jsb_var2D_onChunk(SEB_,   latent_hflx)
      sensible_hflx_ptr     => dsl4jsb_var2D_onChunk(SEB_,   sensible_hflx)
      alb_vis_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_vis)      ! TODO
      alb_vis_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_vis)      ! TODO
      alb_nir_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_nir)      ! TODO
      alb_nir_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_nir)
      kh_ptr                => dsl4jsb_var2D_onChunk(TURB_,  kh)
      km_ptr                => dsl4jsb_var2D_onChunk(TURB_,  km)
      kh_neutral_ptr        => dsl4jsb_var2D_onChunk(TURB_,  kh_neutral)
      km_neutral_ptr        => dsl4jsb_var2D_onChunk(TURB_,  km_neutral)
    ELSE
      t_filt_ptr            => dsl4jsb_var2D_onChunk(SEB_,   t_filt)
      evapotrans_lnd_ptr    => dsl4jsb_var2D_onChunk(HYDRO_, evapotrans_lnd)
      latent_hflx_lnd_ptr   => dsl4jsb_var2D_onChunk(SEB_,   latent_hflx_lnd)
      sensible_hflx_lnd_ptr => dsl4jsb_var2D_onChunk(SEB_,   sensible_hflx_lnd)
      alb_vis_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_vis_lnd)  ! TODO
      alb_vis_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_vis_lnd)  ! TODO
      alb_nir_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_nir_lnd)  ! TODO
      alb_nir_lnd_ptr       => dsl4jsb_var2D_onChunk(RAD_,   alb_nir_lnd)
    END IF

    IF (tile_is_carbon_active) THEN
      IF (model_scheme == MODEL_QUINCY) THEN
#ifndef __NO_QUINCY__
        ! Note: the co2flux components of JSBACH have a different unit and different sign
        net_biosphere_production_ptr => dsl4jsb_var2D_onChunk(VEG_, net_biosphere_production)
#endif
      ELSE
        co2flux_npp_2_atm_ta_ptr      => dsl4jsb_var2D_onChunk(CARBON_,co2flux_npp_2_atm_ta)
        co2flux_soilresp_2_atm_ta_ptr => dsl4jsb_var2D_onChunk(CARBON_,co2flux_soilresp_2_atm_ta)
        co2flux_herb_2_atm_ta_ptr     => dsl4jsb_var2D_onChunk(CARBON_,co2flux_herb_2_atm_ta)
        co2flux_fire_all_2_atm_ta_ptr => dsl4jsb_var2D_onChunk(CARBON_,co2flux_fire_all_2_atm_ta)
      END IF
    END IF

    IF (tile%contains_lake .AND. .NOT. jsbach_runs_standalone()) THEN
      t_lwtr_ptr            => dsl4jsb_var2D_onChunk(SEB_,   t_lwtr)
      qsat_lwtr_ptr         => dsl4jsb_var2D_onChunk(SEB_,   qsat_lwtr)
      s_lwtr_ptr            => dsl4jsb_var2D_onChunk(SEB_,   s_lwtr)
      evapo_wtr_ptr         => dsl4jsb_var2D_onChunk(HYDRO_, evapo_wtr)
      latent_hflx_wtr_ptr   => dsl4jsb_var2D_onChunk(SEB_,   latent_hflx_wtr)
      sensible_hflx_wtr_ptr => dsl4jsb_var2D_onChunk(SEB_,   sensible_hflx_wtr)
      albedo_lwtr_ptr       => dsl4jsb_var2D_onChunk(RAD_,   albedo_lwtr)
      fract_lice_ptr        => dsl4jsb_var2D_onChunk(SEB_,   fract_lice)
    END IF

    IF (tile%contains_lake .AND. .NOT. jsbach_runs_standalone() .AND. dsl4jsb_Config(SEB_)%l_ice_on_lakes) THEN
      t_lice_ptr            => dsl4jsb_var2D_onChunk(SEB_,   t_lice)
      qsat_lice_ptr         => dsl4jsb_var2D_onChunk(SEB_,   qsat_lice)
      s_lice_ptr            => dsl4jsb_var2D_onChunk(SEB_,   s_lice)
      evapo_ice_ptr         => dsl4jsb_var2D_onChunk(HYDRO_, evapo_ice)
      latent_hflx_ice_ptr   => dsl4jsb_var2D_onChunk(SEB_,   latent_hflx_ice)
      sensible_hflx_ice_ptr => dsl4jsb_var2D_onChunk(SEB_,   sensible_hflx_ice)
      albedo_lice_ptr       => dsl4jsb_var2D_onChunk(RAD_,   albedo_lice)
    END IF

    original_stream = acc_stream
    IF (model%use_multiple_queues) acc_stream = acc_stream + 1
    !$ACC WAIT(original_stream) ASYNC(acc_stream)

    !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

    IF (use_tmx) THEN
      IF (PRESENT(t_srf)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_srf(i) = t_ptr(i)
        END DO
      END IF
      IF (PRESENT(t_srf_rad)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_srf_rad(i) = t_rad_ptr(i)**0.25_wp
        END DO
      END IF
      IF (PRESENT(evapotrans)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapotrans(i) = evapotrans_ptr(i)
        END DO
      END IF
      IF (PRESENT(latent_hflx)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx(i) = latent_hflx_ptr(i)
        END DO
      END IF
      IF (PRESENT(sensible_hflx)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx(i) = sensible_hflx_ptr(i)
        END DO
      END IF
    ELSE
      IF (PRESENT(t_srf)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_srf(i) = t_filt_ptr(i)
        END DO
      END IF
      IF (PRESENT(evapotrans)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapotrans(i) = fact_land(i) * evapotrans_lnd_ptr(i)
        END DO
      END IF
      IF (PRESENT(latent_hflx)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx(i) = fact_land(i) * latent_hflx_lnd_ptr(i)
        END DO
      END IF
      IF (PRESENT(sensible_hflx)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx(i) = fact_land(i) * sensible_hflx_lnd_ptr(i)
        END DO
      END IF
    END IF
    IF (PRESENT(t_eff_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        t_eff_srf(i) = t_eff_ptr(i)**0.25_wp
      END DO
    END IF
    IF (PRESENT(qsat_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        qsat_srf(i) = qsat_star_ptr(i)
      END DO
    END IF
    IF (PRESENT(s_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        s_srf(i) = s_star_ptr(i)
      END DO
    END IF
    IF (PRESENT(fact_q_air)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        fact_q_air(i) = fact_q_air_land_tile(i)
      END DO
    END IF
    IF (PRESENT(fact_qsat_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        fact_qsat_srf(i) = fact_qsat_srf_land_tile(i)
      END DO
    END IF
    IF (PRESENT(evapopot)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        evapopot(i) = evapopot_ptr(i)
      END DO
    END IF
    IF (PRESENT(grnd_hflx)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        grnd_hflx(i) = forc_hflx_ptr(i)
      END DO
    END IF
    IF (PRESENT(grnd_hcap)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        grnd_hcap(i) = heat_cap_ptr(i)
      END DO
    END IF
    IF (PRESENT(rough_h_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        rough_h_srf(i) = rough_h_ptr(i)
      END DO
    END IF
    IF (PRESENT(rough_m_srf)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        rough_m_srf(i) = rough_m_ptr(i)
      END DO
    END IF
    IF (PRESENT(q_snocpymlt)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        q_snocpymlt(i) = fact_land(i) * q_snocpymlt_ptr(i)
      END DO
    END IF
    IF (PRESENT(alb_vis_dir)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        alb_vis_dir(i) = alb_vis_lnd_ptr(i)
      END DO
    END IF
    IF (PRESENT(alb_vis_dif)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        alb_vis_dif(i) = alb_vis_lnd_ptr(i)
      END DO
    END IF
    IF (PRESENT(alb_nir_dir)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        alb_nir_dir(i) = alb_nir_lnd_ptr(i)
      END DO
    END IF
    IF (PRESENT(alb_nir_dif)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        alb_nir_dif(i) = alb_nir_lnd_ptr(i)
      END DO
    END IF
    IF (PRESENT(CO2_flux)) THEN
      IF (tile_is_carbon_active) THEN
        IF (.NOT. model_scheme == MODEL_QUINCY) THEN
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO i=1,nc
            CO2_flux(i) = fact_land(i) *              &
              & (   co2flux_npp_2_atm_ta_ptr(i)       &
              &   + co2flux_soilresp_2_atm_ta_ptr(i)  &
              &   + co2flux_herb_2_atm_ta_ptr(i)      &
              &   + co2flux_fire_all_2_atm_ta_ptr(i)  &
              & )
          END DO
        ELSE
#ifndef __NO_QUINCY__
          !$ACC LOOP GANG(STATIC: 1) VECTOR
          DO i=1,nc
            ! micro-mol CO2 m-2 s-1 -> kg(CO2) m-2 s-1
            ! * -1.0_wp because the atmosphere expects CO2 sources as positive and sinks as negative values
            !   ... differs to the convention for fluxes in ICON which would be "fluxes point downwards"
            !   ... note: in jsbach the negation of the fluxes is already done in the carbon interface
            CO2_flux(i) = fact_land(i) * net_biosphere_production_ptr(i) * molarMassCO2 * 0.001_wp * 0.000001_wp * (-1.0_wp)
          END DO
#endif
        END IF
      ELSE
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          CO2_flux(i) = 0._wp
        END DO
      END IF
    END IF
    IF (PRESENT(kh)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        kh(i) = kh_ptr(i)
      END DO
    END IF
    IF (PRESENT(km)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        km(i) = km_ptr(i)
      END DO
    END IF
    IF (PRESENT(kh_neutral)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        kh_neutral(i) = kh_neutral_ptr(i)
      END DO
    END IF
    IF (PRESENT(km_neutral)) THEN
      !$ACC LOOP GANG(STATIC: 1) VECTOR
      DO i=1,nc
        km_neutral(i) = km_neutral_ptr(i)
      END DO
    END IF

    !$ACC END PARALLEL

    ! Exchange fields on lake water fractions
    ! ---------------------------------------

    IF (model%use_multiple_queues) acc_stream = acc_stream + 1
    !$ACC WAIT(original_stream) ASYNC(acc_stream)

    IF (tile%contains_lake .AND. .NOT. jsbach_runs_standalone()) THEN

      !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

      IF (PRESENT(t_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_lwtr(i) = t_lwtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(qsat_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          qsat_lwtr(i) = qsat_lwtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(s_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          s_lwtr(i) = s_lwtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(evapo_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapo_wtr(i) = fact_lake(i) * evapo_wtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(latent_hflx_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx_wtr(i) = fact_lake(i) * latent_hflx_wtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(sensible_hflx_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx_wtr(i) = fact_lake(i) * sensible_hflx_wtr_ptr(i)
        END DO
      END IF
      IF (PRESENT(albedo_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          albedo_lwtr(i) = albedo_lwtr_ptr(i)
        END DO
      END IF

      ! Exchange fields on the ice fraction of lakes
      ! --------------------------------------------

      IF (PRESENT(ice_fract_lake)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          ice_fract_lake(i) = fract_lice_ptr(i)
        END DO
      END IF

      !$ACC END PARALLEL

    ELSE

      !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

      IF (PRESENT(t_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_lwtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(qsat_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          qsat_lwtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(s_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          s_lwtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(evapo_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapo_wtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(latent_hflx_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx_wtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(sensible_hflx_wtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx_wtr(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(albedo_lwtr)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          albedo_lwtr(i) = 0.07_wp
        END DO
      END IF

    ! Exchange fields on the ice fraction of lakes
    ! --------------------------------------------

      IF (PRESENT(ice_fract_lake)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          ice_fract_lake(i) = 0._wp
        END DO
      END IF

      !$ACC END PARALLEL

    END IF

    IF (model%use_multiple_queues) acc_stream = acc_stream + 1
    !$ACC WAIT(original_stream) ASYNC(acc_stream)
    IF (tile%contains_lake .AND. .NOT. jsbach_runs_standalone() .AND. dsl4jsb_Config(SEB_)%l_ice_on_lakes) THEN

      !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

      IF (PRESENT(t_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_lice(i) = t_lice_ptr(i)
        END DO
      END IF
      IF (PRESENT(qsat_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          qsat_lice(i) = qsat_lice_ptr(i)
        END DO
      END IF
      IF (PRESENT(s_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          s_lice(i) = s_lice_ptr(i)
        END DO
      END IF
      IF (PRESENT(evapo_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapo_ice(i) = fact_lake(i) * evapo_ice_ptr(i)
        END DO
      END IF
      IF (PRESENT(latent_hflx_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx_ice(i) = fact_lake(i) * latent_hflx_ice_ptr(i)
        END DO
      END IF
      IF (PRESENT(sensible_hflx_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx_ice(i) = fact_lake(i) * sensible_hflx_ice_ptr(i)
        END DO
      END IF
      IF (PRESENT(albedo_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          albedo_lice(i) = albedo_lice_ptr(i)
        END DO
      END IF

      !$ACC END PARALLEL

    ELSE

      !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(acc_stream)

      IF (PRESENT(t_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          t_lice(i) = 273.15_wp
        END DO
      END IF
      IF (PRESENT(qsat_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          qsat_lice(i) = 0.0075_wp
        END DO
      END IF
      IF (PRESENT(s_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          s_lice(i) = 2.9E5_wp
        END DO
      END IF
      IF (PRESENT(evapo_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          evapo_ice(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(latent_hflx_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          latent_hflx_ice(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(sensible_hflx_ice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          sensible_hflx_ice(i) = 0._wp
        END DO
      END IF
      IF (PRESENT(albedo_lice)) THEN
        !$ACC LOOP GANG(STATIC: 1) VECTOR
        DO i=1,nc
          albedo_lice(i) = 0.55_wp
        END DO
      END IF

      !$ACC END PARALLEL

    END IF

    IF (model%use_multiple_queues) THEN
      DO stream = original_stream + 1, acc_stream
          !$ACC WAIT(stream) ASYNC(original_stream)
      ENDDO
      acc_stream = original_stream
    END IF

    !$ACC END DATA

  END SUBROUTINE update_land2atm

#endif
END MODULE mo_atmland_interface
