!> Contains the interfaces to the turbulence process
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

!NEC$ options "-finline-file=externals/jsbach/src/base/mo_jsb_control.pp-jsb.f90"
!NEC$ options "-finline-file=externals/jsbach/src/soil_snow_energy/mo_sse_process.pp-jsb.f90"
!NEC$ options "-finline-file=externals/jsbach/src/shared/mo_phy_schemes.pp-jsb.f90"

MODULE mo_turb_interface
#ifndef __NO_JSBACH__

  ! -------------------------------------------------------------------------------------------------------
  ! Used variables of module

  ! Use of basic structures
  USE mo_jsb_control,         ONLY: debug_on, jsbach_runs_standalone, acc_stream
  USE mo_kind,                ONLY: wp
  USE mo_jsb_math_constants,  ONLY: eps8
  USE mo_exception,           ONLY: message, finish

  USE mo_jsb_model_class,    ONLY: t_jsb_model, MODEL_QUINCY, MODEL_JSBACH
  USE mo_jsb_class,          ONLY: Get_model
  USE mo_jsb_tile_class,     ONLY: t_jsb_tile_abstract, t_jsb_aggregator
  USE mo_jsb_process_class,  ONLY: t_jsb_process
  USE mo_jsb_task_class,     ONLY: t_jsb_process_task, t_jsb_task_options

  ! Use of processes in this module
  dsl4jsb_Use_processes TURB_, A2L_, SEB_, HYDRO_, SSE_


  ! Use process configurations
  dsl4jsb_Use_config(TURB_)
  dsl4jsb_Use_config(HYDRO_)
  dsl4jsb_Use_config(SEB_)

  ! Use process memories
  dsl4jsb_Use_memory(A2L_)
  dsl4jsb_Use_memory(TURB_)
  dsl4jsb_Use_memory(SEB_)
  dsl4jsb_Use_memory(HYDRO_)

#ifndef __QUINCY_STANDALONE__
  ! Use of processes in this module
  dsl4jsb_Use_processes PHENO_, ASSIMI_

  ! Use process memories
  dsl4jsb_Use_memory(PHENO_)
  dsl4jsb_Use_memory(ASSIMI_)
#endif
#ifndef __NO_QUINCY__
  dsl4jsb_Use_processes VEG_, Q_ASSIMI_
  dsl4jsb_Use_memory(VEG_)
  dsl4jsb_Use_memory(Q_ASSIMI_)
#endif



  ! -------------------------------------------------------------------------------------------------------
  ! Module variables

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: Register_turb_tasks !,t_turb_process
  PUBLIC :: update_exchange_coefficients

#ifdef __QUINCY_STANDALONE__
  ! USEd by the mo_qs_model_interface
  PUBLIC :: update_humidity_scaling, update_roughness, aggregate_roughness
#endif

  CHARACTER(len=*), PARAMETER :: modname = 'mo_turb_interface'

  !> Type definition for exchange_coefficients
  TYPE, EXTENDS(t_jsb_process_task) :: tsk_exchange_coefficients
  CONTAINS
    PROCEDURE, NOPASS :: Integrate => update_exchange_coefficients    !< Advances task computation for one timestep
    PROCEDURE, NOPASS :: Aggregate => aggregate_exchange_coefficients !< Aggregates computed task variables
  END TYPE tsk_exchange_coefficients

  !> Constructor interface for exchange_coefficients task
  INTERFACE tsk_exchange_coefficients
    PROCEDURE Create_task_exchange_coefficients                        !< Constructor function for task
  END INTERFACE tsk_exchange_coefficients

  !> Type definition for surface_roughness task
  TYPE, EXTENDS(t_jsb_process_task) :: tsk_surface_roughness
  CONTAINS
    PROCEDURE, NOPASS :: Integrate => update_roughness     !< Advances task computation for one timestep
    PROCEDURE, NOPASS :: Aggregate => aggregate_roughness  !< Aggregates computed task variables
  END TYPE tsk_surface_roughness

  !> Constructor interface for surface_roughness task
  INTERFACE tsk_surface_roughness
    PROCEDURE Create_task_surface_roughness                        !< Constructor function for task
  END INTERFACE tsk_surface_roughness

  !> Type definition for humidity_scaling task
  TYPE, EXTENDS(t_jsb_process_task) :: tsk_humidity_scaling
  CONTAINS
    PROCEDURE, NOPASS :: Integrate => update_humidity_scaling     !< Advances task computation for one timestep
    PROCEDURE, NOPASS :: Aggregate => aggregate_humidity_scaling  !< Aggregates computed task variables
  END TYPE tsk_humidity_scaling

  !> Constructor interface for humidity scaling task
  INTERFACE tsk_humidity_scaling
    PROCEDURE Create_task_humidity_scaling                        !< Constructor function for task
  END INTERFACE tsk_humidity_scaling

CONTAINS

  ! ================================================================================================================================
  !! Constructors for tasks

  ! -------------------------------------------------------------------------------------------------------
  !> Constructor for exchange_coefficients task
  !!
  !! @param[in]     model_id     Model id
  !! @return        return_ptr   Instance of process task "tsk_exchange_coefficients"
  !!
  FUNCTION Create_task_exchange_coefficients(model_id) RESULT(return_ptr)

    INTEGER,                   INTENT(in) :: model_id
    CLASS(t_jsb_process_task), POINTER    :: return_ptr

    ALLOCATE(tsk_exchange_coefficients::return_ptr)
    CALL return_ptr%Construct(name='exchange_coefficients', process_id=TURB_, owner_model_id=model_id)

  END FUNCTION Create_task_exchange_coefficients

  ! -------------------------------------------------------------------------------------------------------
  !> Constructor for surface_roughness task
  !!
  !! @param[in]     model_id     Model id
  !! @return        return_ptr   Instance of process task "tsk_surface_roughness"
  !!
  FUNCTION Create_task_surface_roughness(model_id) RESULT(return_ptr)

    INTEGER,                   INTENT(in) :: model_id
    CLASS(t_jsb_process_task), POINTER    :: return_ptr

    ALLOCATE(tsk_surface_roughness::return_ptr)
    CALL return_ptr%Construct(name='surface_roughness', process_id=TURB_, owner_model_id=model_id)

  END FUNCTION Create_task_surface_roughness

  ! -------------------------------------------------------------------------------------------------------
  !> Constructor for humidity_scaling task
  !!
  !! @param[in]     model_id     Model id
  !! @return        return_ptr   Instance of process task "tsk_humidity_scaling"
  !!
  FUNCTION Create_task_humidity_scaling(model_id) RESULT(return_ptr)

    INTEGER,                   INTENT(in) :: model_id
    CLASS(t_jsb_process_task), POINTER    :: return_ptr

    ALLOCATE(tsk_humidity_scaling::return_ptr)
    CALL return_ptr%Construct(name='humidity_scaling', process_id=TURB_, owner_model_id=model_id)

  END FUNCTION Create_task_humidity_scaling

  ! -------------------------------------------------------------------------------------------------------
  !> Register tasks for turbulence process
  !!
  !! @param[in]     model_id  Model id
  !! @param[in,out] this      Instance of TURB process class
  !!
  SUBROUTINE Register_turb_tasks(this, model_id)

    CLASS(t_jsb_process), INTENT(inout) :: this
    INTEGER,              INTENT(in)    :: model_id

    CALL this%Register_task(tsk_exchange_coefficients(model_id))
    CALL this%Register_task(tsk_surface_roughness(model_id))
    CALL this%Register_task(tsk_humidity_scaling(model_id))

  END SUBROUTINE Register_turb_Tasks

#ifdef __QUINCY_STANDALONE__
  ! necessary to compile QUINCY-standalone
  SUBROUTINE update_exchange_coefficients(tile, options)
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options
  END SUBROUTINE update_exchange_coefficients
  SUBROUTINE aggregate_exchange_coefficients(tile, options)
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options
  END SUBROUTINE aggregate_exchange_coefficients
#else
  ! ======================================================================================================= !
  !>
  !> Implementation of "update" for task "exchange_coefficients" - only called with 'use_tmx = .TRUE.'
  !>
  SUBROUTINE update_exchange_coefficients(tile, options)

    USE mo_aes_thermo,  ONLY: potential_temperature ! Todo: use from mo_util_jsbach
    USE mo_phy_schemes, ONLY: qsat_water, qsat_ice, sfc_exchange_coefficients
      ! &                 exchange_coefficients, registered_exchange_coefficients_procedure

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_config(TURB_)
    dsl4jsb_Def_config(SEB_)

    ! Declare process memories
    dsl4jsb_Def_memory(TURB_)
    dsl4jsb_Def_memory(A2L_)
    dsl4jsb_Def_memory(HYDRO_)
    dsl4jsb_Def_memory(PHENO_)
    dsl4jsb_Def_memory(SEB_)

    ! Declare pointers to variables in memory
    dsl4jsb_Real2D_onChunk :: t_unfilt
    dsl4jsb_Real2D_onChunk :: dz_srf
    dsl4jsb_Real2D_onChunk :: press_srf
    dsl4jsb_Real2D_onChunk :: rho_srf
    dsl4jsb_Real2D_onChunk :: t_air
    dsl4jsb_Real2D_onChunk :: q_air
    dsl4jsb_Real2D_onChunk :: press_air
    dsl4jsb_Real2D_onChunk :: wind_air
    dsl4jsb_Real2D_onChunk :: rough_m
    dsl4jsb_Real2D_onChunk :: qsat_star
    dsl4jsb_Real2D_onChunk :: fract_lice
    dsl4jsb_Real2D_onChunk :: kh
    dsl4jsb_Real2D_onChunk :: km
    dsl4jsb_Real2D_onChunk :: kh_neutral
    dsl4jsb_Real2D_onChunk :: km_neutral
    dsl4jsb_Real2D_onChunk :: ch

    ! Local variables
    TYPE(t_jsb_model), POINTER :: model

    INTEGER  :: iblk, ics, ice, nc, ic
    REAL(wp), DIMENSION(options%nc) :: &
      & pot_temperature_srf, pot_temperature_air, qsat_srf
    REAL(wp) :: qsat_lice
    LOGICAL :: use_tmx

    CHARACTER(len=*), PARAMETER :: routine = modname//':update_exchange_coefficients'

    ! Get local variables from options argument
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc

    ! If process is not active on this tile, do nothing
    IF (.NOT. tile%Is_process_calculated(TURB_)) RETURN

    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    model => Get_model(tile%owner_model_id)

    use_tmx = model%config%use_tmx

    IF (.NOT. model%config%use_tmx) CALL finish(routine, 'can only be used with tmx')

    dsl4jsb_Get_config(TURB_)
    dsl4jsb_Get_config(SEB_)

    ! Get process memories
    dsl4jsb_Get_memory(TURB_)
    dsl4jsb_Get_memory(A2L_)
    dsl4jsb_Get_memory(SEB_)

    ! Get process variables
    dsl4jsb_Get_var2D_onChunk(A2L_,   dz_srf)          ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   press_srf)       ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   rho_srf)         ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   t_air)           ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   q_air)           ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   press_air)       ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   wind_air)        ! in
    dsl4jsb_Get_var2D_onChunk(SEB_,   t_unfilt)        ! in
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_m)         ! in
    dsl4jsb_Get_var2D_onChunk(TURB_,  kh)              ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  km)              ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  kh_neutral)      ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  km_neutral)      ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  ch)              ! out

    !$ACC DATA CREATE(pot_temperature_srf, pot_temperature_air, qsat_srf) ASYNC(acc_stream)

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      pot_temperature_srf(ic) = potential_temperature(t_unfilt(ic), press_srf(ic))
      pot_temperature_air(ic) = potential_temperature(t_air(ic), press_air(ic))
      qsat_srf(ic) = qsat_water(t_unfilt(ic), press_srf(ic), use_convect_tables=.NOT. use_tmx)
    END DO
    !$ACC END PARALLEL LOOP

    IF (tile%is_lake .AND. dsl4jsb_Config(SEB_)%l_ice_on_lakes) THEN
        dsl4jsb_Get_var2D_onChunk(SEB_, fract_lice)        ! in
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream) PRIVATE(qsat_lice)
        DO ic = 1, nc
          qsat_lice = qsat_ice(t_unfilt(ic), press_srf(ic), use_convect_tables=.NOT. use_tmx)
          qsat_srf(ic) = (1._wp - fract_lice(ic)) * qsat_srf(ic) + fract_lice(ic) * qsat_lice
        END DO
        !$ACC END PARALLEL LOOP
    END IF

    ! CALL exchange_coefficients( &
    !   & dz_srf(:),                                                 &
    !   & q_air(:), pot_temperature_air(:), wind_air(:), rough_m(:), pot_temperature_srf(:), qsat_srf(:), &
    !   & km(:), kh(:), km_neutral(:), kh_neutral(:)                                                      &
    !   &)
    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      ! CALL registered_exchange_coefficients_procedure( &
      CALL sfc_exchange_coefficients( &
        & dz_srf(ic), q_air(ic), pot_temperature_air(ic), wind_air(ic), rough_m(ic), pot_temperature_srf(ic), &
        & qsat_srf(ic), km(ic), kh(ic), km_neutral(ic), kh_neutral(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      IF (tile%fract(ics+ic-1,iblk) <= 0._wp) THEN
        km(ic) = 0._wp
        kh(ic) = 0._wp
      END IF
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      ch(ic) = wind_air(ic) * rho_srf(ic) * kh(ic)
    END DO
    !$ACC END PARALLEL LOOP
#if defined(_CRAYFTN) && _RELEASE_MAJOR <= 19
    !$ACC WAIT(acc_stream)
#endif
    !$ACC END DATA

  END SUBROUTINE update_exchange_coefficients

  ! -------------------------------------------------------------------------------------------------------
  !>
  !! Implementation of "aggregate" for task "exchange_coefficients"
  !!
  !! @param[in,out] tile    Tile for which routine is executed.
  !! @param[in]     options Additional run-time parameters.
  !!
  SUBROUTINE aggregate_exchange_coefficients(tile, options)

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_config(TURB_)
    dsl4jsb_Def_memory(TURB_)

    dsl4jsb_Real2D_onChunk :: rough_m
    dsl4jsb_Real2D_onChunk :: rough_h
    dsl4jsb_Real2D_onChunk :: rough_m_star
    dsl4jsb_Real2D_onChunk :: rough_h_star

    REAL(wp) :: config_blending_height, config_roughness_momentum_to_heat

    TYPE(t_jsb_model),       POINTER :: model
    CLASS(t_jsb_aggregator), POINTER :: weighted_by_fract

    CHARACTER(len=*), PARAMETER :: routine = modname//':aggregate_exchange_coefficients'

    ! Local variables
    INTEGER  :: iblk, ics, ice

    ! Get local variables from options argument
    iblk = options%iblk
    ics  = options%ics
    ice  = options%ice

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    weighted_by_fract => tile%Get_aggregator("weighted_by_fract")

    dsl4jsb_Get_memory(TURB_)

    CALL weighted_by_fract%BeginAggregate()

    dsl4jsb_Aggregate_onChunk(TURB_, kh,         weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, km,         weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, kh_neutral, weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, km_neutral, weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, ch,         weighted_by_fract)

    CALL weighted_by_fract%EndAggregate()

  END SUBROUTINE aggregate_exchange_coefficients
#endif

  ! ======================================================================================================= !
  !>
  !> Implementation of "update" for task "surface_roughness"
  !>
  !> the variables rough_m_star & rough_h_star are aggregated, and
  !> in the aggregate_roughness() rough_m and rough_h are calculated from these var
  !>
  SUBROUTINE update_roughness(tile, options)

    USE mo_jsb_lct_class,          ONLY: LAND_TYPE, VEG_TYPE

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_config(TURB_)
    dsl4jsb_Def_config(SEB_)

    ! Declare process memories
    dsl4jsb_Def_memory(TURB_)
    dsl4jsb_Def_memory(HYDRO_)
    dsl4jsb_Def_memory(SEB_)
#ifndef __QUINCY_STANDALONE__
    dsl4jsb_Def_memory(PHENO_)
#endif
#ifndef __NO_QUINCY__
    dsl4jsb_Def_memory(VEG_)
#endif

    ! Declare pointers to variables in memory
    dsl4jsb_Real2D_onChunk :: rough_m
    dsl4jsb_Real2D_onChunk :: rough_h
    dsl4jsb_Real2D_onChunk :: rough_m_star
    dsl4jsb_Real2D_onChunk :: rough_h_star
    dsl4jsb_Real2D_onChunk :: fract_snow_soil
    dsl4jsb_Real2D_onChunk :: fract_lice
    dsl4jsb_Real2D_onChunk :: lai
    dsl4jsb_Real2d_onChunk :: fract_fpc_max
    ! quincy
    dsl4jsb_Real2d_onChunk :: fract_fpc
    dsl4jsb_Real2d_onChunk :: rough_veg_star

    ! Local variables
    TYPE(t_jsb_model), POINTER :: model
    INTEGER  :: iblk, ics, ice, nc, ic
    INTEGER  :: model_scheme
    REAL(wp), DIMENSION(options%nc) :: &
      & rough_veg_star_local, rough_bare_star, rough_snow_star, fract_fol_proj_cov
    REAL(wp) :: &
      & config_rough_bare, config_rough_snow, config_rough_water, config_blending_height, config_roughness_lai_saturation, &
      & config_roughness_momentum_to_heat, lctlib_MinVegRoughness, lctlib_MaxVegRoughness, lctlib_VegRoughness
    LOGICAL :: lctlib_ForestFlag
    LOGICAL                         :: l_veg_bare_glac_tile      !< use for jsbach & quincy, as helper variable
    CHARACTER(len=*), PARAMETER :: routine = modname//':update_roughness'

    ! Get local variables from options argument
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc

    ! If process is not to be calculated on this tile, do nothing
    IF (.NOT. tile%Is_process_calculated(TURB_)) RETURN

    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    model => Get_model(tile%owner_model_id)
    model_scheme = model%config%model_scheme

    dsl4jsb_Get_config(TURB_)
    dsl4jsb_Get_config(SEB_)

    !$ACC DATA CREATE(fract_fol_proj_cov, rough_veg_star_local, rough_bare_star, rough_snow_star) ASYNC(acc_stream)

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      fract_fol_proj_cov(ic) = 0._wp
    END DO
    !$ACC END PARALLEL LOOP

    ! Get process memories
    dsl4jsb_Get_memory(TURB_)
    dsl4jsb_Get_memory(HYDRO_)
    SELECT CASE (model_scheme)
#ifndef __NO_QUINCY__
    CASE (MODEL_QUINCY)
      dsl4jsb_Get_memory(VEG_)
#endif
    END SELECT

    ! Get process variables
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_m)         ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_h)         ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_m_star)    ! out
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_h_star)    ! out

    config_blending_height = dsl4jsb_Config(TURB_)%blending_height
    config_rough_snow      = dsl4jsb_Config(TURB_)%roughness_snow
    config_rough_water     = dsl4jsb_Config(TURB_)%roughness_water


    ! set logical (vegetated, bare and glacier tiles) model specific to TRUE/FALSE
    l_veg_bare_glac_tile = .FALSE.
    SELECT CASE (model_scheme)
#ifndef __QUINCY_STANDALONE__
    CASE (MODEL_JSBACH)
      ! icon-land
      IF (tile%Has_process_memory(SSE_)) THEN  ! Should catch all vegetated, bare and glacier tiles
        l_veg_bare_glac_tile = .TRUE.
      END IF
#endif
#ifndef __NO_QUINCY__
    CASE (MODEL_QUINCY)
      ! quincy-standalone, and IQ w/o SSE_
      IF (tile%contains_vegetation .OR. tile%contains_bare .OR. tile%contains_land) THEN
        l_veg_bare_glac_tile = .TRUE.
      END IF
#endif
    END SELECT

    ! all vegetated, bare and glacier tiles
    IF (l_veg_bare_glac_tile) THEN

      dsl4jsb_Get_var2D_onChunk(HYDRO_, fract_snow_soil) ! in
      SELECT CASE (model_scheme)
#ifndef __NO_QUINCY__
      CASE (MODEL_QUINCY)
        dsl4jsb_Get_var2D_onChunk(VEG_,   rough_veg_star)          ! in
#endif
      END SELECT

      config_rough_bare                 = dsl4jsb_Config(TURB_)%roughness_bare
      config_roughness_momentum_to_heat = dsl4jsb_Config(TURB_)%roughness_momentum_to_heat

      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        rough_bare_star(ic) = 1._wp  / LOG(config_blending_height / config_rough_bare) ** 2
        rough_snow_star(ic) = 1._wp  / LOG(config_blending_height / config_rough_snow) ** 2
      END DO
      !$ACC END PARALLEL LOOP

      IF (tile%is_vegetation .AND. tile%lcts(1)%lib_id /= 0) THEN  ! vegetated PFT tile

        SELECT CASE (model_scheme)
#ifndef __QUINCY_STANDALONE__
        CASE (MODEL_JSBACH)
          dsl4jsb_Get_memory(PHENO_)
          dsl4jsb_Get_var2D_onChunk(PHENO_, fract_fpc_max)   ! in
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
          DO ic = 1, nc
            fract_fol_proj_cov(ic) = fract_fpc_max(ic)
          END DO
          !$ACC END PARALLEL LOOP
#endif
#ifndef __NO_QUINCY__
        CASE (MODEL_QUINCY)
          dsl4jsb_Get_memory(VEG_)
          dsl4jsb_Get_var2D_onChunk(VEG_, fract_fpc)         ! in
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
          DO ic = 1, nc
            fract_fol_proj_cov(ic) = fract_fpc(ic)
          END DO
          !$ACC END PARALLEL LOOP
#endif
        END SELECT

        IF (dsl4jsb_Config(TURB_)%l_roughness_lai) THEN

          config_roughness_lai_saturation  = dsl4jsb_Config(TURB_)%roughness_lai_saturation
          lctlib_MinVegRoughness           = dsl4jsb_Lctlib_param(MinVegRoughness)
          lctlib_MaxVegRoughness           = dsl4jsb_Lctlib_param(MaxVegRoughness)

          SELECT CASE (model_scheme)
#ifndef __QUINCY_STANDALONE__
          CASE (MODEL_JSBACH)
            dsl4jsb_Get_var2D_onChunk(PHENO_, lai)              ! in
            !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
            DO ic = 1, nc
              rough_veg_star_local(ic) =                                                                           &
                & 1._wp / LOG(config_blending_height                                                        &
                &             / (lctlib_MinVegRoughness + (lctlib_MaxVegRoughness - lctlib_MinVegRoughness) &
                &                                         * TANH(lai(ic) * config_roughness_lai_saturation)) &
                &            )**2
            END DO
            !$ACC END PARALLEL LOOP
#endif
#ifndef __NO_QUINCY__
          CASE (MODEL_QUINCY)
            !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
            DO ic = 1, nc
              ! calculated at PFT-tile level and aggregated to box tile (TURB_ could be calculated at box tile)
              rough_veg_star_local(ic) = rough_veg_star(ic)
            END DO
            !$ACC END PARALLEL LOOP
#endif
          END SELECT

        ELSE
#ifndef __NO_QUINCY__
          IF (model%config%model_scheme == MODEL_QUINCY) CALL finish(TRIM(routine), 'QUINCY: roughness calc. should depend on LAI')
#endif
          lctlib_VegRoughness = dsl4jsb_Lctlib_param(VegRoughness)
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
          DO ic = 1, nc
            rough_veg_star_local(ic) = 1._wp / LOG(config_blending_height / lctlib_VegRoughness) ** 2
          END DO
          !$ACC END PARALLEL LOOP
        ENDIF

        lctlib_ForestFlag = dsl4jsb_Lctlib_param(ForestFlag)
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_m_star(ic) = (1._wp - fract_fol_proj_cov(ic)) &
            &                * rough_bare_star(ic) + fract_fol_proj_cov(ic) * rough_veg_star_local(ic)

          ! Modify roughness length heat for non-vegetated part by snow cover
          ! TODO Note by TR from JSBACH3: lower z0 for snow covered surfaces should be discussed again,
          !      if a multi-layer snow model is implemented.
          rough_bare_star(ic) = (1._wp - fract_snow_soil(ic)) * rough_bare_star(ic) + fract_snow_soil(ic) * rough_snow_star(ic)
          ! Reduce roughness length of non-forest vegetation in case of snow
          IF (.NOT. lctlib_ForestFlag) THEN
            rough_veg_star_local(ic) = (1._wp - fract_snow_soil(ic)) &
              &                        * rough_veg_star_local(ic) + fract_snow_soil(ic) * rough_snow_star(ic)
          END IF

          rough_h_star(ic) = (1._wp - fract_fol_proj_cov(ic)) &
            &                * rough_bare_star(ic) + fract_fol_proj_cov(ic) * rough_veg_star_local(ic)
        END DO
        !$ACC END PARALLEL LOOP

      ELSE IF (tile%is_vegetation) THEN ! vegetated tile in a run without PFTs, e.g. jsbach_lite
#ifndef __NO_QUINCY__
        IF (model%config%model_scheme == MODEL_QUINCY) CALL finish(TRIM(routine), 'QUINCY: always running with PFT.')
#endif
        ! rough_m and rough_m_star are taken from values from init
        !
        ! We don't have PFTs so use rough_m from ini file
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_h_star(ic) = (1._wp - fract_snow_soil(ic)) * 1._wp / LOG(config_blending_height / MIN(1._wp, rough_m(ic)))**2 &
            &               +        fract_snow_soil(ic)  * rough_snow_star(ic)
        END DO
        !$ACC END PARALLEL LOOP

      ELSE  ! bare or glacier tile
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_m_star(ic) = rough_bare_star(ic)
          ! Modify roughness length heat for bare part by snow cover
          rough_h_star(ic) = (1._wp - fract_snow_soil(ic)) * rough_bare_star(ic) + fract_snow_soil(ic) * rough_snow_star(ic)
        END DO
        !$ACC END PARALLEL LOOP

      END IF  ! vegetated or bare tile

      IF (tile%lcts(1)%lib_id /= 0 .OR. .NOT. tile%is_vegetation) THEN ! PFT or bare/glacier
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_m(ic) = config_blending_height * EXP(-1._wp / SQRT(rough_m_star(ic)))
        END DO
        !$ACC END PARALLEL LOOP
      END IF

      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        rough_h(ic) = config_blending_height * EXP(-1._wp / SQRT(rough_h_star(ic))) / config_roughness_momentum_to_heat
      END DO
      !$ACC END PARALLEL LOOP

    ELSE IF (tile%contains_lake) THEN

      IF (dsl4jsb_Config(SEB_)%l_ice_on_lakes) THEN
        dsl4jsb_Get_memory(SEB_)
        dsl4jsb_Get_var2D_onChunk(SEB_, fract_lice) ! IN

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_m_star(ic) =            fract_lice(ic)  / (LOG(config_blending_height / config_rough_snow)  ** 2)  &
            &               + (1._wp - fract_lice(ic)) / (LOG(config_blending_height / config_rough_water) ** 2)
        END DO
        !$ACC END PARALLEL LOOP
      ELSE
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          rough_m_star(ic) = 1._wp / (LOG(config_blending_height / config_rough_water) ** 2)
        END DO
        !$ACC END PARALLEL LOOP
      END IF
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        rough_h_star(ic) = rough_m_star(ic)
        rough_m(ic) = config_blending_height * EXP(-1._wp / SQRT(rough_m_star(ic)))
        rough_h(ic) = rough_m(ic)
      END DO
      !$ACC END PARALLEL LOOP

    ELSE
      CALL finish(TRIM(routine), 'Don"t know what to do.')
    END IF

    ! Limit roughness length for heat by 1 m (see below Eq.5.6 in ECHAM5 manual)
    ! $noACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    ! DO ic = 1, nc
    !   rough_h(ic) = MIN(1._wp, blending_height * EXP(-1._wp / SQRT(rough_h(ic))))
    ! END DO
    ! $noACC END PARALLEL LOOP

    !$ACC END DATA

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Finished.')

  END SUBROUTINE update_roughness

  ! -------------------------------------------------------------------------------------------------------
  !>
  !! Implementation of "aggregate" for task "surface_roughness"
  !!
  !! @param[in,out] tile    Tile for which routine is executed.
  !! @param[in]     options Additional run-time parameters.
  !!
  SUBROUTINE aggregate_roughness(tile, options)

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_config(TURB_)
    dsl4jsb_Def_memory(TURB_)

    dsl4jsb_Real2D_onChunk :: rough_m
    dsl4jsb_Real2D_onChunk :: rough_h
    dsl4jsb_Real2D_onChunk :: rough_m_star
    dsl4jsb_Real2D_onChunk :: rough_h_star

    REAL(wp) :: config_blending_height, config_roughness_momentum_to_heat

    TYPE(t_jsb_model),       POINTER :: model
    CLASS(t_jsb_aggregator), POINTER :: weighted_by_fract

    CHARACTER(len=*), PARAMETER :: routine = modname//':aggregate_roughness'

    ! Local variables
    INTEGER  :: iblk, ics, ice, nc, ic

    ! Get local variables from options argument
    iblk = options%iblk
    ics  = options%ics
    ice  = options%ice
    nc   = options%nc

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    weighted_by_fract => tile%Get_aggregator("weighted_by_fract")

    dsl4jsb_Get_memory(TURB_)

    CALL weighted_by_fract%BeginAggregate()

    dsl4jsb_Aggregate_onChunk(TURB_, rough_m_star,   weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, rough_h_star,   weighted_by_fract)

    CALL weighted_by_fract%EndAggregate()

    model => Get_model(tile%owner_model_id)

    dsl4jsb_Get_config(TURB_)

    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_m)
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_h)
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_m_star)
    dsl4jsb_Get_var2D_onChunk(TURB_,  rough_h_star)

    config_blending_height            = dsl4jsb_Config(TURB_)%blending_height
    config_roughness_momentum_to_heat = dsl4jsb_Config(TURB_)%roughness_momentum_to_heat

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      rough_m(ic) = config_blending_height * EXP(-1._wp / SQRT(rough_m_star(ic)))
      rough_h(ic) = config_blending_height * EXP(-1._wp / SQRT(rough_h_star(ic))) / config_roughness_momentum_to_heat
    END DO
    !$ACC END PARALLEL LOOP

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Finished.')

  END SUBROUTINE aggregate_roughness

  ! ================================================================================================================================
  !>
  !> Implementation of "update" for task "humidity_scaling"
  !! Task "<PROCESS_NAME_LOWER_CASE>" <EXPLANATIION_OF_PROCESS>.
  !!
  !! @param[in,out] tile    Tile for which routine is executed.
  !! @param[in]     options Additional run-time parameters.
  !!
  SUBROUTINE update_humidity_scaling(tile, options)

    !USE mo_jsb_physical_constants, ONLY: rhoh2o
    USE mo_sse_process,            ONLY: relative_humidity_soil

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_config(HYDRO_)

    ! Declare pointers to process configs and memory
    dsl4jsb_Def_memory(TURB_)
    dsl4jsb_Def_memory(A2L_)
    dsl4jsb_Def_memory(SEB_)
    dsl4jsb_Def_memory(HYDRO_)
#ifndef __QUINCY_STANDALONE__
    dsl4jsb_Def_memory(PHENO_)
    dsl4jsb_Def_memory(ASSIMI_)
#endif
#ifndef __NO_QUINCY__
    dsl4jsb_Def_memory(VEG_)
    dsl4jsb_Def_memory(Q_ASSIMI_)
#endif

    ! Declare pointers to variables in memory
    dsl4jsb_Real2D_onChunk ::   &
      & fract_fpc,              &
      & fract_snow,             &
      & fract_wet,              &
      & wtr_rootzone_avail_max, &
      & wtr_rootzone_avail,     &
      & fact_q_air,             &
      & fact_qsat_srf,          &
      & fact_qsat_trans_srf,    &
      & qsat_star,              &
      & canopy_cond_limited,    &
      & pch,                    &
      & pch_stdalone,           &
      & kh,                     &
      & wind_air,               &
      & q_air

    dsl4jsb_Real2D_onChunk ::   &
      & canopy_cond,            &
      & q_sat_srf
    dsl4jsb_Real3D_onChunk ::   &
      & wtr_soil_fc_sl,         &
      & wtr_soil_pwp_sl

    dsl4jsb_Real3D_onChunk :: &
      & wtr_soil_sl,          &
      & soil_depth_sl,        &
      & vol_porosity_sl,      &
      & vol_wres_sl

    ! Local variables
    REAL(wp) ::                    &
      fact_qsat_veg(options%nc),   &
      fact_qsat_trans(options%nc), &
      fact_qsat_soil(options%nc),  &
      fact_qair_veg(options%nc),   &
      fact_qair_soil(options%nc),  &
      zfract_fpc(options%nc),      &
      w_soil_wilt_fract_config
    REAL(wp) ::                    &
      rel_hum,                     &
      weq_wsat_sl1,                &
      weq_wres_sl1
    REAL(wp) ::                    &
      ch_tmp(options%nc)                      !< local variable for pch / kh from jsbach, quincy, A2L_, TURB_
    LOGICAL :: &
      mask(options%nc)
    REAL(wp), ALLOCATABLE      :: canopy_conductance(:)    !< use for jsbach & quincy, as helper variable
    LOGICAL                    :: jsb_standalone           !< model runs standalone?

    INTEGER :: iblk, ics, ice, nc, ic
    INTEGER :: model_scheme

    TYPE(t_jsb_model), POINTER :: model

    CHARACTER(len=*), PARAMETER :: routine = modname//':update_humidity_scaling'

    ! Get local variables from options argument
    iblk = options%iblk
    ics  = options%ics
    ice  = options%ice
    nc   = options%nc

    IF (.NOT. tile%Is_process_calculated(TURB_)) RETURN

    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    model => Get_model(tile%owner_model_id)
    model_scheme = model%config%model_scheme
    jsb_standalone = jsbach_runs_standalone()

    dsl4jsb_Get_config(HYDRO_)

    !$ACC DATA ASYNC(acc_stream) &
    !$ACC   CREATE(fact_qsat_veg, fact_qsat_trans, fact_qsat_soil) &
    !$ACC   CREATE(fact_qair_veg, fact_qair_soil, zfract_fpc, mask, ch_tmp)

    ! Get pointers to process configs and memory
    dsl4jsb_Get_memory(A2L_)
    dsl4jsb_Get_memory(TURB_)
    dsl4jsb_Get_memory(SEB_)
    dsl4jsb_Get_memory(HYDRO_)
    SELECT CASE (model_scheme)
#ifndef __QUINCY_STANDALONE__
    CASE (MODEL_JSBACH)
      IF (tile%contains_vegetation) THEN
        dsl4jsb_Get_memory(PHENO_)
        IF (tile%Is_process_active(ASSIMI_)) THEN
          dsl4jsb_Get_memory(ASSIMI_)
        END IF
      END IF
#endif
#ifndef __NO_QUINCY__
    CASE (MODEL_QUINCY)
      dsl4jsb_Get_memory(VEG_)
      dsl4jsb_Get_memory(Q_ASSIMI_)
#endif
    END SELECT

    ! Set pointers to variables in memory
    IF (model%config%use_tmx) THEN
      dsl4jsb_Get_var2D_onChunk(TURB_, kh)                 ! in
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        ch_tmp(ic) = kh(ic)
      END DO
      !$ACC END PARALLEL LOOP
    ELSE
      SELECT CASE (model_scheme)
      CASE (MODEL_JSBACH)
        dsl4jsb_Get_var2D_onChunk(A2L_,   pch)               ! in
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          ch_tmp(ic) = pch(ic)
        END DO
        !$ACC END PARALLEL LOOP
      CASE (MODEL_QUINCY)
        IF (jsb_standalone) THEN
          dsl4jsb_Get_var2D_onChunk(TURB_,   pch_stdalone)     ! in
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
          DO ic = 1, nc
            ch_tmp(ic) = pch_stdalone(ic)
          END DO
          !$ACC END PARALLEL LOOP
        ELSE
          dsl4jsb_Get_var2D_onChunk(A2L_,   pch)               ! in
          !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
          DO ic = 1, nc
            ch_tmp(ic) = pch(ic)
          END DO
          !$ACC END PARALLEL LOOP
        END IF
      END SELECT
    END IF
    dsl4jsb_Get_var2D_onChunk(A2L_,   wind_air)            ! in
    dsl4jsb_Get_var2D_onChunk(A2L_,   q_air)               ! in
    dsl4jsb_Get_var2D_onChunk(HYDRO_, fract_snow)          ! in
    dsl4jsb_Get_var2D_onChunk(TURB_,  fact_q_air)          ! OUT
    dsl4jsb_Get_var2D_onChunk(TURB_,  fact_qsat_srf)       ! OUT
    dsl4jsb_Get_var2D_onChunk(TURB_,  fact_qsat_trans_srf) ! OUT
    IF (tile%contains_soil) THEN

      dsl4jsb_Get_var2D_onChunk(SEB_,    qsat_star)                ! in
      dsl4jsb_Get_var2D_onChunk(HYDRO_,  fract_wet)                ! in
      dsl4jsb_Get_var3D_onChunk(HYDRO_,  wtr_soil_sl)              ! in
      SELECT CASE (model_scheme)
      CASE (MODEL_JSBACH)
        dsl4jsb_Get_var2D_onChunk(HYDRO_,  wtr_rootzone_avail_max) ! in
        dsl4jsb_Get_var2D_onChunk(HYDRO_,  wtr_rootzone_avail)     ! in
        dsl4jsb_Get_var3D_onChunk(HYDRO_,  soil_depth_sl)          ! in
        dsl4jsb_Get_var3D_onChunk(HYDRO_,  vol_porosity_sl)        ! in
        dsl4jsb_Get_var3D_onChunk(HYDRO_,  vol_wres_sl)            ! in
      CASE (MODEL_QUINCY)
        dsl4jsb_Get_var3D_onChunk(HYDRO_,  wtr_soil_fc_sl)         ! in
        dsl4jsb_Get_var3D_onChunk(HYDRO_,  wtr_soil_pwp_sl)        ! in
      END SELECT
    END IF

    ! ----------------------------------------------------------------------------------------------------- !

    IF (tile%contains_vegetation) THEN
      ALLOCATE(canopy_conductance(nc))
      !$ACC ENTER DATA CREATE(canopy_conductance) ASYNC(acc_stream)
      SELECT CASE (model_scheme)
#ifndef __QUINCY_STANDALONE__
      CASE (MODEL_JSBACH)
        dsl4jsb_Get_var2D_onChunk(PHENO_,    fract_fpc)           ! in
        IF (tile%Is_process_active(ASSIMI_)) THEN
          dsl4jsb_Get_var2D_onChunk(ASSIMI_,   canopy_cond_limited)         ! in
        ELSE
          dsl4jsb_Get_var2D_onChunk(HYDRO_,    canopy_cond_limited)         ! in
        END IF
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1,nc
          canopy_conductance(ic) = canopy_cond_limited(ic)
        END DO
        !$ACC END PARALLEL LOOP
        w_soil_wilt_fract_config = dsl4jsb_Config(HYDRO_)%w_soil_wilt_fract

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          mask(ic) = wtr_rootzone_avail(ic) > w_soil_wilt_fract_config * wtr_rootzone_avail_max(ic)
        END DO
        !$ACC END PARALLEL LOOP
#endif
#ifndef __NO_QUINCY__
      CASE (MODEL_QUINCY)
        dsl4jsb_Get_var2D_onChunk(VEG_,      fract_fpc)                     ! in
        dsl4jsb_Get_var2D_onChunk(Q_ASSIMI_, canopy_cond)                   ! in
        dsl4jsb_Get_var2D_onChunk(SEB_,      q_sat_srf)                     ! in
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1,nc
          canopy_conductance(ic) = canopy_cond(ic)
        END DO
        !$ACC END PARALLEL LOOP

        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
        DO ic = 1, nc
          mask(ic) = canopy_conductance(ic) > eps8 .AND. (q_sat_srf(ic) - q_air(ic)) > 0.0_wp
        END DO
        !$ACC END PARALLEL LOOP
#endif
      END SELECT

      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        IF (mask(ic)) THEN
          fact_qsat_veg(ic) = fract_snow(ic) + (1._wp - fract_snow(ic)) *  &
            & (fract_wet(ic) + ( 1._wp - fract_wet(ic)) /                  &
            & (1._wp + ch_tmp(ic) * MAX(1._wp, wind_air(ic)) / MAX(1.E-20_wp, canopy_conductance(ic))))
          fact_qsat_trans(ic) = (1._wp - fract_snow(ic)) *  &
            & (1._wp - fract_wet(ic)) /                     &
            & (1._wp + ch_tmp(ic) * MAX(1._wp, wind_air(ic)) / MAX(1.E-20_wp, canopy_conductance(ic)))
        ELSE
          fact_qsat_veg(ic)   = fract_snow(ic) + (1._wp - fract_snow(ic)) * fract_wet(ic)
          fact_qsat_trans(ic) = 0._wp
        END IF
      END DO
      !$ACC END PARALLEL LOOP

      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        fact_qair_veg(ic) = fact_qsat_veg(ic)
        zfract_fpc(ic)    = fract_fpc(ic)
      END DO
      !$ACC END PARALLEL LOOP
      !$ACC EXIT DATA DELETE(canopy_conductance) ASYNC(acc_stream)
      DEALLOCATE(canopy_conductance)
    ELSE
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        fact_qsat_veg(ic)   = 0._wp
        fact_qsat_trans(ic) = 0._wp
        fact_qair_veg(ic)   = 0._wp
        zfract_fpc(ic)      = 0._wp
      END DO
      !$ACC END PARALLEL LOOP
    END IF

    IF (tile%contains_soil) THEN
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream) PRIVATE(rel_hum)
      DO ic = 1, nc
        SELECT CASE (model_scheme)
        CASE (MODEL_JSBACH)
          weq_wres_sl1 = vol_wres_sl(ic,1) * soil_depth_sl(ic,1)
          weq_wsat_sl1 = vol_porosity_sl(ic,1) * soil_depth_sl(ic,1)
          rel_hum      = relative_humidity_soil(wtr_soil_sl(ic,1), weq_wres_sl1, weq_wsat_sl1)
        CASE (MODEL_QUINCY)
          rel_hum = MIN(1.0_wp, (wtr_soil_sl(ic,1) - wtr_soil_pwp_sl(ic,1)) / (wtr_soil_fc_sl(ic,1) - wtr_soil_pwp_sl(ic,1)))
        END SELECT
        IF (qsat_star(ic) * rel_hum > q_air(ic) .AND. rel_hum > 1.e-10_wp) THEN
          fact_qsat_soil(ic) = fract_snow(ic) + (1._wp - fract_snow(ic)) * &
            & (fract_wet(ic) + (1._wp - fract_wet(ic)) * rel_hum)
          fact_qair_soil(ic) = 1._wp
        ELSE
          fact_qsat_soil(ic) = fract_snow(ic) + (1._wp - fract_snow(ic)) * fract_wet(ic)
          fact_qair_soil(ic) = fact_qsat_soil(ic)
        END IF
        IF (q_air(ic) > qsat_star(ic)) THEN
          fact_qsat_soil(ic) = 1._wp
          fact_qair_soil(ic) = 1._wp
        END IF
      END DO
      !$ACC END PARALLEL LOOP

    ELSE IF (tile%contains_glacier .OR. tile%is_lake) THEN
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        fact_qsat_soil(ic) = 1._wp
        fact_qair_soil(ic) = 1._wp
      END DO
      !$ACC END PARALLEL LOOP
    ELSE
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        fact_qsat_soil(ic) = 0._wp
        fact_qair_soil(ic) = 0._wp
      END DO
      !$ACC END PARALLEL LOOP
    END IF

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
    DO ic = 1, nc
      fact_qsat_srf(ic) = zfract_fpc(ic) * fact_qsat_veg(ic) + (1._wp - zfract_fpc(ic)) * fact_qsat_soil(ic)
      fact_q_air(ic)    = zfract_fpc(ic) * fact_qair_veg(ic) + (1._wp - zfract_fpc(ic)) * fact_qair_soil(ic)
      fact_qsat_trans_srf(ic) = zfract_fpc(ic) * fact_qsat_trans(ic)
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC END DATA

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Finished.')

  END SUBROUTINE update_humidity_scaling

  ! -------------------------------------------------------------------------------------------------------
  !>
  !! Implementation of "aggregate" for task "humidity_scaling"
  !!
  !! @param[in,out] tile    Tile for which routine is executed.
  !! @param[in]     options Additional run-time parameters.
  !!
  SUBROUTINE aggregate_humidity_scaling(tile, options)

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options

    dsl4jsb_Def_memory(TURB_)

    CLASS(t_jsb_aggregator), POINTER :: weighted_by_fract

    CHARACTER(len=*), PARAMETER :: routine = modname//':aggregate_humidity_scaling'

    ! Local variables
    INTEGER  :: iblk, ics, ice

    ! Get local variables from options argument
    iblk = options%iblk
    ics  = options%ics
    ice  = options%ice

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')

    dsl4jsb_Get_memory(TURB_)

    weighted_by_fract => tile%Get_aggregator("weighted_by_fract")

    CALL weighted_by_fract%BeginAggregate()

    dsl4jsb_Aggregate_onChunk(TURB_, fact_q_air,          weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, fact_qsat_srf,       weighted_by_fract)
    dsl4jsb_Aggregate_onChunk(TURB_, fact_qsat_trans_srf, weighted_by_fract)

    CALL weighted_by_fract%EndAggregate()

    IF (debug_on() .AND. iblk==1) CALL message(TRIM(routine), 'Finished.')

  END SUBROUTINE aggregate_humidity_scaling

  ! ================================================================================================================================
  !>
  !> Another task for "<PROCESS_NAME_LOWER_CASE>"
  !!
  !! @param[in,out] tile    Tile for which routine is executed.
  !! @param[in]     options Additional run-time parameters.
  !  ....

#endif
END MODULE mo_turb_interface
