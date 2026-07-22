!> QUINCY sylviculture variables init
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
!>#### initialization of sylviculture memory variables using, e.g., ic & bc input files
!>
MODULE mo_q_syl_init
#ifndef __NO_QUINCY__

  USE mo_kind,                ONLY: wp
  USE mo_exception,           ONLY: message, finish
  USE mo_jsb_control,         ONLY: debug_on
  USE mo_jsb_class,           ONLY: Get_model
  USE mo_jsb_model_class,     ONLY: t_jsb_model
  USE mo_jsb_grid_class,      ONLY: t_jsb_grid
  USE mo_jsb_tile_class,      ONLY: t_jsb_tile_abstract

  USE mo_jsb_process_class,     ONLY: Q_SYL_, VEG_
  dsl4jsb_Use_memory(Q_SYL_)
  dsl4jsb_Use_config(Q_SYL_)
  dsl4jsb_Use_config(VEG_)

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: q_syl_init

#ifndef __QUINCY_STANDALONE__
  PUBLIC read_harvest_fraction
#endif

  CHARACTER(len=*), PARAMETER :: modname = 'mo_q_syl_init'

CONTAINS

  ! ======================================================================================================= !
  !> Run sylviculture init
  !>
  SUBROUTINE q_syl_init(tile)
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model), POINTER :: model
    CHARACTER(len=*), PARAMETER :: routine = modname//':q_syl_init'
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_memory(Q_SYL_)
    dsl4jsb_Def_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Real2D_onDomain :: fract_wood_to_slash
    ! ----------------------------------------------------------------------------------------------------- !
    model => Get_model(tile%owner_model_id)

    ! current variables of q_syl only need initialisation on the box tile
    IF (ASSOCIATED(tile%parent)) RETURN

#ifdef __QUINCY_STANDALONE__
    ! QS simulation currently do not have a tile hierarchy, so we are always on the box tile
    ! Because the runtime environment of QS requires to have Q_SYL always active,
    ! we need to return here if its not a forest tile
    IF (tile%lcts(1)%lib_id /= 0) THEN
      IF (.NOT. dsl4jsb_Lctlib_param(ForestFlag)) RETURN
    END IF
#endif

    ! ----------------------------------------------------------------------------------------------------- !
    IF (debug_on()) CALL message(TRIM(routine), 'Setting initial conditions of sh memory (quincy) for tile '// &
      &                          TRIM(tile%name))
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_memory(Q_SYL_)
    dsl4jsb_Get_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    IF (dsl4jsb_Config(VEG_)%l_use_product_pools) THEN
      dsl4jsb_Get_var2D_onDomain(Q_SYL_, fract_wood_to_slash)

      ! QS previously used a wood_extraction_rate of 0.8, thus, we use an initial default slash of 0.2 (= 1.0 - 0.8)
      fract_wood_to_slash(:,:) = 0.2_wp

#ifndef __QUINCY_STANDALONE__
      ! for IQ slash is overwritten with spatially varrying values from a bc file
      CALL q_syl_read_int_vars(tile)
#endif
    END IF

  END SUBROUTINE q_syl_init

#ifndef __QUINCY_STANDALONE__

  ! ====================================================================================================== !
  !
  !> Read variables (currently forest harvest slash) from bc file
  !>
  SUBROUTINE q_syl_read_int_vars(tile)
    USE mo_jsb_io_netcdf,             ONLY: t_input_file, jsb_netcdf_open_input
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model), POINTER :: model
    REAL(wp), POINTER ::  &
      & ptr_2D(:,:)
    TYPE(t_input_file) :: input_file

    CHARACTER(len=*), PARAMETER :: routine = modname//':q_syl_read_int_vars'
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_memory(Q_SYL_)
    dsl4jsb_Def_config(Q_SYL_)
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Real2D_onDomain :: fract_wood_to_slash
    ! ----------------------------------------------------------------------------------------------------- !
    model => get_model(tile%owner_model_id)

    dsl4jsb_Get_config(Q_SYL_)
    dsl4jsb_Get_memory(Q_SYL_)
    dsl4jsb_Get_var2D_onDomain(Q_SYL_, fract_wood_to_slash)

    IF (debug_on()) CALL message(TRIM(routine), 'Reading Q_SYL vars from ' &
        &                          //TRIM(dsl4jsb_Config(Q_SYL_)%bc_filename))

    input_file = jsb_netcdf_open_input(TRIM(dsl4jsb_Config(Q_SYL_)%bc_filename), model%grid_id)

    IF (debug_on()) CALL message(TRIM(routine), 'Reading slash faction for forest harvest ...')

    ptr_2D => input_file%Read_2d(variable_name='slash', fill_array = fract_wood_to_slash(:,:))

    CALL input_file%Close()

  END SUBROUTINE q_syl_read_int_vars

  ! ====================================================================================================== !
  !
  !> Reading of this years to be harvested forested grid-cell area from input file
  !
  SUBROUTINE read_harvest_fraction(model_id, current_datetime)
    USE mo_io_units,              ONLY: filename_max
    USE mo_jsb_time_iface,        ONLY: t_datetime
    USE mo_jsb_time,              ONLY: is_time_experiment_start, get_year
    USE mo_jsb_io_netcdf,         ONLY: t_input_file, jsb_netcdf_open_input
    USE mo_q_syl_config_class,    ONLY: CONST_HARV, REF_HARV, TRANS_HARV

    ! -------------------------------------------------------------------------------------------------- !
    INTEGER,                   INTENT(in) :: model_id
    TYPE(t_datetime), POINTER, INTENT(in) :: current_datetime
    ! -------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model),            POINTER :: model
    TYPE(t_jsb_grid),             POINTER :: hgrid
    CLASS(t_jsb_tile_abstract),   POINTER :: tile

    dsl4jsb_Def_config(Q_SYL_)
    dsl4jsb_Def_memory(Q_SYL_)

    dsl4jsb_Real2D_onDomain :: fract_forest_harvest_y_read
    ! -------------------------------------------------------------------------------------------------- !
    REAL(wp),           POINTER :: ptr_2D(:,:)  ! tmp pointer
    INTEGER                     :: current_year
    TYPE(t_input_file)          :: input_file
    CHARACTER(len=filename_max) :: filename_harvest_data
    CHARACTER(len=*), PARAMETER :: routine = modname//':read_harvest_fraction'
    ! -------------------------------------------------------------------------------------------------- !
    model => Get_model(model_id)
    CALL model%Get_top_tile(tile)

    dsl4jsb_Get_config(Q_SYL_)
    dsl4jsb_Get_memory(Q_SYL_)
    dsl4jsb_Get_var2D_onDomain(Q_SYL_, fract_forest_harvest_y_read)

    IF (dsl4jsb_Config(Q_SYL_)%harvest_scheme == CONST_HARV) THEN
      ! No reading necessary if constant harvest
      RETURN
    END IF

    IF (dsl4jsb_Config(Q_SYL_)%harvest_scheme == REF_HARV &
        & .AND. .NOT. is_time_experiment_start(current_datetime)) THEN
      ! In case of a reference harvest file, reading is only necessary at the beginning of the experiment
      RETURN
    END IF

    IF (debug_on()) CALL message( TRIM(routine), 'Starting routine')

    IF (dsl4jsb_Config(Q_SYL_)%harvest_scheme == REF_HARV) THEN
      ! search a file not ending on a year ...
      WRITE (filename_harvest_data,'(a,a)') &
        & TRIM(dsl4jsb_Config(Q_SYL_)%harvest_filename_prefix), ".nc"
    ELSE
      ! search a file ending on the current year
      current_year  = get_year(current_datetime)

      !>
      !> Assertion: routine currently expects filenames with 4 digits
      !>
      IF ( current_year > 9999) THEN
        CALL finish(TRIM(routine), 'Violation of assertion: this routine currently expects filenames with 4 digits.')
      ELSE IF (current_year < 1000) THEN
        CALL message(TRIM(routine), 'Warning: this routine currently expects filenames with 4 digits, '&
                                    //' for years < 1000 leading zeros are expected.')
      END IF
      WRITE (filename_harvest_data,'(a,a,I4.4,a)') &
        & TRIM(dsl4jsb_Config(Q_SYL_)%harvest_filename_prefix), '_', current_year, ".nc"
    END IF

    input_file = jsb_netcdf_open_input(TRIM(filename_harvest_data), model%grid_id)
    ptr_2D => input_file%Read_2d(variable_name='harvest_fract', fill_array = fract_forest_harvest_y_read(:,:))

    CALL input_file%Close()

    IF (debug_on()) CALL message(TRIM(routine), 'Finishing routine')

  END SUBROUTINE read_harvest_fraction

#endif
#endif
END MODULE mo_q_syl_init
