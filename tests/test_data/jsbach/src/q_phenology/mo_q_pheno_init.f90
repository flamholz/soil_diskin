!> QUINCY phenology variables init
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
!>#### initialization of phenology memory variables using, e.g., ic & bc input files
!>
MODULE mo_q_pheno_init
#ifndef __NO_QUINCY__

  USE mo_kind,                ONLY: wp
  USE mo_exception,           ONLY: message, finish
  USE mo_jsb_control,         ONLY: debug_on

  USE mo_jsb_model_class,     ONLY: t_jsb_model
  USE mo_jsb_grid_class,      ONLY: t_jsb_grid
  USE mo_jsb_grid,            ONLY: Get_grid
  USE mo_jsb_class,           ONLY: get_model
  USE mo_jsb_tile_class,      ONLY: t_jsb_tile_abstract

  dsl4jsb_Use_processes Q_PHENO_
  dsl4jsb_Use_config(Q_PHENO_)
  dsl4jsb_Use_memory(Q_PHENO_)

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: q_pheno_init

  CHARACTER(len=*), PARAMETER :: modname = 'mo_q_pheno_init'

CONTAINS

  ! ======================================================================================================= !
  !> run quincy phenology init
  !>
  !>
  SUBROUTINE q_pheno_init(tile)

    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile

    TYPE(t_jsb_model), POINTER :: model
    CHARACTER(len=*), PARAMETER :: routine = modname//':q_pheno_init'

    model => Get_model(tile%owner_model_id)

    ! init boundary conditions
    CALL q_pheno_init_bc(tile)
  END SUBROUTINE q_pheno_init

  ! ======================================================================================================= !
  !> Intialize pheno variables for boundary conditions (bc)
  !>
  !>
  SUBROUTINE q_pheno_init_bc(tile)

    ! see also the USE statements in the module header
    USE mo_q_pheno_constants,   ONLY: ievergreen
    USE mo_jsb_impl_constants,  ONLY: false, true
    USE mo_jsb_grid_class,      ONLY: t_jsb_grid
    USE mo_jsb_grid,            ONLY: Get_grid
    ! ---------------------------
    ! 0.1 InOut
    CLASS(t_jsb_tile_abstract), INTENT(inout)     :: tile         !< one tile with data structure for one lct
    ! ---------------------------
    ! 0.2 Local
    TYPE(t_jsb_model), POINTER  :: model
    TYPE(t_jsb_grid),  POINTER  :: grid
    INTEGER                     :: nblks, ib, nproma, ic
    REAL(wp)                    :: config_lai_max, lctlib_phenology_type, lctlib_lai_max
    CHARACTER(len=*), PARAMETER :: routine = modname//':q_pheno_init_bc'
    ! ---------------------------
    ! 0.3 Declare Memory
    ! Declare process configuration and memory Pointers
    dsl4jsb_Def_config(Q_PHENO_)
    dsl4jsb_Def_memory(Q_PHENO_)
    ! Declare pointers to variables in memory
    dsl4jsb_Real2D_onDomain    :: growing_season
    dsl4jsb_Real2D_onDomain    :: lai_max
    ! ---------------------------
    ! 0.4
    IF (.NOT. tile%Is_process_active(Q_PHENO_)) RETURN
    IF (debug_on()) CALL message(routine, 'Setting boundary conditions of pheno memory (quincy) for tile '// &
      &                          TRIM(tile%name))
    ! ---------------------------
    ! 0.5 Get Memory
    model => get_model(tile%owner_model_id)
    grid  => Get_grid(model%grid_id)
    ! Get process config
    dsl4jsb_Get_config(Q_PHENO_)
    ! Get process memories
    dsl4jsb_Get_memory(Q_PHENO_)
    ! Get process variables (Set pointers to variables in memory)
    dsl4jsb_Get_var2D_onDomain(Q_PHENO_, growing_season)
    dsl4jsb_Get_var2D_onDomain(Q_PHENO_, lai_max)

    nproma = grid%nproma
    nblks  = grid%nblks

    !> 1.0 init
    !>
    ! init growing_season and lai_max
    IF (tile%lcts(1)%lib_id /= 0) THEN
      ! work with lctlib only if the present tile is a pft

      config_lai_max        = dsl4jsb_Config(Q_PHENO_)%lai_max
      lctlib_lai_max        = dsl4jsb_Lctlib_param(lai_max)
      lctlib_phenology_type = dsl4jsb_Lctlib_param(phenology_type)

      ! in case the confgi%lai_max value is less than zero, the lctlib value is used
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1)
      DO ib = 1,nblks
        DO ic = 1,nproma
          IF (config_lai_max > 0.0_wp) THEN ! value is set in config script
            ! use configuration script value but constrain to plausbile values between 0.1 and 10
            lai_max(ic,ib) = MAX(0.1_wp, MIN(10.0_wp, config_lai_max))
          ELSE ! use lctlib parameter if config%lai_max was not set (i.e., -1.0)
            lai_max(ic,ib) = lctlib_lai_max
          END IF

          ! growing season flag, set depending on whether this is a evergreen plant (TRUE) or not (FALSE)
          ! NOTE: false and true are of type REAL(wp)
          IF (lctlib_phenology_type == ievergreen) THEN
            growing_season(ic,ib) = true
          ELSE
            growing_season(ic,ib) = false
          END IF
        END DO
      END DO
      !$ACC END PARALLEL LOOP
    ELSE
      ! default for non-PFT tiles
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1)
      DO ib = 1,nblks
        DO ic = 1,nproma
          lai_max(ic,ib)        = 1.0_wp
          growing_season(ic,ib) = false
        END DO
      END DO
      !$ACC END PARALLEL LOOP
    ENDIF

  END SUBROUTINE q_pheno_init_bc

#endif
END MODULE mo_q_pheno_init
