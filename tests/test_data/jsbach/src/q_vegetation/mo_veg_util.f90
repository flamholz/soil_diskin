!> helper routines for vegetation (QUINCY)
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
!>#### various helper routines for the vegetation process
!>

!NEC$ options "-finline-file=externals/jsbach/src/base/mo_jsb_control.pp-jsb.f90"

MODULE mo_veg_util
#ifndef __NO_QUINCY__

  USE mo_kind,                  ONLY: wp
  USE mo_jsb_control,           ONLY: debug_on
  USE mo_exception,             ONLY: message
  USE mo_jsb_math_constants,    ONLY: zero, eps8, eps1

  USE mo_lnd_bgcm_idx
  USE mo_lnd_bgcm_store,          ONLY: t_lnd_bgcm_store
  USE mo_lnd_bgcm_store_class,    ONLY: VEG_BGCM_POOL_ID, VEG_BGCM_GROWTH_ID, VEG_BGCM_LITTERFALL_ID, &
    &                                   VEG_BGCM_ESTABLISHMENT_ID, VEG_BGCM_EXUDATION_ID, VEG_BGCM_RESERVE_USE_ID, &
    &                                   VEG_BGCM_HARVEST_LITTER_ID, VEG_BGCM_HARVEST_TO_PROD_ID

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: reset_veg_fluxes, calculate_time_average_vegetation, test_carbon_conservation

  CHARACTER(len=*), PARAMETER :: modname = 'mo_veg_util'

CONTAINS

  ! ======================================================================================================= !
  !>reset vegetation fluxes (to zero)
  !>
  SUBROUTINE reset_veg_fluxes(tile, options)

    USE mo_jsb_class,             ONLY: Get_model
    USE mo_jsb_tile_class,        ONLY: t_jsb_tile_abstract
    USE mo_jsb_task_class,        ONLY: t_jsb_task_options
    USE mo_jsb_model_class,       ONLY: t_jsb_model
    USE mo_jsb_process_class,     ONLY: VEG_, Q_SYL_, Q_AGR_
    USE mo_jsb_grid_class,        ONLY: t_jsb_vgrid
    USE mo_jsb_grid,              ONLY: Get_vgrid
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Use_memory(VEG_)
    dsl4jsb_Use_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile         !< one tile with data structure for one lct
    TYPE(t_jsb_task_options),   INTENT(in)    :: options      !< model options
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model),      POINTER   :: model                  !< the model
    TYPE(t_lnd_bgcm_store), POINTER   :: bgcm_store             !< the bgcm store of this tile
    TYPE(t_jsb_vgrid),      POINTER   :: vgrid_soil_sb          !< Vertical grid
    INTEGER                           :: iblk, ics, ice, nc     !< dimensions
    INTEGER                           :: nsoil_sb               !< number of soil layers
    INTEGER                           :: nr_of_veg_compartments !< number of compartments in veg bgcms
    INTEGER                           :: nr_of_elements         !< number of elements in veg bgcms
    INTEGER                           :: ic, id_elem, i_soil
    INTEGER                           :: ix_comp, ix_elem       !< loop index
    LOGICAL                           :: config_l_use_pp

    dsl4jsb_Real2D_onChunk      :: maint_respiration_pot
    dsl4jsb_Real2D_onChunk      :: maint_respiration
    dsl4jsb_Real2D_onChunk      :: maint_respiration_c13
    dsl4jsb_Real2D_onChunk      :: maint_respiration_c14
    dsl4jsb_Real2D_onChunk      :: growth_respiration
    dsl4jsb_Real2D_onChunk      :: growth_respiration_c13
    dsl4jsb_Real2D_onChunk      :: growth_respiration_c14
    dsl4jsb_Real2D_onChunk      :: n_transform_respiration
    dsl4jsb_Real2D_onChunk      :: n_fixation_respiration
    dsl4jsb_Real2D_onChunk      :: n_processing_respiration
    dsl4jsb_Real2D_onChunk      :: n_processing_respiration_c13
    dsl4jsb_Real2D_onChunk      :: n_processing_respiration_c14
    dsl4jsb_Real2D_onChunk      :: herbivory_leaf_resp
    dsl4jsb_Real2D_onChunk      :: herbivory_leaf_resp_c13
    dsl4jsb_Real2D_onChunk      :: herbivory_leaf_resp_c14
    dsl4jsb_Real2D_onChunk      :: herbivory_fruit_resp
    dsl4jsb_Real2D_onChunk      :: herbivory_fruit_resp_c13
    dsl4jsb_Real2D_onChunk      :: herbivory_fruit_resp_c14
    dsl4jsb_Real2D_onChunk      :: npp
    dsl4jsb_Real2D_onChunk      :: npp_c13
    dsl4jsb_Real2D_onChunk      :: npp_c14
    dsl4jsb_Real2D_onChunk      :: net_growth
    dsl4jsb_Real2D_onChunk      :: uptake_nh4
    dsl4jsb_Real2D_onChunk      :: uptake_nh4_n15
    dsl4jsb_Real2D_onChunk      :: uptake_no3
    dsl4jsb_Real2D_onChunk      :: uptake_no3_n15
    dsl4jsb_Real2D_onChunk      :: n_fixation
    dsl4jsb_Real2D_onChunk      :: n_fixation_n15
    dsl4jsb_Real2D_onChunk      :: uptake_po4
    dsl4jsb_Real2D_onChunk      :: recycling_leaf_n
    dsl4jsb_Real2D_onChunk      :: recycling_leaf_n15
    dsl4jsb_Real2D_onChunk      :: recycling_leaf_p
    dsl4jsb_Real2D_onChunk      :: recycling_fine_root_n
    dsl4jsb_Real2D_onChunk      :: recycling_fine_root_n15
    dsl4jsb_Real2D_onChunk      :: recycling_fine_root_p
    dsl4jsb_Real2D_onChunk      :: recycling_heart_wood_n
    dsl4jsb_Real2D_onChunk      :: recycling_heart_wood_n15
    dsl4jsb_Real2D_onChunk      :: recycling_heart_wood_p
    dsl4jsb_Real2D_onChunk      :: net_biosphere_production
    dsl4jsb_Real2D_onChunk      :: net_biosphere_production_c13
    dsl4jsb_Real2D_onChunk      :: net_biosphere_production_c14
    dsl4jsb_Real2D_onChunk      :: biological_n_fixation
    dsl4jsb_Real2D_onChunk      :: delta_dens_ind
    dsl4jsb_Real2D_onChunk      :: unit_transpiration
    dsl4jsb_Real3D_onChunk      :: delta_root_fraction_sl

    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':reset_veg_fluxes'
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_mt2L2D :: veg_litterfall_mt
    dsl4jsb_Def_mt2L2D :: veg_growth_mt
    dsl4jsb_Def_mt1L2D :: veg_exudation_mt
    dsl4jsb_Def_mt1L2D :: veg_establishment_mt
    dsl4jsb_Def_mt1L2D :: veg_reserve_use_mt
    dsl4jsb_Def_mt2L2D :: veg_litter_flux_harvest_mt
    dsl4jsb_Def_mt1L2D :: veg_pp_flux_harvest_mt
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_memory(VEG_)
    dsl4jsb_Def_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc
    ! ----------------------------------------------------------------------------------------------------- !
    IF (.NOT. tile%Is_process_calculated(VEG_)) RETURN
    ! ----------------------------------------------------------------------------------------------------- !
    model  => Get_model(tile%owner_model_id)
    vgrid_soil_sb => Get_vgrid('soil_layer_sb')
    nsoil_sb      =  vgrid_soil_sb%n_levels
    ! ----------------------------------------------------------------------------------------------------- !
    IF (model%lctlib(tile%lcts(1)%lib_id)%BareSoilFlag) RETURN !< do not run this routine at tiles like "bare soil" and "urban area"
    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_memory(VEG_)
    dsl4jsb_Get_config(VEG_)
    config_l_use_pp     = dsl4jsb_Config(VEG_)%l_use_product_pools
    ! ----------------------------------------------------------------------------------------------------- !
    bgcm_store => tile%bgcm_store
    dsl4jsb_Get_mt2L2D(VEG_BGCM_LITTERFALL_ID, veg_litterfall_mt)
    dsl4jsb_Get_mt2L2D(VEG_BGCM_GROWTH_ID, veg_growth_mt)
    dsl4jsb_Get_mt1L2D(VEG_BGCM_EXUDATION_ID, veg_exudation_mt)
    dsl4jsb_Get_mt1L2D(VEG_BGCM_ESTABLISHMENT_ID, veg_establishment_mt)
    dsl4jsb_Get_mt1L2D(VEG_BGCM_RESERVE_USE_ID, veg_reserve_use_mt)

    IF (tile%Is_process_calculated(Q_SYL_) .OR. tile%Is_process_calculated(Q_AGR_)) THEN
      dsl4jsb_Get_mt2L2D(VEG_BGCM_HARVEST_LITTER_ID, veg_litter_flux_harvest_mt)

      IF (config_l_use_pp) THEN
        dsl4jsb_Get_mt1L2D(VEG_BGCM_HARVEST_TO_PROD_ID, veg_pp_flux_harvest_mt)
      END IF

    END IF
    nr_of_veg_compartments = SIZE(veg_litterfall_mt,1)
    ! ----------------------------------------------------------------------------------------------------- !

    !>1.0 bgcm fluxes
    !>
    ! set the bgcm matrices to zero
    DO id_elem = FIRST_ELEM_ID, LAST_ELEM_ID
      IF (model%config%is_element_used(id_elem)) THEN
        ix_elem = model%config%elements_index_map(id_elem)    ! get element index in bgcm
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1)
        DO ix_comp = 1, nr_of_veg_compartments
          DO ic = 1, nc
            veg_growth_mt(ix_comp,ix_elem,ic)      = zero
            veg_litterfall_mt(ix_comp,ix_elem,ic)  = zero
            IF (tile%Is_process_calculated(Q_SYL_) .OR. tile%Is_process_calculated(Q_AGR_)) THEN
              veg_litter_flux_harvest_mt(ix_comp,ix_elem,ic)      = zero
            END IF
          END DO
        END DO
        !$ACC END PARALLEL
        !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
        DO ic = 1, nc
          veg_establishment_mt(ix_elem,ic) = zero
          veg_exudation_mt(ix_elem,ic)     = zero
          veg_reserve_use_mt(ix_elem,ic)   = zero

          IF (config_l_use_pp) THEN
            IF (tile%Is_process_calculated(Q_SYL_) .OR. tile%Is_process_calculated(Q_AGR_)) THEN
              veg_pp_flux_harvest_mt(ix_elem,ic)      = zero
            END IF
          END IF
        END DO
        !$ACC END PARALLEL
      END IF
    END DO

    !>2.0 variables
    !>
    ! flux variables
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration_pot)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_respiration_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_respiration_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_transform_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_fixation_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_processing_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_processing_respiration_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_processing_respiration_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_processing_respiration_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_leaf_resp)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_leaf_resp_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_leaf_resp_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_fruit_resp)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_fruit_resp_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, herbivory_fruit_resp_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_growth)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_nh4)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_nh4_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_no3)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_no3_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_fixation)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_fixation_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_po4)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_leaf_n)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_leaf_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_leaf_p)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_fine_root_n)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_fine_root_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_fine_root_p)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_heart_wood_n)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_heart_wood_n15)
    dsl4jsb_Get_var2D_onChunk(VEG_, recycling_heart_wood_p)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_biosphere_production)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_biosphere_production_c13)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_biosphere_production_c14)
    dsl4jsb_Get_var2D_onChunk(VEG_, biological_n_fixation)
    dsl4jsb_Get_var2D_onChunk(VEG_, delta_dens_ind)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_transpiration)
    dsl4jsb_Get_var3D_onChunk(VEG_, delta_root_fraction_sl)

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
    DO ic = 1, nc
      maint_respiration_pot(ic) = zero
      maint_respiration(ic) = zero
      maint_respiration_c13(ic) = zero
      maint_respiration_c14(ic) = zero
      growth_respiration(ic) = zero
      growth_respiration_c13(ic) = zero
      growth_respiration_c14(ic) = zero
      n_transform_respiration(ic) = zero
      n_fixation_respiration(ic) = zero
      n_processing_respiration(ic) = zero
      n_processing_respiration_c13(ic) = zero
      n_processing_respiration_c14(ic) = zero
      herbivory_leaf_resp(ic) = zero
      herbivory_leaf_resp_c13(ic) = zero
      herbivory_leaf_resp_c14(ic) = zero
      herbivory_fruit_resp(ic) = zero
      herbivory_fruit_resp_c13(ic) = zero
      herbivory_fruit_resp_c14(ic) = zero
      npp(ic) = zero
      npp_c13(ic) = zero
      npp_c14(ic) = zero
      net_growth(ic) = zero
      uptake_nh4(ic) = zero
      uptake_nh4_n15(ic) = zero
      uptake_no3(ic) = zero
      uptake_no3_n15(ic) = zero
      n_fixation(ic) = zero
      n_fixation_n15(ic) = zero
      uptake_po4(ic) = zero
      recycling_leaf_n(ic) = zero
      recycling_leaf_n15(ic) = zero
      recycling_leaf_p(ic) = zero
      recycling_fine_root_n(ic) = zero
      recycling_fine_root_n15(ic) = zero
      recycling_fine_root_p(ic) = zero
      recycling_heart_wood_n(ic) = zero
      recycling_heart_wood_n15(ic) = zero
      recycling_heart_wood_p(ic) = zero
      net_biosphere_production(ic) = zero
      net_biosphere_production_c13(ic) = zero
      net_biosphere_production_c14(ic) = zero
      biological_n_fixation(ic) = zero
      delta_dens_ind(ic) = zero
      unit_transpiration(ic) = zero
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR COLLAPSE(2) ASYNC(1)
    DO i_soil = 1, nsoil_sb
      DO ic = 1, nc
        delta_root_fraction_sl(ic, i_soil) = zero
      END DO
    END DO
    !$ACC END PARALLEL LOOP


  END SUBROUTINE reset_veg_fluxes


  ! ======================================================================================================= !
  !> calculate time moving averages and daytime averages for VEG_
  !>
  SUBROUTINE calculate_time_average_vegetation(tile, options)

    USE mo_jsb_impl_constants,      ONLY: test_false_true
    USE mo_jsb_tile_class,          ONLY: t_jsb_tile_abstract
    USE mo_jsb_task_class,          ONLY: t_jsb_task_options
    USE mo_jsb_lctlib_class,        ONLY: t_lctlib_element
    USE mo_jsb_model_class,         ONLY: t_jsb_model
    USE mo_jsb_class,               ONLY: Get_model
    USE mo_jsb_grid_class,          ONLY: t_jsb_vgrid
    USE mo_jsb_grid,                ONLY: Get_vgrid
    USE mo_lnd_time_averages        ! e.g. calc_time_mavg, mavg_period_tphen, mavg_period_weekly
    dsl4jsb_Use_processes A2L_, Q_ASSIMI_, VEG_, Q_PHENO_, HYDRO_, SSE_
    !------------------------------------------------------------------------------------------------------ !
    dsl4jsb_Use_config(VEG_)
    dsl4jsb_Use_config(Q_ASSIMI_)
    dsl4jsb_Use_memory(A2L_)
    dsl4jsb_Use_memory(Q_ASSIMI_)
    dsl4jsb_Use_memory(VEG_)
    dsl4jsb_Use_memory(Q_PHENO_)
    dsl4jsb_Use_memory(HYDRO_)
    dsl4jsb_Use_memory(SSE_)
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout) :: tile
    TYPE(t_jsb_task_options),   INTENT(in)    :: options
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model),      POINTER         :: model                  !< instance of the model
    TYPE(t_lnd_bgcm_store), POINTER         :: bgcm_store             !< the bgcm store of this tile
    TYPE(t_lctlib_element), POINTER         :: lctlib                 !< land-cover-type library - parameter across pft's
    TYPE(t_jsb_vgrid),      POINTER         :: vgrid_canopy_q_assimi  !< Vertical grid
    LOGICAL,  DIMENSION(options%nc)         :: l_growing_season       !< growing_season LOGICAL
    LOGICAL,  ALLOCATABLE, DIMENSION(:,:)   :: l_growing_season_cl    !< growing_season LOGICAL, at vgrid_canopy_q_assimi
    LOGICAL                                 :: l_hlp                  !< helper
    REAl(wp)                                :: hlp_r                  !< helper
    REAl(wp)                                :: mavg_period_hlp        !< helper
    INTEGER                                 :: ic                     !< looping over chunk
    INTEGER                                 :: icanopy                !< looping
    INTEGER                                 :: ncanopy                !< number of canopy layers, from vgrid
    INTEGER                                 :: iblk, ics, ice, nc     !< dimensions
    REAL(wp)                                :: dtime                  !< timestep
    REAL(wp)                                :: lctlib_tau_mycorrhiza  !< lctlib parameter (tau of mycorrhiza)
    REAL(wp)                                :: lctlib_tau_fine_root   !< lctlib parameter (tau of fine roots)
    CHARACTER(len=*), PARAMETER             :: routine = modname//':calculate_time_average_vegetation'
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_mt2L2D :: veg_pool_mt
    dsl4jsb_Def_mt1L2D :: veg_exudation_mt
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_config(Q_ASSIMI_)
    dsl4jsb_Def_config(VEG_)
    dsl4jsb_Def_memory(A2L_)
    dsl4jsb_Def_memory(Q_ASSIMI_)
    dsl4jsb_Def_memory(VEG_)
    dsl4jsb_Def_memory(Q_PHENO_)
    dsl4jsb_Def_memory(HYDRO_)
    dsl4jsb_Def_memory(SSE_)
    ! ----------------------------------------------------------------------------------------------------- !
    ! A2L_
    dsl4jsb_Real2D_onChunk      :: t_air
    dsl4jsb_Real2D_onChunk      :: press_srf
    dsl4jsb_Real2D_onChunk      :: co2_mixing_ratio
    dsl4jsb_Real2D_onChunk      :: swpar_srf_down
    dsl4jsb_Real2D_onChunk      :: daytime_counter
    dsl4jsb_Real2D_onChunk      :: local_time_day_seconds
    ! Q_ASSIMI_
    dsl4jsb_Real2D_onChunk      :: gross_assimilation
    dsl4jsb_Real2D_onChunk      :: net_assimilation_boc
    dsl4jsb_Real2D_onChunk      :: t_jmax_opt
    dsl4jsb_Real2D_onChunk      :: aerodyn_cond
    ! Q_PHENO_ 2D
    dsl4jsb_Real2D_onChunk      :: growing_season
    ! VEG_ 2D
    dsl4jsb_Real2D_onChunk      :: lai
    dsl4jsb_Real2D_onChunk      :: n_fixation
    dsl4jsb_Real2D_onChunk      :: press_srf_daytime
    dsl4jsb_Real2D_onChunk      :: t_air_daytime
    dsl4jsb_Real2D_onChunk      :: f_p_demand
    dsl4jsb_Real2D_onChunk      :: f_n_demand
    dsl4jsb_Real2D_onChunk      :: npp
    dsl4jsb_Real2D_onChunk      :: net_growth
    dsl4jsb_Real2D_onChunk      :: maint_respiration_pot
    dsl4jsb_Real2D_onChunk      :: growth_req_n
    dsl4jsb_Real2D_onChunk      :: growth_req_p
    dsl4jsb_Real2D_onChunk      :: unit_npp
    dsl4jsb_Real2D_onChunk      :: unit_transpiration
    dsl4jsb_Real2D_onChunk      :: dphi
    dsl4jsb_Real2D_onChunk      :: uptake_nh4
    dsl4jsb_Real2D_onChunk      :: uptake_no3
    dsl4jsb_Real2D_onChunk      :: unit_uptake_n_act
    dsl4jsb_Real2D_onChunk      :: unit_uptake_p_act
    dsl4jsb_Real2D_onChunk      :: unit_uptake_n_pot
    dsl4jsb_Real2D_onChunk      :: unit_uptake_p_pot
    dsl4jsb_Real2D_onChunk      :: fmaint_rate_root
    dsl4jsb_Real2D_onChunk      :: t_jmax_opt_daytime
    dsl4jsb_Real2D_onChunk      :: co2_mixing_ratio_daytime
    dsl4jsb_Real2D_onChunk      :: ga_daytime
    dsl4jsb_Real2D_onChunk      :: beta_sinklim_ps_daytime
    dsl4jsb_Real2D_onChunk      :: beta_sinklim_ps
    dsl4jsb_Real2D_onChunk      :: t_jmax_opt_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_month_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_week_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_tacclim_mavg
    dsl4jsb_Real2D_onChunk      :: t_soil_root_tacclim_mavg
    dsl4jsb_Real2D_onChunk      :: beta_sinklim_ps_tacclim_mavg
    dsl4jsb_Real2D_onChunk      :: an_boc_tvegdyn_mavg
    dsl4jsb_Real2D_onChunk      :: net_growth_tvegdyn_mavg
    dsl4jsb_Real2D_onChunk      :: lai_tvegdyn_mavg
    dsl4jsb_Real2D_onChunk      :: fmaint_rate_troot_mavg
    dsl4jsb_Real2D_onChunk      :: unit_uptake_p_pot_troot_mavg
    dsl4jsb_Real2D_onChunk      :: unit_uptake_n_pot_troot_mavg
    dsl4jsb_Real2D_onChunk      :: unit_npp_troot_mavg
    dsl4jsb_Real2D_onChunk      :: leaf2root_troot_mavg
    dsl4jsb_Real2D_onChunk      :: growth_cp_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: growth_cn_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: growth_np_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: growth_p_limit_based_on_n_mavg
    dsl4jsb_Real2D_onChunk      :: unit_npp_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: n_fixation_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: npp_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: beta_sinklim_ps_tfrac_mavg
    dsl4jsb_Real2D_onChunk      :: ga_tfrac_mavg
    dsl4jsb_Real2D_onChunk      :: co2_mixing_ratio_tfrac_mavg
    dsl4jsb_Real2D_onChunk      :: press_srf_tfrac_mavg
    dsl4jsb_Real2D_onChunk      :: gpp_tlabile_mavg
    dsl4jsb_Real2D_onChunk      :: maint_respiration_tlabile_mavg
    dsl4jsb_Real2D_onChunk      :: growth_req_n_tlabile_mavg
    dsl4jsb_Real2D_onChunk      :: growth_req_p_tlabile_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_tphen_mavg
    dsl4jsb_Real2D_onChunk      :: t_soil_srf_tphen_mavg
    dsl4jsb_Real2D_onChunk      :: npp_tuptake_mavg
    dsl4jsb_Real2D_onChunk      :: demand_uptake_n_tuptake_mavg
    dsl4jsb_Real2D_onChunk      :: demand_uptake_p_tuptake_mavg
    dsl4jsb_Real2D_onChunk      :: growth_req_n_tuptake_mavg
    dsl4jsb_Real2D_onChunk      :: growth_req_p_tuptake_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_tfrac_mavg
    dsl4jsb_Real2D_onChunk      :: t_air_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: press_srf_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: co2_mixing_ratio_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: ga_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: uptake_n_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: growth_cn_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: growth_np_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: npp_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: fmaint_rate_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: labile_carbon_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: labile_nitrogen_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: labile_phosphorus_tcnl_mavg
    dsl4jsb_Real2D_onChunk      :: transpiration_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: unit_transpiration_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: dphi_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: unit_uptake_n_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: unit_uptake_p_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: labile_carbon_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: labile_nitrogen_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: labile_phosphorus_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: reserve_carbon_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: reserve_nitrogen_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: reserve_phosphorus_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: w_root_lim_talloc_mavg
    dsl4jsb_Real2D_onChunk      :: growth_respiration
    dsl4jsb_Real2D_onChunk      :: maint_respiration
    dsl4jsb_Real2D_onChunk      :: n_processing_respiration
    dsl4jsb_Real2D_onChunk      :: exudation_c_tmyc_mavg
    dsl4jsb_Real2D_onChunk      :: press_srf_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: ga_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: t_jmax_opt_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: co2_mixing_ratio_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: beta_sinklim_ps_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: t_air_daytime_dacc
    dsl4jsb_Real2D_onChunk      :: t_soil_root
    dsl4jsb_Real2D_onChunk      :: rfr_ratio_boc
    dsl4jsb_Real2D_onChunk      :: rfr_ratio_boc_tvegdyn_mavg
    ! VEG_ 3D
    dsl4jsb_Real3D_onChunk      :: fleaf_sunlit_tcnl_mavg_cl
    dsl4jsb_Real3D_onChunk      :: fleaf_sunlit_tfrac_mavg_cl
    dsl4jsb_Real3D_onChunk      :: fleaf_sunlit_daytime_cl
    dsl4jsb_Real3D_onChunk      :: fleaf_sunlit_cl
    dsl4jsb_Real3D_onChunk      :: fleaf_sunlit_daytime_dacc_cl
    ! HYDRO_ 2D
    dsl4jsb_Real2D_onChunk      :: w_soil_root_fc
    dsl4jsb_Real2D_onChunk      :: wtr_rootzone
    dsl4jsb_Real2D_onChunk      :: transpiration
    ! SSE_ 3D
    dsl4jsb_Real3D_onChunk      :: t_soil_sl
    ! ----------------------------------------------------------------------------------------------------- !
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc
    dtime   = options%dtime
    ! ----------------------------------------------------------------------------------------------------- !
    IF (.NOT. tile%Is_process_calculated(VEG_)) RETURN
    IF (tile%lcts(1)%lib_id == 0) RETURN !< only if the present tile is a pft
    ! ----------------------------------------------------------------------------------------------------- !
    model                 => Get_model(tile%owner_model_id)
    lctlib                => model%lctlib(tile%lcts(1)%lib_id)
    vgrid_canopy_q_assimi => Get_vgrid('q_canopy_layer')
    ncanopy               =  vgrid_canopy_q_assimi%n_levels
    ! ----------------------------------------------------------------------------------------------------- !
    IF (lctlib%BareSoilFlag) RETURN !< do not run this routine at tiles like "bare soil" and "urban area"
    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_config(Q_ASSIMI_)
    dsl4jsb_Get_config(VEG_)
    dsl4jsb_Get_memory(A2L_)
    dsl4jsb_Get_memory(Q_ASSIMI_)
    dsl4jsb_Get_memory(VEG_)
    dsl4jsb_Get_memory(Q_PHENO_)
    dsl4jsb_Get_memory(HYDRO_)
    dsl4jsb_Get_memory(SSE_)
    ! ----------------------------------------------------------------------------------------------------- !
    bgcm_store => tile%bgcm_store
    dsl4jsb_Get_mt2L2D(VEG_BGCM_POOL_ID, veg_pool_mt)
    dsl4jsb_Get_mt1L2D(VEG_BGCM_EXUDATION_ID, veg_exudation_mt)
    ! ----------------------------------------------------------------------------------------------------- !
    ! A2L_
    dsl4jsb_Get_var2D_onChunk(A2L_, t_air)
    dsl4jsb_Get_var2D_onChunk(A2L_, press_srf)
    dsl4jsb_Get_var2D_onChunk(A2L_, co2_mixing_ratio)
    dsl4jsb_Get_var2D_onChunk(A2L_, swpar_srf_down)
    dsl4jsb_Get_var2D_onChunk(A2L_, daytime_counter)
    dsl4jsb_Get_var2D_onChunk(A2L_, local_time_day_seconds)
    ! Q_ASSIMI_
    dsl4jsb_Get_var2D_onChunk(Q_ASSIMI_, gross_assimilation)
    dsl4jsb_Get_var2D_onChunk(Q_ASSIMI_, net_assimilation_boc)
    dsl4jsb_Get_var2D_onChunk(Q_ASSIMI_, t_jmax_opt)
    dsl4jsb_Get_var2D_onChunk(Q_ASSIMI_, aerodyn_cond)
    ! Q_PHENO_ 2D
    dsl4jsb_Get_var2D_onChunk(Q_PHENO_, growing_season)             ! in
    ! VEG_ 2D
    dsl4jsb_Get_var2D_onChunk(VEG_, lai)                            ! in
    dsl4jsb_Get_var2D_onChunk(VEG_, n_fixation)
    dsl4jsb_Get_var2D_onChunk(VEG_, press_srf_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, f_p_demand)
    dsl4jsb_Get_var2D_onChunk(VEG_, f_n_demand)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_growth)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration_pot)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_n)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_p)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_npp)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_transpiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, dphi)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_n_act)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_p_act)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_n_pot)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_p_pot)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_nh4)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_no3)
    dsl4jsb_Get_var2D_onChunk(VEG_, fmaint_rate_root)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_jmax_opt_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, co2_mixing_ratio_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, ga_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, beta_sinklim_ps_daytime)
    dsl4jsb_Get_var2D_onChunk(VEG_, beta_sinklim_ps)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_jmax_opt_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_month_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_week_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_tacclim_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_soil_root_tacclim_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, beta_sinklim_ps_tacclim_mavg) !New in QS (8b10a1d)
    dsl4jsb_Get_var2D_onChunk(VEG_, an_boc_tvegdyn_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_growth_tvegdyn_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, lai_tvegdyn_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, fmaint_rate_troot_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_p_pot_troot_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_n_pot_troot_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_npp_troot_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, leaf2root_troot_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_cp_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_cn_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_np_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_p_limit_based_on_n_mavg) !New in QS (d1b74e0)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_npp_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_fixation_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, beta_sinklim_ps_tfrac_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, ga_tfrac_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, co2_mixing_ratio_tfrac_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, press_srf_tfrac_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, gpp_tlabile_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration_tlabile_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_n_tlabile_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_p_tlabile_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_tphen_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_soil_srf_tphen_mavg) !New in QS (0e0a3dc)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp_tuptake_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, demand_uptake_n_tuptake_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, demand_uptake_p_tuptake_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_n_tuptake_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_req_p_tuptake_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_tfrac_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, press_srf_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, co2_mixing_ratio_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, ga_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, uptake_n_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_cn_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_np_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, npp_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, fmaint_rate_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_carbon_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_nitrogen_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_phosphorus_tcnl_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, transpiration_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_transpiration_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, dphi_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_n_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, unit_uptake_p_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_carbon_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_nitrogen_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, labile_phosphorus_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, reserve_carbon_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, reserve_nitrogen_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, reserve_phosphorus_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, w_root_lim_talloc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, growth_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, maint_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, n_processing_respiration)
    dsl4jsb_Get_var2D_onChunk(VEG_, exudation_c_tmyc_mavg)
    dsl4jsb_Get_var2D_onChunk(VEG_, press_srf_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, ga_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_jmax_opt_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, co2_mixing_ratio_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, beta_sinklim_ps_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_air_daytime_dacc)
    dsl4jsb_Get_var2D_onChunk(VEG_, t_soil_root)
    dsl4jsb_Get_var2D_onChunk(VEG_, rfr_ratio_boc)                      ! in
    dsl4jsb_Get_var2D_onChunk(VEG_, rfr_ratio_boc_tvegdyn_mavg)         ! inout
    ! VEG_ 3D
    dsl4jsb_Get_var3D_onChunk(VEG_, fleaf_sunlit_tcnl_mavg_cl)
    dsl4jsb_Get_var3D_onChunk(VEG_, fleaf_sunlit_tfrac_mavg_cl)
    dsl4jsb_Get_var3D_onChunk(VEG_, fleaf_sunlit_daytime_cl)
    dsl4jsb_Get_var3D_onChunk(VEG_, fleaf_sunlit_cl)
    dsl4jsb_Get_var3D_onChunk(VEG_, fleaf_sunlit_daytime_dacc_cl)
    ! HYDRO_ 2D
    dsl4jsb_Get_var2D_onChunk(HYDRO_, w_soil_root_fc)
    dsl4jsb_Get_var2D_onChunk(HYDRO_, wtr_rootzone)
    dsl4jsb_Get_var2D_onChunk(HYDRO_, transpiration)
    ! SSE_ 3D
    dsl4jsb_Get_var3D_onChunk(SSE_, t_soil_sl)                      ! in
    ! ----------------------------------------------------------------------------------------------------- !

    lctlib_tau_mycorrhiza = dsl4jsb_Lctlib_param(tau_mycorrhiza)
    lctlib_tau_fine_root  = dsl4jsb_Lctlib_param(tau_fine_root)

    ALLOCATE(l_growing_season_cl(nc, ncanopy))
    !$ACC DATA CREATE(l_growing_season_cl, l_growing_season)

    !>0.9 transform REAL growing_season(:) into LOGICAL l_growing_season(:)/_cl(:,:)
    !>
    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      IF (growing_season(ic) > test_false_true) THEN
        l_growing_season(ic)      = .TRUE.
        DO icanopy = 1, ncanopy
          l_growing_season_cl(ic,icanopy) = .TRUE.
        END DO
      ELSE
        l_growing_season(ic)      = .FALSE.
        DO icanopy = 1, ncanopy
          l_growing_season_cl(ic,icanopy) = .FALSE.
        END DO
      END IF
    END DO
    !$ACC END PARALLEL LOOP

    !>1.0 daytime averages
    !>

    !>  1.1 accumulate values - if light is available (i.e. daytime)
    !>
    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      IF (swpar_srf_down(ic) > eps8) THEN
        ! 1D var - add values
        t_air_daytime_dacc(ic)            = t_air_daytime_dacc(ic)            + t_air(ic)
        press_srf_daytime_dacc(ic)        = press_srf_daytime_dacc(ic)        + press_srf(ic)
        co2_mixing_ratio_daytime_dacc(ic) = co2_mixing_ratio_daytime_dacc(ic) + co2_mixing_ratio(ic)
        ga_daytime_dacc(ic)               = ga_daytime_dacc(ic)               + aerodyn_cond(ic)
        beta_sinklim_ps_daytime_dacc(ic)  = beta_sinklim_ps_daytime_dacc(ic)  + beta_sinklim_ps(ic)
        t_jmax_opt_daytime_dacc(ic)       = t_jmax_opt_daytime_dacc(ic)       + t_jmax_opt(ic)
        ! 2D var - add values
        DO icanopy = 1, ncanopy
          fleaf_sunlit_daytime_dacc_cl(ic,icanopy) = &
            & fleaf_sunlit_daytime_dacc_cl(ic,icanopy) + fleaf_sunlit_cl(ic,icanopy)
        END DO
      END IF
    END DO
    !$ACC END PARALLEL LOOP

    !>  1.2 calc daytime averages - at the timestep after local midnight (1st timestep of the new day)
    !>
    !>    calculate the average of the previous day and pass this value to the according variable
    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
       IF (ABS(local_time_day_seconds(ic) - dtime) < eps1) THEN
        ! check if at least one value had been assigned at the current day, avoiding division by zero
        IF (daytime_counter(ic) > eps8) THEN
          ! 1D
          t_air_daytime(ic)             = t_air_daytime_dacc(ic)              / daytime_counter(ic)
          press_srf_daytime(ic)         = press_srf_daytime_dacc(ic)          / daytime_counter(ic)
          co2_mixing_ratio_daytime(ic)  = co2_mixing_ratio_daytime_dacc(ic)   / daytime_counter(ic)
          ga_daytime(ic)                = ga_daytime_dacc(ic)                 / daytime_counter(ic)
          beta_sinklim_ps_daytime(ic)   = beta_sinklim_ps_daytime_dacc(ic)    / daytime_counter(ic)
          t_jmax_opt_daytime(ic)        = t_jmax_opt_daytime_dacc(ic)         / daytime_counter(ic)
          ! 2D
          DO icanopy = 1, ncanopy
            fleaf_sunlit_daytime_cl(ic,icanopy) = fleaf_sunlit_daytime_dacc_cl(ic,icanopy)  / daytime_counter(ic)
          END DO
        ELSE
          ! if there was no daylight during the previous day, just leave previous values for most variables
          DO icanopy = 1, ncanopy
            fleaf_sunlit_daytime_cl(ic,icanopy) = 0.0_wp
          END DO
        END IF
        ! zero the accumulation variables after daily average has been calculated
        ! 1D
        t_air_daytime_dacc(ic)                 = 0.0_wp
        press_srf_daytime_dacc(ic)             = 0.0_wp
        co2_mixing_ratio_daytime_dacc(ic)      = 0.0_wp
        ga_daytime_dacc(ic)                    = 0.0_wp
        beta_sinklim_ps_daytime_dacc(ic)       = 0.0_wp
        t_jmax_opt_daytime_dacc(ic)            = 0.0_wp
        ! 2D
        DO icanopy = 1, ncanopy
          fleaf_sunlit_daytime_dacc_cl(ic,icanopy)     = 0.0_wp
        END DO
      END IF
    END DO
    !$ACC END PARALLEL LOOP

    ! ----------------------------------------------------------------------------------------------------- !
    !>2.0 moving averages
    !>
    !>

    ! docu:
    ! calc_time_mavg(dtime, current average, new value, length of avg_period,  !
    !                do_calc=LOGICAL, avg_period_unit='day')            ! OPTIONAL
    !                RETURN(new current average)
    ! the unit of the averaging period is 'day' by default, but can also be 'week' or 'year'

    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      !>  2.1 tlabile (averages at the timescale of the labile pool)
      !>
      gpp_tlabile_mavg(ic)                = calc_time_mavg(dtime, gpp_tlabile_mavg(ic), gross_assimilation(ic), &
                                                          mavg_period_tlabile)
      maint_respiration_tlabile_mavg(ic)  = calc_time_mavg(dtime, maint_respiration_tlabile_mavg(ic), maint_respiration_pot(ic), &
                                                          mavg_period_tlabile)
      growth_req_n_tlabile_mavg(ic)       = calc_time_mavg(dtime, growth_req_n_tlabile_mavg(ic), growth_req_n(ic), &
                                                          mavg_period_tlabile, do_calc=(growth_req_n(ic) > eps8))
      growth_req_p_tlabile_mavg(ic)       = calc_time_mavg(dtime, growth_req_p_tlabile_mavg(ic), growth_req_p(ic), &
                                                          mavg_period_tlabile, do_calc=(growth_req_p(ic) > eps8))

      !>  2.2 tphen (averages at the time-scale of phenology)
      !>
      t_air_tphen_mavg(ic)                = calc_time_mavg(dtime, t_air_tphen_mavg(ic), t_air(ic), mavg_period_tphen)
      t_soil_srf_tphen_mavg(ic)           = calc_time_mavg(dtime, t_soil_srf_tphen_mavg(ic), t_soil_sl(ic,1), mavg_period_tphen)   ! one could also try taking the first few layers here


      !>  2.3 tuptake (averages at the nutrient-uptake demand time-scale)
      !>
      npp_tuptake_mavg(ic)                 = calc_time_mavg(dtime, npp_tuptake_mavg(ic), npp(ic), mavg_period_tuptake,         &
        &                                     do_calc=(veg_pool_mt(ix_fine_root,ixC,ic) > eps8))
      demand_uptake_n_tuptake_mavg(ic)     = calc_time_mavg(dtime, demand_uptake_n_tuptake_mavg(ic), f_n_demand(ic),           &
        &                                     mavg_period_tuptake, do_calc=(veg_pool_mt(ix_fine_root,ixC,ic) > eps8))
      demand_uptake_p_tuptake_mavg(ic)     = calc_time_mavg(dtime, demand_uptake_p_tuptake_mavg(ic), f_p_demand(ic),           &
        &                                     mavg_period_tuptake, do_calc=(veg_pool_mt(ix_fine_root,ixC,ic) > eps8))
      growth_req_n_tuptake_mavg(ic)        = calc_time_mavg(dtime, growth_req_n_tuptake_mavg(ic), growth_req_n(ic), &
                                                           mavg_period_tuptake, do_calc=(growth_req_n(ic) > eps8))
      growth_req_p_tuptake_mavg(ic)        = calc_time_mavg(dtime, growth_req_p_tuptake_mavg(ic), growth_req_p(ic), &
                                                           mavg_period_tuptake, do_calc=(growth_req_p(ic) > eps8))

      !>  2.4 tfrac (averages at the time-scale of within-leaf N allocation fractions)
      !>
      DO icanopy = 1, ncanopy
        fleaf_sunlit_tfrac_mavg_cl(ic,icanopy)   = calc_time_mavg(dtime, fleaf_sunlit_tfrac_mavg_cl(ic,icanopy), &
          & fleaf_sunlit_daytime_cl(ic,icanopy), mavg_period_tfrac, do_calc=l_growing_season_cl(ic,icanopy))
      END DO

      t_air_tfrac_mavg(ic)               = calc_time_mavg(dtime, t_air_tfrac_mavg(ic), t_air_daytime(ic), &
                                                         mavg_period_tfrac)
      press_srf_tfrac_mavg(ic)           = calc_time_mavg(dtime, press_srf_tfrac_mavg(ic), press_srf_daytime(ic), &
                                                         mavg_period_tfrac)
      co2_mixing_ratio_tfrac_mavg(ic)    = calc_time_mavg(dtime, co2_mixing_ratio_tfrac_mavg(ic), co2_mixing_ratio_daytime(ic), &
                                                         mavg_period_tfrac)
      ga_tfrac_mavg(ic)                  = calc_time_mavg(dtime, ga_tfrac_mavg(ic), ga_daytime(ic), &
                                                         mavg_period_tfrac)
      beta_sinklim_ps_tfrac_mavg(ic)     = calc_time_mavg(dtime, beta_sinklim_ps_tfrac_mavg(ic), beta_sinklim_ps_daytime(ic), &
                                                         mavg_period_tfrac)

      !>  2.5 tcnl (averages at the time-scale of leaf N allocation fractions)
      !>
      DO icanopy = 1, ncanopy
        fleaf_sunlit_tcnl_mavg_cl(ic,icanopy) = calc_time_mavg(dtime, fleaf_sunlit_tcnl_mavg_cl(ic,icanopy),  &
          &                                                    fleaf_sunlit_cl(ic,icanopy), mavg_period_tcnl, &
          &                                                    do_calc=l_growing_season_cl(ic,icanopy))
      END DO

      t_air_tcnl_mavg(ic)               = calc_time_mavg(dtime, t_air_tcnl_mavg(ic), t_air(ic), &
                                                        mavg_period_tcnl, do_calc=l_growing_season(ic))
      press_srf_tcnl_mavg(ic)           = calc_time_mavg(dtime, press_srf_tcnl_mavg(ic), press_srf(ic), &
                                                        mavg_period_tcnl,  do_calc=l_growing_season(ic))
      co2_mixing_ratio_tcnl_mavg(ic)    = calc_time_mavg(dtime, co2_mixing_ratio_tcnl_mavg(ic), co2_mixing_ratio(ic), &
                                                        mavg_period_tcnl,  do_calc=l_growing_season(ic))
      ga_tcnl_mavg(ic)                  = calc_time_mavg(dtime, ga_tcnl_mavg(ic), aerodyn_cond(ic), &
                                                        mavg_period_tcnl,  do_calc=l_growing_season(ic))
      uptake_n_tcnl_mavg(ic)               = calc_time_mavg(dtime, uptake_n_tcnl_mavg(ic),                                 &
        &                                  unit_uptake_n_pot(ic) * veg_pool_mt(ix_fine_root,ixC,ic) + n_fixation(ic), &
        &                                  mavg_period_tgrowth, do_calc=l_growing_season(ic))
      fmaint_rate_tcnl_mavg(ic)            = calc_time_mavg(dtime, fmaint_rate_tcnl_mavg(ic), fmaint_rate_root(ic), &
                                                        mavg_period_tgrowth, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF (growth_req_n(ic) > 0._wp) THEN
        hlp_r = 1._wp / growth_req_n(ic)
      ELSE
        hlp_r = 0._wp
      END IF
      growth_cn_tcnl_mavg(ic)  = calc_time_mavg(dtime, growth_cn_tcnl_mavg(ic), hlp_r, &
        &                                       mavg_period_tcnl,  do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF (growth_req_p(ic) > 0._wp) THEN
        hlp_r = 1._wp / growth_req_p(ic)
      ELSE
        hlp_r = 0._wp
      END IF
      growth_np_tcnl_mavg(ic)  = calc_time_mavg(dtime, growth_np_tcnl_mavg(ic), hlp_r,   &
        &                                       mavg_period_tcnl,  do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      hlp_r                    = gross_assimilation(ic) / MAX(eps8, beta_sinklim_ps(ic)) &
        &                        - growth_respiration(ic) - maint_respiration(ic) - n_processing_respiration(ic)
      npp_tcnl_mavg(ic)        = calc_time_mavg(dtime, npp_tcnl_mavg(ic), hlp_r, mavg_period_tcnl, do_calc=l_growing_season(ic))

      labile_carbon_tcnl_mavg(ic)     = calc_time_mavg(dtime, labile_carbon_tcnl_mavg(ic), veg_pool_mt(ix_labile,ixC,ic),     &
        &                                   mavg_period_tcnl, do_calc=l_growing_season(ic))
      labile_nitrogen_tcnl_mavg(ic)   = calc_time_mavg(dtime, labile_nitrogen_tcnl_mavg(ic), veg_pool_mt(ix_labile,ixN,ic),   &
        &                                   mavg_period_tcnl, do_calc=l_growing_season(ic))
      labile_phosphorus_tcnl_mavg(ic) = calc_time_mavg(dtime, labile_phosphorus_tcnl_mavg(ic), veg_pool_mt(ix_labile,ixP,ic), &
        &                                   mavg_period_tcnl, do_calc=l_growing_season(ic))

      !>  2.6 talloc (time-scale of plant allocation calculation (veg process))
      !>
      hlp_r                = gross_assimilation(ic) / MAX(eps8, beta_sinklim_ps(ic)) &
        &                    - growth_respiration(ic) - maint_respiration(ic) - n_processing_respiration(ic)
      npp_talloc_mavg(ic)  = calc_time_mavg(dtime, npp_talloc_mavg(ic), hlp_r, mavg_period_talloc, do_calc=l_growing_season(ic))
      !Note: different than in IQ [changed in QS (!23)]
      n_fixation_talloc_mavg(ic)         = calc_time_mavg(dtime, n_fixation_talloc_mavg(ic), n_fixation(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      unit_npp_talloc_mavg(ic)           = calc_time_mavg(dtime, unit_npp_talloc_mavg(ic), unit_npp(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      transpiration_talloc_mavg(ic)      = calc_time_mavg(dtime, transpiration_talloc_mavg(ic), transpiration(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      unit_transpiration_talloc_mavg(ic) = calc_time_mavg(dtime, unit_transpiration_talloc_mavg(ic), unit_transpiration(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      dphi_talloc_mavg(ic)               = calc_time_mavg(dtime, dphi_talloc_mavg(ic), dphi(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      unit_uptake_n_talloc_mavg(ic)      = calc_time_mavg(dtime, unit_uptake_n_talloc_mavg(ic), unit_uptake_n_pot(ic), &
        &                                                 mavg_period_talloc, do_calc=l_growing_season(ic))
      unit_uptake_p_talloc_mavg(ic)      = calc_time_mavg(dtime, unit_uptake_p_talloc_mavg(ic), unit_uptake_p_pot(ic), &
                                                          mavg_period_talloc, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !>  2.7 talloc / talloc_dynamic
    !>
    SELECT CASE (TRIM(dsl4jsb_Config(VEG_)%biomass_alloc_scheme))
    CASE ("optimal", "fixed")
      mavg_period_hlp = mavg_period_talloc
    CASE ("dynamic")
      mavg_period_hlp = mavg_period_talloc_dynamic
    END SELECT

    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      labile_carbon_talloc_mavg(ic)      = calc_time_mavg(dtime, labile_carbon_talloc_mavg(ic), veg_pool_mt(ix_labile,ixC,ic),   &
        &                                     mavg_period_hlp, do_calc=l_growing_season(ic))
      labile_nitrogen_talloc_mavg(ic)    = calc_time_mavg(dtime, labile_nitrogen_talloc_mavg(ic), veg_pool_mt(ix_labile,ixN,ic), &
        &                                     mavg_period_hlp, do_calc=l_growing_season(ic))
      labile_phosphorus_talloc_mavg(ic)  = calc_time_mavg(dtime, labile_phosphorus_talloc_mavg(ic),  &
        &                                     veg_pool_mt(ix_labile,ixP,ic), mavg_period_hlp, do_calc=l_growing_season(ic))
      reserve_carbon_talloc_mavg(ic)     = calc_time_mavg(dtime, reserve_carbon_talloc_mavg(ic), veg_pool_mt(ix_reserve,ixC,ic), &
        &                                     mavg_period_hlp, do_calc=l_growing_season(ic))
      reserve_nitrogen_talloc_mavg(ic)   = calc_time_mavg(dtime, reserve_nitrogen_talloc_mavg(ic), &
        &                                     veg_pool_mt(ix_reserve,ixN,ic), mavg_period_hlp, do_calc=l_growing_season(ic))
      reserve_phosphorus_talloc_mavg(ic) = calc_time_mavg(dtime, reserve_phosphorus_talloc_mavg(ic),  &
        &                                     veg_pool_mt(ix_reserve,ixP,ic), mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      ! avoid division by zero
      IF (w_soil_root_fc(ic) > eps8) THEN
        hlp_r = wtr_rootzone(ic) / w_soil_root_fc(ic)
      ELSE
        hlp_r = 0._wp
      END IF
      w_root_lim_talloc_mavg(ic)           = calc_time_mavg(dtime, w_root_lim_talloc_mavg(ic), hlp_r, &
        &                                                   mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !>  2.8 tgrowth / talloc_dynamic
    !>
    SELECT CASE (TRIM(dsl4jsb_Config(VEG_)%biomass_alloc_scheme))
    CASE ("optimal", "fixed")
      mavg_period_hlp = mavg_period_tgrowth
    CASE ("dynamic")
      mavg_period_hlp = mavg_period_talloc_dynamic
    END SELECT

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF(growth_req_n(ic) > 0._wp) THEN
        hlp_r = 1._wp / growth_req_n(ic)
      ELSE
        hlp_r = 0._wp
      END IF
      growth_cn_talloc_mavg(ic) = calc_time_mavg(dtime, growth_cn_talloc_mavg(ic), hlp_r, &
        &                                        mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF(growth_req_n(ic) > 0._wp .AND. growth_req_p(ic) > 0._wp) THEN
        hlp_r = 1._wp / (growth_req_n(ic) * growth_req_p(ic))
      ELSE
        hlp_r = 0._wp
      END IF
      growth_cp_talloc_mavg(ic) = calc_time_mavg(dtime, growth_cp_talloc_mavg(ic), hlp_r, &
        &                                        mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF(growth_req_p(ic) > 0._wp) THEN
        hlp_r = 1._wp / growth_req_p(ic)
      ELSE
        hlp_r = 0._wp
      END IF
      growth_np_talloc_mavg(ic) = calc_time_mavg(dtime, growth_np_talloc_mavg(ic), hlp_r, &
        &                                        mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r)
    DO ic = 1, nc
      IF(labile_nitrogen_talloc_mavg(ic) > 0._wp .AND. labile_phosphorus_talloc_mavg(ic) > 0._wp) THEN
        hlp_r = growth_req_p_tuptake_mavg(ic) * (labile_nitrogen_talloc_mavg(ic)/labile_phosphorus_talloc_mavg(ic))
      ELSE
        hlp_r = 0._wp
      END IF
      growth_p_limit_based_on_n_mavg(ic) = &
        & calc_time_mavg(dtime, growth_p_limit_based_on_n_mavg(ic), hlp_r, mavg_period_hlp, do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !>  2.9 various lctlib tau
    !>
    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      exudation_c_tmyc_mavg(ic) = calc_time_mavg(dtime, exudation_c_tmyc_mavg(ic), veg_exudation_mt(ixC,ic), &
        &                                        lctlib_tau_mycorrhiza, avg_period_unit='year')
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(hlp_r, l_hlp)
    DO ic = 1, nc
      l_hlp = .FALSE.
      IF (veg_pool_mt(ix_fine_root,ixC,ic) > eps8) THEN
        hlp_r   = veg_pool_mt(ix_leaf,ixC,ic) / veg_pool_mt(ix_fine_root,ixC,ic)
        IF ((veg_pool_mt(ix_leaf,ixC,ic) > eps8)) THEN
          l_hlp = .TRUE.
        END IF
      ELSE
        hlp_r   = 0._wp
      END IF

      leaf2root_troot_mavg(ic)         = calc_time_mavg(dtime, leaf2root_troot_mavg(ic), hlp_r, &
        &                                               lctlib_tau_fine_root, do_calc=l_hlp, avg_period_unit='year')
      unit_npp_troot_mavg(ic)          = calc_time_mavg(dtime, unit_npp_troot_mavg(ic), unit_npp(ic), &
        &                                               lctlib_tau_fine_root, do_calc=l_growing_season(ic), avg_period_unit='year')
      unit_uptake_n_pot_troot_mavg(ic) = calc_time_mavg(dtime, unit_uptake_n_pot_troot_mavg(ic), unit_uptake_n_pot(ic), &
        &                                               lctlib_tau_fine_root, do_calc=l_growing_season(ic), avg_period_unit='year')
      unit_uptake_p_pot_troot_mavg(ic) = calc_time_mavg(dtime, unit_uptake_p_pot_troot_mavg(ic), unit_uptake_p_pot(ic), &
        &                                               lctlib_tau_fine_root, do_calc=l_growing_season(ic), avg_period_unit='year')
      fmaint_rate_troot_mavg(ic)       = calc_time_mavg(dtime, fmaint_rate_troot_mavg(ic), fmaint_rate_root(ic), &
                                                        lctlib_tau_fine_root, do_calc=l_growing_season(ic), avg_period_unit='year')

      !>  2.10 tvegdyn (averages at the vegetation dynamics time-scale)
      !>
      an_boc_tvegdyn_mavg(ic)     = calc_time_mavg(dtime, an_boc_tvegdyn_mavg(ic), net_assimilation_boc(ic), mavg_period_monthly, &
        &                                          do_calc=(veg_pool_mt(ix_leaf,ixC,ic) > eps8))
      net_growth_tvegdyn_mavg(ic) = calc_time_mavg(dtime, net_growth_tvegdyn_mavg(ic), net_growth(ic), mavg_period_tvegdyn)
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1) &
    !$ACC   PRIVATE(l_hlp)
    DO ic = 1, nc
      IF (lai(ic) > 0.1_wp) THEN
        l_hlp = .TRUE.
      ELSE
        l_hlp = .FALSE.
      END IF
      lai_tvegdyn_mavg(ic) = calc_time_mavg(dtime, lai_tvegdyn_mavg(ic), lai(ic), mavg_period_tvegdyn, do_calc=l_hlp)
    END DO
    !$ACC END PARALLEL LOOP

    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      rfr_ratio_boc_tvegdyn_mavg(ic) = calc_time_mavg(dtime, rfr_ratio_boc_tvegdyn_mavg(ic), rfr_ratio_boc(ic), &
        &                               mavg_period_tvegdyn, do_calc=(veg_pool_mt(ix_leaf,ixC,ic) > eps8))
    END DO
    !$ACC END PARALLEL LOOP

    !>  2.11 tacclim (averages at the respiration acclimation time-scale)
    !>    t_air_tacclim_mavg(:) is init with 283.15 and only updated if flag_t_resp_acclimation is true
    !>    this IF statement has a direct impact on mo_q_veg_respiration:temperature_response_respiration
    !>
    IF (dsl4jsb_Config(Q_ASSIMI_)%flag_t_resp_acclimation) THEN
      !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
      DO ic = 1, nc
        t_air_tacclim_mavg(ic)       = calc_time_mavg(dtime, t_air_tacclim_mavg(ic), t_air(ic), mavg_period_tacclim)
        t_soil_root_tacclim_mavg(ic) = calc_time_mavg(dtime, t_soil_root_tacclim_mavg(ic), t_soil_root(ic), mavg_period_tacclim)
      END DO
      !$ACC END PARALLEL LOOP
    END IF

    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      beta_sinklim_ps_tacclim_mavg(ic)  = calc_time_mavg(dtime, beta_sinklim_ps_tacclim_mavg(ic), beta_sinklim_ps(ic), &
                                                      mavg_period_tacclim,  do_calc=l_growing_season(ic))
    END DO
    !$ACC END PARALLEL LOOP

    !>  2.12 averages with a weekly, monthly timescale
    !>
    !$ACC PARALLEL LOOP GANG VECTOR DEFAULT(PRESENT) ASYNC(1)
    DO ic = 1, nc
      t_air_week_mavg(ic)  = calc_time_mavg(dtime, t_air_week_mavg(ic), t_air(ic), mavg_period_weekly)
      t_air_month_mavg(ic) = calc_time_mavg(dtime, t_air_month_mavg(ic), t_air(ic), mavg_period_monthly)
      t_jmax_opt_mavg(ic)  = calc_time_mavg(dtime, t_jmax_opt_mavg(ic), t_jmax_opt_daytime(ic), mavg_period_weekly)
    END DO
    !$ACC END PARALLEL LOOP
    !$ACC WAIT(1)
    !$ACC EXIT DATA DELETE(l_growing_season_cl)
    DEALLOCATE(l_growing_season_cl)

    !$ACC END DATA

  END SUBROUTINE calculate_time_average_vegetation


  ! ======================================================================================================= !
  !>
  !> Determine if carbon is conserved
  !>
  SUBROUTINE test_carbon_conservation(tile, options)
    USE mo_jsb_class,             ONLY: Get_model
    USE mo_jsb_tile_class,        ONLY: t_jsb_tile_abstract
    USE mo_jsb_task_class,        ONLY: t_jsb_task_options
    USE mo_jsb_model_class,       ONLY: t_jsb_model
    USE mo_jsb_process_class,     ONLY: VEG_, SB_, HYDRO_, L2A_
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Use_memory(L2A_)
    dsl4jsb_Use_memory(VEG_)
    dsl4jsb_Use_memory(SB_)
    dsl4jsb_Use_memory(HYDRO_)
    dsl4jsb_Use_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    CLASS(t_jsb_tile_abstract), INTENT(inout)     :: tile         !< one tile with data structure for one lct
    TYPE(t_jsb_task_options),   INTENT(in)        :: options      !< model options
    ! ----------------------------------------------------------------------------------------------------- !
    TYPE(t_jsb_model),        POINTER :: model                  !< the model
    INTEGER                           :: iblk, ics, ice, nc     !< dimensions
    REAL(wp)                          :: dtime                  !< timestep length
    REAL(wp), DIMENSION(options%nc)   :: q_total_c_new          !< helper variable
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':test_carbon_conservation'
    ! ----------------------------------------------------------------------------------------------------- !
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Def_memory(L2A_)
    dsl4jsb_Def_memory(VEG_)
    dsl4jsb_Def_memory(SB_)
    dsl4jsb_Def_memory(HYDRO_)
    dsl4jsb_Def_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Real2D_onChunk :: veg_pool_total_c
    dsl4jsb_Real2D_onChunk :: veg_products_total_c
    dsl4jsb_Real2D_onChunk :: net_biosphere_production
    dsl4jsb_Real2D_onChunk :: q_c_conservation_test
    dsl4jsb_Real2D_onChunk :: q_total_c
    dsl4jsb_Real3D_onChunk :: sb_pool_total_c
    dsl4jsb_Real3D_onChunk :: soil_depth_sl
    ! ----------------------------------------------------------------------------------------------------- !
    iblk    = options%iblk
    ics     = options%ics
    ice     = options%ice
    nc      = options%nc
    dtime   = options%dtime
    ! ----------------------------------------------------------------------------------------------------- !
    IF (.NOT. tile%Is_process_calculated(L2A_)) RETURN
    ! ----------------------------------------------------------------------------------------------------- !
    model  => Get_model(tile%owner_model_id)
    ! ----------------------------------------------------------------------------------------------------- !
    IF (debug_on() .AND. iblk == 1) CALL message(TRIM(routine), 'Starting on tile '//TRIM(tile%name)//' ...')
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_memory(L2A_)
    dsl4jsb_Get_memory(VEG_)
    dsl4jsb_Get_memory(SB_)
    dsl4jsb_Get_memory(HYDRO_)
    dsl4jsb_Get_config(VEG_)
    ! ----------------------------------------------------------------------------------------------------- !
    dsl4jsb_Get_var2D_onChunk(VEG_, veg_pool_total_c)
    dsl4jsb_Get_var2D_onChunk(VEG_, net_biosphere_production)
    dsl4jsb_Get_var2D_onChunk(L2A_, q_c_conservation_test)
    dsl4jsb_Get_var2D_onChunk(L2A_, q_total_c)
    dsl4jsb_Get_var3D_onChunk(SB_, sb_pool_total_c)
    dsl4jsb_Get_var3D_onChunk(HYDRO_, soil_depth_sl)
    ! ----------------------------------------------------------------------------------------------------- !

    ! vegetation and soil carbon
    q_total_c_new(:) = veg_pool_total_c(:) + SUM(sb_pool_total_c(:,:) * soil_depth_sl(:,:), DIM = 2)

    IF(dsl4jsb_Config(VEG_)%l_use_product_pools) THEN
      dsl4jsb_Get_var2D_onChunk(VEG_, veg_products_total_c)
      q_total_c_new(:) = q_total_c_new(:) + veg_products_total_c(:)
    END IF

    q_c_conservation_test(:) = q_total_c_new - q_total_c - (net_biosphere_production * dtime / 1000000.0_wp)
    q_total_c(:) = q_total_c_new(:)

  END SUBROUTINE test_carbon_conservation

#endif
END MODULE mo_veg_util
