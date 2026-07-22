!> Routines for physical schemes
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
!>#### Contains some routines for physical schemes
!>
MODULE mo_phy_schemes
#ifndef __NO_JSBACH__

  USE mo_kind, ONLY: wp
  USE mo_jsb_convect_tables_iface, ONLY: tlucua, jptlucu1, jptlucu2
  USE mo_jsb_thermo_iface, ONLY: specific_humidity, sat_pres_water, sat_pres_ice, potential_temperature
  USE mo_jsb_surface_exchange_iface, ONLY: sfc_exchange_coefficients
  USE mo_jsb_control, ONLY: acc_stream

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: specific_humidity, qsat_water, qsat_ice, sat_pres_water, &
    & update_drag, tlucua, jptlucu1, jptlucu2, &
    & q_effective, surface_dry_static_energy, heat_transfer_coef, &
    & thermal_radiation, lwnet_from_lwdown, potential_temperature, &
    & sfc_exchange_coefficients
#ifndef __NO_QUINCY__
  PUBLIC :: calc_peaked_arrhenius_function
#endif

    ! Note: nvhpc doesn't support function pointers
    ! & register_exchange_coefficients_procedure, exchange_coefficients, registered_exchange_coefficients_procedure

  ! ABSTRACT INTERFACE
  !   PURE SUBROUTINE i_exchange_coefficients_procedure(           &
  !     & dz,                                                 &
  !     & pqm1, thetam1, mwind, rough_m, theta_sfc, qsat_sfc, &
  !     & km, kh, km_neutral, kh_neutral                      &
  !     & )
  !     IMPORT :: wp

  !     REAL(wp), INTENT(in) :: &
  !       dz,        &
  !       thetam1,   &
  !       pqm1 ,     &
  !       mwind,     &
  !       rough_m,   &
  !       theta_sfc, &
  !       qsat_sfc
  !       !
  !     REAL(wp), INTENT(out) :: &
  !       km,         &
  !       kh,         &
  !       km_neutral, &
  !       kh_neutral

  !   END SUBROUTINE
  ! END INTERFACE

  ! PROCEDURE(i_exchange_coefficients_procedure), POINTER :: registered_exchange_coefficients_procedure => NULL()

  CHARACTER(len=*), PARAMETER :: modname = 'mo_phy_schemes'

CONTAINS

  ! SUBROUTINE register_exchange_coefficients_procedure(exchange_coefficients_procedure)

  !   PROCEDURE(i_exchange_coefficients_procedure) :: exchange_coefficients_procedure

  !   registered_exchange_coefficients_procedure => exchange_coefficients_procedure

  ! END SUBROUTINE register_exchange_coefficients_procedure

  ! SUBROUTINE exchange_coefficients(                       &
  !   & dz,                                                 &
  !   & pqm1, thetam1, mwind, rough_m, theta_sfc, qsat_sfc, &
  !   & km, kh, km_neutral, kh_neutral                      &
  !   & )

  !   REAL(wp), DIMENSION(:), INTENT(in) :: &
  !     dz,        &
  !     thetam1,   &
  !     pqm1 ,     &
  !     mwind,     &
  !     rough_m,   &
  !     theta_sfc, &
  !     qsat_sfc
  !     !
  !   REAL(wp), DIMENSION(:), INTENT(out) :: &
  !     km,         &
  !     kh,         &
  !     km_neutral, &
  !     kh_neutral

  !   INTEGER :: ic, nc

  !   nc = SIZE(dz)

  !   !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(1)
  !   DO ic = 1, nc
  !     CALL registered_exchange_coefficients_procedure(                                &
  !       & dz(ic),                                                                     &
  !       & pqm1(ic), thetam1(ic), mwind(ic), rough_m(ic), theta_sfc(ic), qsat_sfc(ic), &
  !       & km(ic), kh(ic), km_neutral(ic), kh_neutral(ic)                              &
  !       & )
  !   END DO
  !   !$ACC END PARALLEL LOOP

  ! END SUBROUTINE exchange_coefficients

  REAL(wp) FUNCTION qsat_water(temperature, pressure, use_convect_tables)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in) :: temperature      ! Air temperature [K]
    REAL(wp), INTENT(in) :: pressure         ! Pressure [Pa]
    LOGICAL,  INTENT(in) :: use_convect_tables

    IF (use_convect_tables) THEN
      qsat_water = calc_qsat(temperature, pressure)
    ELSE
      qsat_water = specific_humidity(sat_pres_water(temperature), pressure)
    END IF

  END FUNCTION qsat_water

  REAL(wp) FUNCTION qsat_ice(temperature, pressure, use_convect_tables)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in) :: temperature      ! Air temperature [K]
    REAL(wp), INTENT(in) :: pressure         ! Pressure [Pa]
    LOGICAL,  INTENT(in) :: use_convect_tables

    IF (use_convect_tables) THEN
      qsat_ice = calc_qsat(temperature, pressure)
    ELSE
      qsat_ice = specific_humidity(sat_pres_ice(temperature), pressure)
    END IF

  END FUNCTION qsat_ice

  ! Returns saturation specific humidity for given temperature and pressure (at some atmospheric level or at surface)
  ! Uses Eq. 2.27 of Fundamentals of Atmospheric Modeling for the saturated case, but the saturation vapor pressure
  ! of water over a liquid surface resp. ice (see pp 32-34 of Ref.) is computed as in ECHAM5 (mo_convect_tables)
  !
  ! Note: temporary for backward compatibility
  FUNCTION calc_qsat(temperature, pressure) RESULT(qsat)

    !$ACC ROUTINE SEQ

    USE mo_jsb_convect_tables_iface, ONLY: tlucua, &        ! Table for water vapor pressure e_s multiplied by R_d/R_v
      &                                    jptlucu1, jptlucu2
    USE mo_jsb_physical_constants, ONLY: rvd1                    ! = R_v/R_d - 1

    REAL(wp), INTENT(in) :: temperature      !< Air temperature [K]
    REAL(wp), INTENT(in) :: pressure         !< Pressure [Pa]
    REAL(wp)             :: qsat             !< saturation specific humidity

    INTEGER :: it
    REAL(wp) :: tluc

    it = NINT(temperature * 1000._wp)
    ! NOTE a segmentation fault for unplausible surface temperatures occurs (e.g. see mo_jsb4_forcing)
    !      convection tables work within [50,400] Kelvin
    !      unplausible surface temperatures are caught with "seb_check_temperature_range()"
    !      and could be caught in additionally here with:
    !      it = MIN( MAX( NINT(temperature * 1000._wp), 50000), 400000)
    tluc = tlucua(it)

    IF (it >= jptlucu1 .AND. it <= jptlucu2) THEN
      qsat = tluc / (pressure - rvd1*tluc)
    ELSE
      qsat = HUGE(1._wp)
    END IF

  END FUNCTION calc_qsat

  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION q_effective
  !!
  !! out: q_effective
  !-----------------------------------------------------------------------------------------------------
  REAL(wp) FUNCTION q_effective(qsat, qair, fsat, fair)
!NEC$ always_inline

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in) :: &
      & qsat,               & !< Surface saturation specific humidity
      & qair,               & !< Air specific humidity
      & fsat, fair            !< Weighing factors for qsat and qair accounting for only partially wet surface.

    q_effective = fsat * qsat + (1._wp -fair) * qair

  END FUNCTION q_effective

  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION surface_dry_static_energy
  !!
  !! out: surface_dry_static_energy
  !-----------------------------------------------------------------------------------------------------
  REAL(wp) FUNCTION surface_dry_static_energy(t_srf, qsat_srf, cpd_or_cvd, jsb_standalone)

    !$ACC ROUTINE SEQ

    USE mo_jsb_physical_constants, ONLY: &
      & cpvd1        !< cpv/cpd-1

    REAL(wp), INTENT(in) :: &
      & t_srf,                      & !< Surface temperature
      & qsat_srf,                   & !< Surface saturation specific humidity
      & cpd_or_cvd

    LOGICAL, INTENT(in) :: &
      & jsb_standalone

    IF (jsb_standalone) THEN
      ! Todo: Check if this formulation is valid. We re-implement it here, as it was used up to now in
      !       jsbach:dev, and we do not want to change the model behavior with this merge.
      surface_dry_static_energy = t_srf * cpd_or_cvd * ( 1._wp + cpvd1 * qsat_srf)
    ELSE
      surface_dry_static_energy = t_srf * cpd_or_cvd
    END IF

  END FUNCTION surface_dry_static_energy


  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION thermal_radiation
  !!
  !! out: thermal_radiation
  !-----------------------------------------------------------------------------------------------------
#ifndef _OPENACC
  ELEMENTAL &
#endif
  REAL(wp) FUNCTION thermal_radiation(t_srf)

    USE mo_jsb_physical_constants, ONLY: &
      stbo,       &  !< Stefan-Boltzmann constant
      zemiss_def     !< Surface emissivity

    REAL(wp), INTENT(in) :: t_srf

    thermal_radiation = stbo * zemiss_def * t_srf**4._wp

  END FUNCTION thermal_radiation


  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION lwnet_from_lwdown
  !!
  !! out: lwnet_from_lwdown
  !-----------------------------------------------------------------------------------------------------
  REAL(wp) FUNCTION lwnet_from_lwdown(lwdown, t_srf)

    !$ACC ROUTINE SEQ

    USE mo_jsb_physical_constants, ONLY: &
      stbo,       &  !< Stefan-Boltzmann constant
      zemiss_def     !< Surface emissivity

    REAL(wp), INTENT(in) :: lwdown   ! downward longwave radiation
    REAL(wp), INTENT(in) :: t_srf    ! surface temperature

    lwnet_from_lwdown = zemiss_def * (lwdown - stbo * t_srf**4._wp)

  END FUNCTION lwnet_from_lwdown


  !-----------------------------------------------------------------------------------------------------
  !> FUNCTION heat_transfer_coef
  !!
  !! out: heat_transfer_coef
  !-----------------------------------------------------------------------------------------------------
  REAL(wp) FUNCTION heat_transfer_coef(drag, dtime, alpha)
!NEC$ always_inline

    !$ACC ROUTINE SEQ

    USE mo_jsb_physical_constants, ONLY: grav

    REAL(wp), INTENT(in) :: &
      & drag,      &
      & dtime,   &
      & alpha

    heat_transfer_coef = drag / (alpha * grav * dtime)

  END FUNCTION heat_transfer_coef

  ! ======================================================================================================= !
  !> Calculates temperature response factor for biological processes
  !>   according to the peaked Arrhenius equation as defined in:
  !>   Medlyn et al. 2002, PCE, eq. 18
  !>
  !>   NB if sensitivity of zero is specified, 1 will be returned
  !>
  !>  Input: temperature, (de-)activiation energy, optimum temperature
  !>
  !>  Output: rate modifier (unitless)
  !>
#ifndef __NO_QUINCY__
  ELEMENTAL FUNCTION calc_peaked_arrhenius_function(temp, Ea, Ed, temp_opt) RESULT(rate_modifier)

    USE mo_jsb_physical_constants,  ONLY: r_gas, Tzero
    USE mo_jsb_math_constants,      ONLY: eps8
    USE mo_sb_constants,            ONLY: temp_ref_tresponse, temp_freeze_thres_bio, temp_freeze_window_bio

    !------------------------------------------------------------------------------------------------------ !
    REAL(wp), INTENT(in)  :: temp           !< temperature [K]
    REAL(wp), INTENT(in)  :: Ea             !< activation energy [J mol-1]
    REAL(wp), INTENT(in)  :: Ed             !< deactivation energy [J mol-1]
    REAL(wp), INTENT(in)  :: temp_opt       !< optimum temperature [K]
    REAL(wp)              :: rate_modifier  !< temperature rate modifier [unitless]
    REAL(wp)              :: frost_modifier !< reduction of rate modifier when approaching -2 degC [-]
                                            !!   only for biological processes
    !------------------------------------------------------------------------------------------------------ !
    REAL(wp)              :: hlp1, hlp2
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//':calc_peaked_arrhenius_function'

    IF (Ea > eps8) THEN
      hlp1            = temp - temp_opt
      hlp2            = temp * temp_opt * r_gas
      frost_modifier  = MIN(MAX((temp - (Tzero - temp_freeze_thres_bio)) / temp_freeze_window_bio, 0.0_wp), 1.0_wp)
      rate_modifier   = (Ed * EXP(Ea * hlp1 / hlp2)) / (Ed - Ea * (1._wp - EXP(Ed * hlp1 / hlp2)))
      rate_modifier   = rate_modifier * frost_modifier
    ELSE
      rate_modifier = 1._wp
    END IF

  END FUNCTION calc_peaked_arrhenius_function
#endif

  ! ======================================================================================================= !
  !>
  !> calculates surface drag and exchange coefficients for jsbach standalone simulation
  !>
  SUBROUTINE update_drag(nc, time_step_len, &
      & config_model_scheme, &
      & temp_air, pressure, qair, wind, &
      & t_srf_proc, fact_q_air_proc, fact_qsat_srf_proc, rough_h_srf_proc, rough_m_srf_proc, &
      & height_wind, height_humidity, coef_ril_tm1, coef_ril_t, coef_ril_tp1, &
      & drag_srf, ddrag_srf, t_acoef, t_bcoef, q_acoef, q_bcoef, pch, &
      & veg_height, t_srf_upd, zril_old)

    USE mo_jsb_model_class,        ONLY: MODEL_QUINCY, MODEL_JSBACH
    USE mo_jsb_physical_constants, ONLY: grav, rd, cpd, rvd1, cpvd1, von_karman, rd_o_cpd

    INTEGER,  INTENT(IN)  :: nc
    REAL(wp), INTENT(IN)  :: time_step_len
    INTEGER,  INTENT(IN)  :: config_model_scheme
    REAL(wp), INTENT(IN)  :: temp_air(:)
    REAL(wp), INTENT(IN)  :: pressure(:)
    REAL(wp), INTENT(IN)  :: qair(:)
    REAL(wp), INTENT(IN)  :: wind(:)
    REAL(wp), INTENT(IN)  :: t_srf_proc(:)             !< Surface temperature
    REAL(wp), INTENT(IN)  :: fact_q_air_proc(:)
    REAL(wp), INTENT(IN)  :: fact_qsat_srf_proc(:)
    REAL(wp), INTENT(IN)  :: rough_h_srf_proc(:)
    REAL(wp), INTENT(IN)  :: rough_m_srf_proc(:)
    REAL(wp), INTENT(in)  :: height_wind               !< Defines lowest layer height-> where wind measurements are taken
    REAL(wp), INTENT(in)  :: height_humidity           !< Defines lowest layer height-> where humidity measurements are taken
    REAL(wp), INTENT(in)  :: coef_ril_tm1              !< Weighting factor for richardson numbers at different steps
    REAL(wp), INTENT(in)  :: coef_ril_t                !<   that are used to calculate a drag coef. that approximates
    REAL(wp), INTENT(in)  :: coef_ril_tp1              !<   the drag coef. at time t, but helps to maintain stability.

    REAL(wp), INTENT(OUT) :: drag_srf(:)               !< Surface drag
    REAL(wp), INTENT(OUT) :: ddrag_srf(:)              !< d(Surface drag)/d(t_srf_proc)
    REAL(wp), INTENT(OUT) :: t_acoef (:)
    REAL(wp), INTENT(OUT) :: t_bcoef (:)
    REAL(wp), INTENT(OUT) :: q_acoef (:)
    REAL(wp), INTENT(OUT) :: q_bcoef (:)
    REAL(wp), INTENT(OUT) :: pch     (:)

    ! Optional variable for QUINCY
    REAL(wp), OPTIONAL, INTENT(IN)    :: veg_height(:)     !< vegetation height
    ! Optional variables for the computation of a filtered Richardson number
    REAL(wp), OPTIONAL, INTENT(IN)    :: t_srf_upd(:)  !< Updated surface temperature
    REAL(wp), OPTIONAL, INTENT(inout) :: zril_old(:)   !< Richardson number at previous time step

    REAL(wp) :: height(nc)
    INTEGER  :: ic
    LOGICAL  :: l_zril_filter
    REAL(wp) :: zcons9, zcons11, zcons12, zsigma
    REAL(wp) :: air_pressure, zdu2, ztvd, ztvir, qsat_surf, qsat_surf_upd, r_exner_surf, qsurf_eff
    REAL(wp) :: ztvs_act, ztvs_upd, zg, zgh, zril, zril_act, zril_upd
    REAL(wp) :: zcons, zchnl, zcfnchl, zcfhl, sfact, zscfl, ufact, zucfhl
    !REAL(wp) :: zcons8, zcdnl, zucfl, zcfncl, zcfml   ! quincy specific

    REAL(wp) :: dqsat_surf, dqsurf_eff, dztvs_act, dzril, dzscfl, dzucfhl, dzcfhl

    REAL(wp), PARAMETER :: cb = 5._wp
    REAL(wp), PARAMETER :: cc = 5._wp
    CHARACTER(len=*), PARAMETER :: routine = modname//':update_drag'

    !$ACC DATA ASYNC(acc_stream) &
    !$ACC   CREATE(height)

    ! in jsb3: rd = GasConstantDryAir; cpd = SpecificHeatDryAirConstPressure; grav = Gravity
    ! zcons8  = 2._wp * cb  ! quincy specific
    zcons9  = 3._wp * cb
    zcons11 = 3._wp * cb * cc
    zcons12 = time_step_len * grav / rd
    zsigma  = 0.99615_wp        ! corresponds to lowest level of 47 layer echam

    IF (PRESENT(t_srf_upd) .AND. PRESENT(zril_old)) THEN
      l_zril_filter = .TRUE.
    ELSE
      l_zril_filter = .FALSE.
    END IF
    IF (PRESENT(veg_height)) THEN
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        height(ic) = veg_height(ic)
      END DO
      !$ACC END PARALLEL LOOP
    ELSE
      !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream)
      DO ic = 1, nc
        height(ic) = 0._wp
      END DO
      !$ACC END PARALLEL LOOP
    END IF

    !$ACC PARALLEL LOOP DEFAULT(PRESENT) GANG VECTOR ASYNC(acc_stream) &
    !$ACC   PRIVATE(qsat_surf, dqsat_surf, qsat_surf_upd, zdu2, air_pressure, ztvd, ztvir)       &
    !$ACC   PRIVATE(r_exner_surf, qsurf_eff, dqsurf_eff, ztvs_act, dztvs_act, ztvs_upd, zg, zgh) &
    !$ACC   PRIVATE(zril, dzril, zril_act, zril_upd, zchnl, sfact, zscfl, dzscfl, ufact, zucfhl) &
    !$ACC   PRIVATE(dzucfhl, zcons, zcfnchl, zcfhl, dzcfhl)
    DO ic = 1, nc
      ! NOTE: a segmentation fault for unplausible surface temperatures (outside [50,400] Kelvin) would occur in "calc_qsat()"
      qsat_surf = calc_qsat(t_srf_proc(ic), pressure(ic))
      dqsat_surf = 1e3_wp * (calc_qsat(t_srf_proc(ic) + 1e-3_wp, pressure(ic)) - qsat_surf)
      IF (l_zril_filter) THEN
        qsat_surf_upd = calc_qsat(t_srf_upd(ic), pressure(ic))
      END IF

      !------------------------------------------------------------------------------------
      ! Approximation of cdrag
      !------------------------------------------------------------------------------------
      ! squared wind shear ! minimum wind speed square from echam/mo_surface_land.f90:precalc_land
      zdu2 = MAX(wind(ic) ** 2, 1._wp)

      ! virtual potential air temperature (see mo_surface_boundary.f90)
      ! according to Saucier, WJ Principles of Meteoroligical Analyses
      ! tv = t * (1 + 0.61 * q )    ! virtual temperature
      ! td = t * ( 1000 / p_mb ) ^ R/cdp  ! potential temperature
      ! tvd = tair * (100000/p_pa)^rd_o_cps * (1 + rvd1 * qair) ! virtual potential temperature
      air_pressure = zsigma * pressure(ic)
      ztvd = ( temp_air(ic) * ( 100000._wp / air_pressure)**rd_o_cpd ) * ( 1._wp + rvd1 * qair(ic))
      ztvir = temp_air(ic) * ( 1._wp + rvd1 * qair(ic) )

      r_exner_surf = ( 100000._wp / pressure(ic))**rd_o_cpd
      qsurf_eff = fact_qsat_srf_proc(ic) * qsat_surf + ( 1._wp - fact_q_air_proc(ic) ) * qair(ic)
      dqsurf_eff = fact_qsat_srf_proc(ic) * dqsat_surf

      ! virtual potential surface temperature for actual and updated states
      ztvs_act = t_srf_proc(ic) * r_exner_surf * (1._wp + rvd1 * qsurf_eff)
      dztvs_act = r_exner_surf * (1._wp + rvd1 * (qsurf_eff + t_srf_proc(ic) * dqsurf_eff))
      IF (l_zril_filter) THEN
        ztvs_upd = t_srf_upd(ic) * ( 100000._wp / pressure(ic))**rd_o_cpd * &
          & ( 1._wp + rvd1 * ( fact_qsat_srf_proc(ic) * qsat_surf_upd + ( 1._wp - fact_q_air_proc(ic) ) &
          & * qair(ic)))
      END IF

      ! geopotential of the surface layer (see echam's auxhybc.f90 & geopot.f90)
      ! adapted according to jsb3 offline modifications by Marvin, Philipp et al. Jan. 24, 2017 (r8893)
      IF(height_wind > 0._wp .AND. height_humidity > 0._wp) THEN
        SELECT CASE (config_model_scheme)
        CASE (MODEL_JSBACH)
          zg  = height_wind * grav
          zgh = height_humidity * grav ! jsb3 "HeightHumidity and HeightTemperature have to be equal"
#ifndef __NO_QUINCY__
        CASE (MODEL_QUINCY)
          zg  = (height_wind + 3._wp / 10._wp * height(ic)) * grav
          zgh = zg
#endif
        END SELECT
      ELSE
        zg = ztvir * rd * LOG(1._wp / zsigma)
        zgh = zg
      ENDIF

      ! Richardson number (dry, Brinkop & Roeckner 1995, Tellus)
      ! ztvd, ztvs are now virtual potential temperatures, changed by Thomas Raddatz 07.2014
      IF (l_zril_filter) THEN
        zril_act = zg * ( ztvd - ztvs_act ) / ( zdu2 * (ztvd + ztvs_act) / 2._wp )
        zril_upd = zg * ( ztvd - ztvs_upd ) / ( zdu2 * (ztvd + ztvs_upd) / 2._wp )

        ! Richardson number needs to be filtered for offline simulations in order to maintain stability
        zril = coef_ril_tm1 * zril_old(ic) &
          &  + coef_ril_t   * zril_act     &
          &  + coef_ril_tp1 * zril_upd
        zril_old(ic) = zril
        dzril = -4._wp * zg / zdu2 * ztvd / (ztvd + ztvs_act)**2 * dztvs_act * coef_ril_t
      ELSE
        SELECT CASE (config_model_scheme)
        CASE (MODEL_JSBACH)
          zril = zg * ( ztvd - ztvs_act ) / ( zdu2 * (ztvd + ztvs_act) / 2._wp )
#ifndef __NO_QUINCY__
        CASE (MODEL_QUINCY)
          zril = MIN(zg * ( ztvd - ztvs_act ) / ( zdu2 * (ztvd + ztvs_act) / 2._wp ), 0.5_wp)
#endif
        END SELECT
      dzril = -4._wp * zg / zdu2 * ztvd / (ztvd + ztvs_act)**2 * dztvs_act
      END IF

      ! Neutral drag coefficient for
      ! momentum
      ! zcdnl = (von_karman / LOG(1._wp + zg / (grav * rough_m_srf_proc(ic) ))) ** 2
      ! heat
      zchnl = von_karman ** 2 / (LOG(1._wp + zg / (grav * rough_m_srf_proc(ic) )) &
                * LOG( ( grav * rough_m_srf_proc(ic) + zgh ) / (grav * rough_h_srf_proc(ic) )))

      ! account for stable/unstable case: helper variables
      ! The stability factors are multiplied by Ri to cancel a singularity in the derivative of zucfhl.
      sfact = SQRT(1._wp + 5._wp * ABS(zril))
      zscfl = zril * sfact
      dzscfl = dzril * (sfact + 2.5_wp * zril / sfact)

      ufact = zcons11 * zchnl * SQRT(ABS(zril) * (1._wp + zgh / (grav * rough_h_srf_proc(ic))))
      ! momentum
      ! zucfl  = 1._wp / (1._wp + zcons11 * zcdnl * SQRT(ABS(zril) * (1._wp + zg  / (grav * rough_m_srf_proc(ic)))))
      ! heat
      zucfhl = - zril / (1._wp + ufact)
      dzucfhl = - dzril * (1._wp + 0.5_wp * ufact) / (1._wp + ufact)**2

      ! ignoring cloud water correction (see mo_surface_land.f90)
      zcons = zcons12 * pressure(ic) / ztvir
      ! momentum
      ! zcfncl   = zcons * SQRT(zdu2) * zcdnl
      ! heat
      zcfnchl  = zcons * SQRT(zdu2) * zchnl

      ! Stable / Unstable case
      IF ( zril > 0._wp ) THEN
        ! momentum
        ! zcfml = zcfncl  / (1._wp + zcons8 * zril / zscfl)
        ! heat
        zcfhl = zcfnchl / (1._wp + zcons9 * zscfl)
        dzcfhl = - zcfnchl * zcons9 * dzscfl / (1._wp + zcons9 * zscfl)**2
      ELSE
        ! zcfml = zcfncl  * (1._wp - zcons8 * zril * zucfl)
        zcfhl = zcfnchl * (1._wp + zcons9 * zucfhl)
        dzcfhl = zcfnchl * zcons9 * dzucfhl
      END IF

      drag_srf(ic) = zcfhl
      ddrag_srf(ic) = dzcfhl
      pch(ic) = zcfhl / zcfnchl * zchnl

      !---------------------------------------------------------------------------------------------------------------
      ! Computation of Richtmeyr-Morton Coefficients
      ! This follows now Jan Polcher's explicit solution, i.e. atmospheric conditions at t+1 are assumed to be valid
      !---------------------------------------------------------------------------------------------------------------
      t_acoef(ic) = 0.0_wp
      t_bcoef(ic) = cpd * (1._wp + cpvd1 * qair(ic)) * temp_air(ic) + zgh
      q_acoef(ic) = 0.0_wp
      q_bcoef(ic) = qair(ic)

    END DO
    !$ACC END PARALLEL LOOP
    !$ACC END DATA

  END SUBROUTINE update_drag

#endif
END MODULE mo_phy_schemes
