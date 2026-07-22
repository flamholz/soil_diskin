!> Driver that calls the unmodified JSBACH yasso routine once and writes the
!> resulting pools to yasso_output.csv, so that the python implementation in
!> soil_diskin.compartmental_models.JSBACH can be checked against it.
!>
!> The setup mirrors tests/test_JSBACH_model.py::test_comparison_with_fortran:
!> every pool starts at 1 mol(C)/m2, the climate is held at 25 C and 1 m/a of
!> precipitation, and the daily litter input is 1 mol(C)/m2/day of NPP split
!> the way JSBACH splits it. yasso is called for non-woody litter only
!> (WoodLitterSize = 0), which is what the first 9 entries of the python state
!> vector represent.
PROGRAM test_yasso_call

  USE mo_kind,                   ONLY: wp
  USE mo_carbon_constants,       ONLY: fract_green_aboveGround, sec_per_year
  USE mo_jsb_physical_constants, ONLY: tmelt
  USE mo_carbon_process,         ONLY: yasso

  IMPLICIT NONE

  ! --- forcing, matching the python test -------------------------------------
  REAL(wp), PARAMETER :: temp_celsius   = 25.0_wp   ! air temperature      [C]
  REAL(wp), PARAMETER :: precip_m_per_a = 1.0_wp    ! precipitation        [m/a]
  REAL(wp), PARAMETER :: npp_daily      = 1.0_wp    ! NPP to litter        [mol(C)/m2/day]
  REAL(wp), PARAMETER :: initial_pool   = 1.0_wp    ! every pool starts at [mol(C)/m2]

  ! NPP is split into a woody, an exudate and a green (= everything else) part.
  ! Only the green part and the exudates feed the non-woody yasso pools.
  REAL(wp), PARAMETER :: fract_npp_2_woodPool = 0.3_wp
  REAL(wp), PARAMETER :: fract_npp_2_exudates = 0.05_wp

  ! Chemical composition of leaf litter (acid, water, ethanol, nonsoluble, humus);
  ! lctlib_nlct21.def values for the AWEN split of green litter.
  REAL(wp), PARAMETER :: LeafLit_coef(5) = &
    & [0.4651_wp, 0.304_wp, 0.0942_wp, 0.1367_wp, 0.0_wp]

  ! --- yasso arguments -------------------------------------------------------
  REAL(wp) :: Yasso_io_pools(9)
  REAL(wp) :: Weather(2)
  REAL(wp) :: Yasso_out(18)
  REAL(wp) :: litter, NPP_2_rootExudates, WoodLitterSize

  CHARACTER(len=24), PARAMETER :: pool_names(18) = [character(len=24) :: &
    & 'c_acid_ag', 'c_water_ag', 'c_ethanol_ag', 'c_nonsoluble_ag',      &
    & 'c_acid_bg', 'c_water_bg', 'c_ethanol_bg', 'c_nonsoluble_bg',      &
    & 'c_humus',                                                        &
    & 'soilResp_rateYasso', 'Cflx_2_humusAG', 'Cflx_2_humusBG',         &
    & 'Cflx_humus_2_atmos', 'soilResp_rateLitterAG',                    &
    & 'soilResp_rateLitterBG', 'd_litter_green',                        &
    & 'humus_loss_diag', 'humus_gain_diag' ]

  INTEGER :: i, unit_out

  ! Every yasso pool starts at 1 mol(C)/m2, as X = np.ones(18) does in python.
  Yasso_io_pools(:) = initial_pool

  ! yasso wants air temperature in K and precipitation in mm/s.
  Weather(1) = temp_celsius + tmelt
  Weather(2) = precip_m_per_a * 1000.0_wp / sec_per_year

  ! Non-woody call: no litter size effect on the decomposition rates.
  WoodLitterSize = 0.0_wp

  litter             = npp_daily * (1.0_wp - fract_npp_2_woodPool - fract_npp_2_exudates)
  NPP_2_rootExudates = npp_daily * fract_npp_2_exudates

  CALL yasso(Yasso_io_pools, Weather, litter, LeafLit_coef, WoodLitterSize, &
    &        Yasso_out, fract_green_aboveGround, NPP_2_rootExudates)

  WRITE(*,'(A)') '--- yasso output after one daily time step ---'
  DO i = 1, 18
    WRITE(*,'(A25,ES24.16)') pool_names(i), Yasso_out(i)
  END DO

  OPEN(newunit=unit_out, file='yasso_output.csv', status='replace', action='write')
  WRITE(unit_out,'(A)') 'Name,Value'
  DO i = 1, 18
    WRITE(unit_out,'(A,A1,ES24.16)') TRIM(pool_names(i)), ',', Yasso_out(i)
  END DO
  CLOSE(unit_out)

  WRITE(*,'(A)') 'Wrote yasso_output.csv'

END PROGRAM test_yasso_call
