!> Stub module for mo_physical_constants_iface 
!> This provides all the physical constants needed by JSBACH modules
MODULE mo_physical_constants_iface
  USE mo_kind, ONLY: wp
  IMPLICIT NONE
  PUBLIC
  
  ! Physical constants from ICON's mo_physical_constants
  REAL(wp), PARAMETER :: &
    & tmelt     = 273.15_wp,      & ! Melting temperature of ice/snow [K]
    & rhoi      = 917.0_wp,       & ! Density of ice [kg/m3]  
    & rhos      = 300.0_wp,       & ! Density of snow [kg/m3]
    & rhoh2o    = 1000.0_wp,      & ! Density of liquid water [kg/m3]
    & argas     = 8.314_wp,       & ! Universal gas constant [J/(mol*K)]
    & rd        = 287.05_wp,      & ! Specific gas constant for dry air [J/(kg*K)]
    & rv        = 461.51_wp,      & ! Specific gas constant for water vapour [J/(kg*K)]
    & alv       = 2.501e6_wp,     & ! Latent heat of vaporization [J/kg]
    & als       = 2.834e6_wp,     & ! Latent heat of sublimation [J/kg]
    & alf       = 3.34e5_wp,      & ! Latent heat of fusion [J/kg]
    & amd       = 28.970_wp,      & ! Molecular weight of dry air [g/mol]
    & amco2     = 44.0095_wp,     & ! Molecular weight of CO2 [g/mol]
    & grav      = 9.80665_wp,     & ! Gravitational acceleration [m/s2]
    & ks        = 2.3_wp,         & ! Thermal conductivity of soil [W/(m*K)]
    & ki        = 2.3_wp,         & ! Thermal conductivity of ice [W/(m*K)]
    & clw       = 4218.0_wp,      & ! Specific heat capacity of liquid water [J/(kg*K)]
    & cs        = 840.0_wp,       & ! Specific heat capacity of soil [J/(kg*K)]
    & ci        = 2106.0_wp,      & ! Specific heat capacity of ice [J/(kg*K)]
    & cvdifts   = 1.0e-5_wp         ! Molecular diffusivity of water vapour [m2/s]

END MODULE mo_physical_constants_iface

!> Stub module for mo_jsb_control 
!> This is a minimal implementation for testing purposes
MODULE mo_jsb_control
  IMPLICIT NONE
  PUBLIC
  
  ! Dummy variable for acc_stream
  LOGICAL :: acc_stream = .FALSE.
  
END MODULE mo_jsb_control

!> Stub module for mo_jsb_varlist
!> This provides the VARNAME_LEN parameter needed by carbon constants
MODULE mo_jsb_varlist
  IMPLICIT NONE
  PUBLIC
  
  INTEGER, PARAMETER :: VARNAME_LEN = 32
  
END MODULE mo_jsb_varlist
