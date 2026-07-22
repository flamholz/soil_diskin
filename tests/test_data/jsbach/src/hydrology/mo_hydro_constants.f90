!> Contains constants used in the hydrology processes
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
!>#### Constants and parameters used in the hydrology process
!>
!> The module defines physical parameters needed in hydrological parametrisations as well as more
!> technical parameters e.g. to ease the handling of namelist parameters.
!>
!> _References_
!>
!
! TODO: alphabetic? more links? correct? move to other moduel?
!
!> Parametrization of snow fraction
!>
!> - A. Roesch et al. Climate Dynamics (2001) 17: 933-947
!>
!> Parametrization of runoff/drainage
!>
!> - Todini, E., The ARNO rainfall-runoff model, Journal of Hydrology 175 (1996) 339-382,
!>   [DOI](https://doi.org/10.1016/S0022-1694(96)80016-30).
!> - Duemenil, L., & Todini, E. (1992). A rainfall-runoff scheme for use in the Hamburg climate model.
!>   In J. P. O'Kane (Ed.), Advances in Theoretical Hydrology: A tribute to James Dooge (pp. 129-157).
!>   Amsterdam: Elsevier Science Publishers B.V.
!>
!> Parametrization of canopy conductance/resistance
!>
!> - The ECHAM 3 Atmospheric General Circulation Model, technical report No.6,
!>   Deutsches Klimarechnezentrum (1994)
!>
!> Parametrization of organic soil components
!>
!> - Letts, M.G. et al. (2000): Parametrization of peatland hydraulic properties for the Canadian
!>   land surface scheme. Atmosphere-Ocean, 38(1), 141-160, [DOI](https://doi.org/10.1080/07055900.2000.9649643).
!> - Chadburn, S.E., et al. (2022): A new approach to simulate peat accumulation, degradation and
!>   stability in a global land surface scheme (JULES vn5. 8_accumulate_soil) for northern and temperate
!>   peatlands, Geoscientific Model Development 15.4, 1633-1657.
!>
!> Hydaulic conductivity
!>
!> - Clapp, R.B. and Hornberger, G.M. (1978): Empirical equations for some soil hydraulic properties.
!>   Water Resour. Res. 14(4): 601-604, [DOI](https://doi.org/10.1029/WR014i004p00601).
!> - Freeze, R.A. and Cherry, J.A. (1979): Groundwater, ISBN: 0-13-365312-9
!> - Van Genuchten, M. T. (1980): A closed-form equation for predicting the hydraulic conductivity of
!>   unsaturated soils. Soil Sci. Soc. Am. J. 44: 892-898.
!> - Brooks, R.H. and Corey, A.T. (1964): Hydraulic properties of porous media. Hydrology Paper No.3,
!>   Civil Engineering Department, Colorado State University, Fort Collins, CO.
!> - Campbell, G. S.: A simple method for determining unsaturated conductivity from moisture retention data,
!>   Soil Science, 117, 311-314, 1974, [DOI](http://dx.doi.org/10.1097/00010694-197406000-00001).
!> - Olsen, K.W. et al. (2013): Technical Description of version 4.5 of the Community Land Model (CLM),
!>   [DOI](https://doi.org/10.5065/D6RR1W7M).
!> - Niu, G. and Yang, Z. (2006): Effects of Frozen Soil on Snowmelt Runoff and Soil Water Storage at a
!>   Continental Scale. J. Hydrometeor., 7, 937-952, [DOI](https://doi.org/10.1175/JHM538.1).
!>
MODULE mo_hydro_constants
#ifndef __NO_JSBACH__

  USE mo_kind, ONLY: wp

  IMPLICIT NONE
  PUBLIC

  INTEGER, PARAMETER :: &
    ! Identifiers used for the pedotransfer functions of a specific soil hydrology model
    & VanGenuchten_     = 1,                & !< Identifier; compare [[t_hydro_config:soilhydmodel]]
    & BrooksCorey_      = 2,                & !< Identifier; compare [[t_hydro_config:soilhydmodel]]
    & Campbell_         = 3,                & !< Identifier; compare [[t_hydro_config:soilhydmodel]]

    ! Identifiers used for the interpolation scheme at soil layer interfaces
    & Upstream_         = 1,                & !< Identifier; compare [[t_hydro_config:interpol_mean]]
    & Arithmetic_       = 2,                & !< Identifier; compare [[t_hydro_config:interpol_mean]]

    ! Identifiers used for the pond size scaling with depth
    & Quad_             = 1,                & !< Identifier; compare [[t_hydro_config:pond_dynamics]]
    & Tanh_             = 2,                & !< Identifier; compare [[t_hydro_config:pond_dynamics]]

    ! Identifiers used for the scale of hydrological parametrizations
    & Semi_Distributed_ = 1,                & !< Identifier; compare [[t_hydro_config:hydro_scale]]
    & Uniform_          = 2                   !< Identifier; compare [[t_hydro_config:hydro_scale]]

  REAL(wp), PARAMETER :: &
    ! Parameters used to calculate snow fraction from snow amount
    & wsn2fract_eps    = 1.E-12_wp,        & !< Used for numerical reasons (Roesch et al. 2001, Eq. 6)
    & wsn2fract_sigfac = 0.15_wp,          & !< Factor to approximate sub-grid scale slopes (Roesch et al. 2001, Eq. 7)
    & wsn2fract_const  = 0.95_wp,          & !< Factor for snow fraction (Roesch et al. 2001, Eq. 6 and 7)

    ! Canopy interception
    & InterceptionEfficiency = 0.25_wp,    & !< Efficiency of precipitation interception (rain and snow)

    & Epar                   = 2.2E5_wp,   & !TODO: remove, not used in HYDRO
    & SoilReflectivityParMin = 0.0_wp,     & !TODO: remove, not used in HYDRO

    & FcMax = 1.0_wp,                      & !TODO: remove, not used in HYDRO
    & FcMin = 1.E-3_wp,                    & !TODO: remove, not used in HYDRO
    & ZenithMinPar = 1.E-3_wp,             & !TODO: remove, not used in HYDRO

    ! Parameters required with ARNO Scheme - compare Todini (1996)
    & oro_var_min = 100._wp,               & !< ARNO Scheme minimum orographic standard deviation threshold
    & oro_var_max = 1000._wp,              & !< ARNO Scheme maximum orographic standard deviation threshold
    & drain_min   = 0.001_wp / (3600._wp*1000._wp), & !< maximum flux for slow subsurface drainage, i.e. below
                                                      !< field capacity (Todini, 1996; Duemenil, 1992)
    & drain_max   = 0.1_wp   / (3600._wp*1000._wp), & !< maximum flux for fast subsurface drainage, i.e. above
                                                      !< field capacity (Todini, 1996; Duemenil, 1992)
    & drain_exp   = 1.5_wp,                & !< drainage scaling exponent for ARNO scheme

    ! Parameters used for pond computations
    & oro_crit = 100._wp,                  & !< Reference topographic standard deviation for pond scaling factor [m]

    ! Parameters for the computation of canopy conductance/resistance using Eq. 3.3.2.12 in ECHAM3 manual
    & conductance_k = 0.9_wp,              & !< Parameter for canopy conductance/resistance []
    & conductance_a = 5000._wp,            & !< Parameter for conductance/resistance [Jm-3]
    & conductance_b = 10._wp,              & !< Parameter for canopy conductance/resistance [Wm-2]
    & conductance_c = 100._wp,             & !< Parameter for conductance/resistance [ms-1]

    ! Parameters for organic soil component - top layer
    & vol_porosity_org_top    = 0.95_wp,       & !< Volumetric porosity of top organic layer [m/m]
    & vol_field_cap_org_top   = 0.95_wp,       & !< Volumetric field capacity of top organic layer [m/m]
    & vol_p_wilt_org_top      = 0.255_wp,      & !< Volumetric wilting point of top organic layer [m/m]
    & vol_wres_org_top        = 0.050_wp,      & !< Volumetric residual water content of top organic
                                                 !< layer [m/m] (Letts et al., 2000)
    & hyd_cond_sat_org_top    = 0.0001_wp,     & !< Saturated hydraulic conductivity of organic part of
                                                 !< top soil layer [m/s]
    & bclapp_org_top          = 4._wp,         & !< Exponent b in Clapp and Hornberger of organic part
                                                 !< of top soil layer []
    & matric_pot_org_top      = -0.1_wp,       & !< Soil matric potential of top organic layer [m]
                                                 !< (values (roughy) from Chadburn 2022)
    & pore_size_index_org_top = 0.7_wp,        & !< Pore size distribution index of top organic layer []

    ! Parameters for organic soil component - deeper layers
    & vol_porosity_org_below  = 0.82_wp,       & !< Volumetric porosity of deep organic layers [m/m]
    & vol_field_cap_org_below = 0.82_wp,       & !< Volumetric field capacity of deep organic layers [m/m]
    & vol_p_wilt_org_below    = 0.255_wp,      & !< Volumetric wilting point of deep organic layers [m/m]
    & vol_wres_org_below      = 0.150_wp,      & !< Volumetric residual water content of deep organic
                                                 !< layers [m/m] (Letts et al., 2000)
    & hyd_cond_sat_org_below  = 0.00000001_wp, & !< Saturated hydraulic conductivity of deep organic layers [m/s]
    & bclapp_org_below        = 8._wp,         & !< Exponent b in Clapp and Hornberger of deep organic layers []
    & matric_pot_org_below    = -1.0_wp,       & !< Soil matric potential of deep organic layers [m]
                                                 !< (values (roughy) from Chadburn 2022)
    & pore_size_index_org_below = 0.7_wp,      & !< Pore size distribution index of deep organic layers []

    ! Parameters for percolation theory (Oleson et al., 2013)
    & thresh_org              = 0.5_wp,        & !< Threshold of organic material above which connected
                                                 !< flow pathways form (Oleson et al., 2013, p. 162)
    & beta_perc               = 0.139_wp,      & !< Parameter from percolation theory (Oleson et al., 2013, p. 162)

    ! Parameter for bedrock
    & k_brock                 = 1.E-6_wp         !< Hydraulic conductivity of fractured bedrock [m/s] (Freeze & Cherry, 1979)

#endif
END MODULE mo_hydro_constants
