!> Contains structures and methods for the hydrology process configuration
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
!>#### Configuration of the hydrology process
!>
!> The module defines the data structure [[t_hydro_config]] that contains the hydrology
!> configuration, i.e. all hydrology variables that are configurable via namelist.
!> In subroutine [[Init_hydro_config]] we read the hydrology namelist 'jsb_hydro_nml'
!> and set the hydrology configuration data structure accordingly. We also perform
!> some consistency checks.
!>
MODULE mo_hydro_config_class
#ifndef __NO_JSBACH__

  USE mo_exception,          ONLY: message_text, message, finish
  USE mo_jsb_impl_constants, ONLY: WB_IGNORE, WB_LOGGING, WB_ERROR
  USE mo_util,               ONLY: real2string
  USE mo_util_string,        ONLY: tolower
  USE mo_io_units,           ONLY: filename_max
  USE mo_kind,               ONLY: wp
  USE mo_jsb_control,        ONLY: debug_on
  USE mo_jsb_config_class,   ONLY: t_jsb_config

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_hydro_config
  !> The data structure contains all configurable parameters of the HYDRO process. It extends
  !> [[mo_jsb_config_class:t_jsb_config]], a data structure containing basic parameters defined
  !> for all ICON-Land processes.
  !>
  !> _Note for the ford documentation_: The table lists the general parameters of
  !> [[mo_jsb_config_class:t_jsb_config]] as well as the parameters specific to the hydrology
  !> actually defined in [[t_hydro_config]].
  TYPE, EXTENDS(t_jsb_config) :: t_hydro_config
    CHARACTER(len=filename_max) :: bc_sso_filename   !< Name of input file with orographic maps
    CHARACTER(len=filename_max) :: ic_moist_filename !TODO: remove
    REAL(wp)               :: &
      & w_skin_max,           & !< Maximum amount of liquid water in skin reservoirs of soil
                                !< and canopy [m], as well as maximum amount of snow on canopy
                                !< [m water equivalent]
      & w_soil_limit,         & !< Set a limit for the maximum amount of root zone moisture; -1. for
                                !< no limitation
      & w_soil_crit_fract,    & !< If the relative soil moisture drops below this critical fraction
                                !< soil desiccation begins, i.e. water stress
                                !< starts to be greater than zero.
      & w_soil_wilt_fract,    & !< If the relative soil moisture drops below this fraction
                                !< the soil is at its wilting point and no transpiration
                                !< occurs anymore.
      & snow_depth_max,       & !< Limit for snow depth [m water equivalent] used to avoid grid
                                !< cells with infinitely growing snow depth; -1. for no limitation
      & ret_macro_blg,        & !< Assumed retention time for below ground runoff,
                                !< i.e. avg. time before water reaches tributary [s]
      & ret_macro_srf           !< Assumed retention time for surface runoff,
                                !< i.e. avg. time before water reaches tributary [s]

    CHARACTER(len=10)     ::  &
      & scheme_stom_cond        !< Scheme to use for computation of stomatal conductance: 'echam5' is only option
    LOGICAL               ::  &
      & l_read_initial_moist, & !< Force reading initial soil moisture from 'ic_filename' (this namelist),
                                !< although [[t_jsb_model_config:init_from_ifs]]=true.
      & sanitize_restart,     & !< Sanitize water budget in case the restart file has inconsistencies.
      & l_organic,            & !< Calculate soil hydrological parameters depending on organic matter fractions
      & l_socmap,             & !< Read organic matter fractions from three dim. map
      & l_soil_texture,       & !< Calculate soil hydrological parameters from soil textures (silt, clay, sand)
      & l_glac_in_soil,       & !< Treat glacier as part of soil, no separate glacier tile (not used)
                                !TODO: remove?!
      & l_infil_subzero,      & !< Allow infiltration of water into uppermost soil layer at soil temperatures
                                !< below zero degC
                                !TODO: might be removed and set to TRUE as default after the old snow scheme is removed
      & l_ponds                 !< Representation of local depressions that might hold additional surface water
    INTEGER               ::  &
      & soilhydmodel,         & !< Soil hydrological model to calculate soil hydraulic properties:</br>
                                !< 'VanGenuchten' (1): van Genuchten (1980) (default)</br>
                                !< 'BrooksCorey' (2): Brooks and Corey (1964)</br>
                                !< 'Campbell' (3): Campbell (1974)
      & interpol_mean,        & !< Interpolation scheme for hydraulic conductivity (K) and conductivity (C)
                                !< at layer interfaces:</br>
                                !< 'Upstream' (1): for K use value of the upper layer and for D the higher
                                !< value of the two layers (default)</br>
                                !< 'Arithmetic' (2): for K and D calculate arithmetic means of upper and
                                !< lower layer values
      & pond_dynamics,        & !< Pond dynamics scheme (i.e. scaling between of pond area and pond depth):</br>
                                !< 'Quadratic' (1): quadratic scaling </br>
                                !< 'Tanh' (2): scaling with tangent hyperbolicus (default)
      & hydro_scale,          & !< Assumed effective scale of the process calculations</br>
                                !< 'semi_distributed' (1): usage of the ARNO scheme assuming sub-grid scale topographic
                                !<    variations (default)</br>
                                !< 'uniform' (2): assume uniform soil moisture distribution for the tile,
                                !<   e.g. for applications like HydroTiles, HighRes or sitelevel
      & enforce_water_budget    !< Logging level in case of water balance violations in the HYDRO process - overwrites
                                !< [[t_jsb_model_config:enforce_water_budget]] set in namelist 'jsb_model_nml' handling
                                !< water balance issues of all ICON-Land processes.</br>
                                !< default: 'unset': use setting from namelist 'jsb_model_nml'</br>
                                !< 'ignore' (0):  short, one-line warning,</br>
                                !< 'logging' (1): detailed debugging output,</br>
                                !< 'error' (2):   debugging output and model stop

    !! quincy standalone
    INTEGER       :: qs_nsoil_water   !< number of soil layers for soil water calculations (equivalent 'qs_nsoil_energy' in SSE_)

  CONTAINS
    PROCEDURE             :: Init => Init_hydro_config
  END type t_hydro_config

  INTEGER, PARAMETER          :: max_soil_layers = 20  ! for quincy soil-layer calculations

  CHARACTER(len=*), PARAMETER :: modname = 'mo_hydro_config_class'

CONTAINS

  ! ============================================================================================== !
  !>
  !>#### Configuration of the hydrology process
  !>
  !> In this subroutine we configure the hydrolology process based on the [[jsb_hydro_nml]]
  !> parameter settings. Init_hydro_config is a type-bound procedure of [[t_hydro_config]].
  !> It is called during initialization phase from [[mo_jsb_model_class:Configure_model_processes]]
  !> ('CALL this%processes(iproc)%p%Configure()').
  !>
  !> _Note_: For a more extensive documentation of the namelist parameters please check the
  !> configuration data type [[t_hydro_config]].
  !>
  SUBROUTINE Init_hydro_config(config)

    USE mo_jsb_model_class,    ONLY: MODEL_JSBACH, MODEL_QUINCY
    USE mo_jsb_namelist_iface, ONLY: open_nml, POSITIONED, position_nml, close_nml
    USE mo_jsb_grid_class,     ONLY: t_jsb_vgrid, new_vgrid
    USE mo_jsb_grid,           ONLY: Register_vgrid
    USE mo_jsb_io,             ONLY: ZAXIS_DEPTH_BELOW_LAND
    USE mo_jsb_io_netcdf,      ONLY: t_input_file, jsb_netcdf_open_input
    USE mo_hydro_constants,    ONLY: BrooksCorey_, Campbell_, VanGenuchten_, Upstream_, Arithmetic_, &
      &                              Quad_, Tanh_, Semi_Distributed_, Uniform_

    CLASS(t_hydro_config), INTENT(inout) :: config

    !--------------------------------------------------------------------------------------------------------
    ! Namelist parameters
    !
    LOGICAL :: &
      & active,               & !< True: the HYDRO process is activated
      & lrestart_cont           !< True: continue simulations even if HYDRO restart data is missing
    CHARACTER(len=filename_max) :: &
      & ic_filename,          & !< Name of input file with initial conditions
      & bc_filename,          & !< Name of input file with boundary conditions
      & bc_sso_filename         !< Name of input file with orographic boundary conditions
    REAL(wp) :: &
      & w_skin_max,           & !< Maximum filling of skin reservoirs: soil (liquid water [m]) and
                                !< canopy (liquid water [m]) or snow [m water equivalent]
      & snow_depth_max,       & !< Maximum snow depth [m water equivalent]
      & w_soil_limit,         & !< Upper limit for maximum root zone soil water content [m]
      & w_soil_crit_fract,    & !< Relative root zone moisture where water stress starts to matter
      & w_soil_wilt_fract,    & !< Relative root zone moisture at wilting point
      & ret_macro_blg,        & !< Retention time for below ground runoff [s]
      & ret_macro_srf           !< Retention time for surface runoff [s]
    CHARACTER(len=16) :: &
      & scheme_stom_cond,     & !< Scheme to compute stomatal conductance
      & soilhydmodel,         & !< Model to calculate soil hydraulic properties
      & interpol_mean,        & !< Interpolation scheme for vertical diffusivity/conductivity
      & pond_dynamics,        & !< Pond dynamics scheme
      & hydro_scale,          & !< Assumption on effective area represented in calculations
      & enforce_water_budget    !< Warn, stopp or ignore in case of water balance issues
    LOGICAL :: &
      & l_read_initial_moist, & !< True: read initial state of snow and soil moisture although
                                !< init_from_ifs=true (namelist 'jsb_model_nml')
      & sanitize_restart,     & !< True: fix possible inconsistencies of the restart file
      & l_organic,            & !< True: soil properties depend on soil organic matter fractions
      & l_socmap,             & !< True: soil organic matter fraction read from 3-dim. map
      & l_soil_texture,       & !< True: calculate soil properties from soil texture data
      & l_glac_in_soil,       & !< True: glaciers are part of the soil - not used
      & l_infil_subzero,      & !< True: infiltration of surface water below zero degC possible
      & l_ponds                 !< True: represent sub-grid scale surface depressions that might
                                !< form small ponds
    ! quincy
    REAL(wp) :: soil_layer_profile_ubound_estimate(max_soil_layers) !< intial estimate of soil profile using the upper bound of each soil layer [m]
    REAL(wp) :: k_soil_profile  !< for soil layer calculations
    REAL(wp) :: min_layer_depth !< for soil layer calculations
    INTEGER  :: isoil           !< loop over soil layers
    INTEGER  :: qs_nsoil_water  !< quincy standalone: number of soil layers consistent with SPQ_

    NAMELIST /jsb_hydro_nml/        &
      & active,                     &
      & lrestart_cont,              &
      & ic_filename,                &
      & bc_filename,                &
      & bc_sso_filename,            &
      & w_skin_max,                 &
      & w_soil_limit,               &
      & w_soil_crit_fract,          &
      & w_soil_wilt_fract,          &
      & snow_depth_max,             &
      & ret_macro_blg,              &
      & ret_macro_srf,              &
      & scheme_stom_cond,           &
      & l_read_initial_moist,       &
      & sanitize_restart,           &
      & enforce_water_budget,       &
      & l_organic,                  &
      & l_socmap,                   &
      & l_soil_texture,             &
      & l_glac_in_soil,             &
      & l_infil_subzero,            &
      & l_ponds,                    &
      & soilhydmodel,               &
      & interpol_mean,              &
      & pond_dynamics,              &
      & hydro_scale,                &
      & qs_nsoil_water

    INTEGER :: &
      & nml_handler,                   &  !< Handle to deal with namelist file
      & nml_unit,                      &  !< IO unit of the namelist file
      & istat                             !< Status flag

    TYPE(t_input_file) :: input_file      !< Input file
    REAL(wp), POINTER :: ptr_1D(:)        !< Pointer to a one dimensional variable
    REAL(wp), ALLOCATABLE :: depths(:)    !< Upper/lower bounds of soil layers
    REAL(wp), ALLOCATABLE :: mids(:)      !< Mid soil layer depth
    REAL(wp), ALLOCATABLE :: dz_water(:)  !< thickness of soil layers
    INTEGER :: nsoil                      !< Number of soil layers

    TYPE(t_jsb_vgrid), POINTER :: soil_w  !< Vertical grid used for soil layers of HYDRO process

    CHARACTER(len=*), PARAMETER :: routine = modname//':Init_hydro_config'

    IF (debug_on()) CALL message(TRIM(routine), 'Starting hydro configuration')

    ! quincy variables for soil layer calculations
    k_soil_profile  = 0.25_wp
    min_layer_depth = 0.065_wp

    !>
    !> Default values are defined for all namelist parameters.
    !
    active                 = .TRUE.
    lrestart_cont          = .FALSE.   ! TRUE: Continue although HYDRO variables are missing in restart file
    bc_filename            = 'bc_land_hydro.nc'
    ic_filename            = 'ic_land_hydro.nc'
    bc_sso_filename        = 'bc_land_sso.nc'
    w_skin_max             = 2.E-4_wp            ! 0.2 mm corresponds to Roesch et al. 2001
    w_soil_limit           = -1.0_wp
    w_soil_crit_fract      = 0.75_wp
    w_soil_wilt_fract      = 0.35_wp
    snow_depth_max         = -1.0_wp             ! -1: no limitation
    ret_macro_blg          = 432000._wp          ! [s]
    ret_macro_srf          = 36000._wp           ! [s]
    scheme_stom_cond       = 'echam5'
    l_read_initial_moist   = .FALSE.
    sanitize_restart       = .FALSE.
    enforce_water_budget   = 'unset'
    l_organic              = .TRUE.
    l_socmap               = .TRUE.
    l_soil_texture         = .FALSE.
    l_glac_in_soil         = .FALSE.
    l_infil_subzero        = .TRUE.
    l_ponds                = .FALSE.
    pond_dynamics          = "tanh"
    soilhydmodel           = "VanGenuchten"
    interpol_mean          = "upstream"
    hydro_scale            = "semi_distributed"
    qs_nsoil_water         = 5

    ! Open the file containing the hydrology namelist
    nml_handler = open_nml(TRIM(config%namelist_filename))

    ! Find and read the hydrology namelist
    ! The parameters defined in the namelist overwrite the default values defined above.
    nml_unit = position_nml('jsb_hydro_nml', nml_handler, STATUS=istat)
    IF (istat == POSITIONED) READ(nml_unit, jsb_hydro_nml)

    CALL close_nml(nml_handler)

    !> After reading the namelist, all configurable parameters of the hydrology - default parameters
    !> and values set in the namelist - are stored in the configuration data structure 'config' to
    !> make them easily accessible in all ICON-Land process modules ("dsl4jsb_Use_config(HYDRO_)").
    !>
    !> The specific choice of parameters is printed to the experiment's log file to document the
    !> setup. Besides, we check the consistency of the settings and issue warnings if parameters
    !> have been set questionably, or even stopp the simulation in case of unreasonable settings.

    config%active              = active
    config%lrestart_cont       = lrestart_cont
    config%ic_filename         = ic_filename
    config%bc_filename         = bc_filename
    config%bc_sso_filename     = bc_sso_filename

    config%w_skin_max          = w_skin_max
    CALL message(TRIM(routine), 'Maximum water holding capacity of soil or canopy skin: '//TRIM(real2string(w_skin_max)))

    config%w_soil_limit        = w_soil_limit
    CALL message(TRIM(routine), 'Upper limit for maximum soil moisture content: '//TRIM(real2string(w_soil_limit)))

    config%w_soil_crit_fract   = w_soil_crit_fract
    CALL message(TRIM(routine), 'Fraction of field capacity at critical point: '//TRIM(real2string(w_soil_crit_fract)))

    config%w_soil_wilt_fract   = w_soil_wilt_fract
    CALL message(TRIM(routine), 'Fraction of field capacity at permanent wilting point: '//TRIM(real2string(w_soil_wilt_fract)))

    config%snow_depth_max      = snow_depth_max
    IF (snow_depth_max >= 0._wp) THEN
      CALL message(TRIM(routine), 'Snow depth limitation: '//TRIM(real2string(snow_depth_max)))
    END IF

    config%ret_macro_blg       = ret_macro_blg
    config%ret_macro_srf       = ret_macro_srf

    config%scheme_stom_cond       = scheme_stom_cond

    config%l_read_initial_moist   = l_read_initial_moist

    config%sanitize_restart       = sanitize_restart
    IF (config%sanitize_restart) THEN
      CALL message(TRIM(routine), 'Soil hydrology of the restart file is sanitized.')
    END IF

    SELECT CASE (tolower(TRIM(enforce_water_budget)))
    CASE ("unset")
      config%enforce_water_budget = config%model_config%enforce_water_budget
    CASE ("ignore")
      config%enforce_water_budget = WB_IGNORE
      CALL message(TRIM(routine), 'WARNING: Land surface water balance will not be checked during simulation.')
    CASE ("logging")
      config%enforce_water_budget = WB_LOGGING
      CALL message(TRIM(routine), 'WARNING: Simulation will not stop due to any land surface water balance violation '// &
        & 'but information will be added to the log file')
    CASE ("error")
      config%enforce_water_budget = WB_ERROR
      CALL message(TRIM(routine), 'WARNING: Simulation will stop due to any land surface water balance violation.')
    CASE DEFAULT
      CALL finish(TRIM(routine), 'enforce_water_budget == '//tolower(TRIM(enforce_water_budget))//' not available.')
    END SELECT
    IF (config%enforce_water_budget == WB_ERROR .AND. sanitize_restart) THEN
      CALL message(TRIM(routine), 'WARNING: Sanitizing restart files causes water balance violations.' &
        &                       //' You should switch off enforce_water_budget in case sanitizing is needed.')
    END IF

    config%l_soil_texture = l_soil_texture
    IF (l_soil_texture) THEN
      CALL message(TRIM(routine), 'Calculate hydrological parameters for mineral soils from soil texture.')
    END IF

    config%l_organic = l_organic
    IF (l_organic) THEN
      CALL message(TRIM(routine), 'Include organic matter in calculation of hydrological soil parameters.')
    ELSE
      IF (l_soil_texture) THEN
        CALL message (TRIM(routine), 'WARNING: l_organic=.F. and l_soil_texture=.T.' &
                                   //' => Hydrological soil parameters represent purely mineral soils!')
      ELSE
        CALL message (TRIM(routine), 'Soil hydrological parameters read from bc file.')
      END IF
    END IF

    config%l_socmap = l_socmap
    IF (l_socmap .AND. l_organic) THEN
      CALL message(TRIM(routine), 'Read map with soil organic carbon fractions from bc file.')
    ELSE IF (.NOT. l_socmap .AND. l_organic) THEN
      CALL message(TRIM(routine), 'Calculate soil organic carbon fractions from forest fractions.')
    ELSE IF (l_socmap .AND. .NOT. l_organic) THEN
      CALL message(TRIM(routine), 'Setting l_socmap=false since l_organic=false and organic matter fractions are not used.')
      config%l_socmap = .FALSE.
    END IF

    config%l_glac_in_soil      = l_glac_in_soil
    config%l_infil_subzero     = l_infil_subzero
    IF (l_infil_subzero) CALL message(TRIM(routine), 'Allow infiltration at sub-zero temperatures')

    config%l_ponds = l_ponds
    IF (l_ponds) THEN
      CALL message(TRIM(routine), '*** WARNING: Use of ponds is still experimental and not thoroughly validated!!')
    ELSE
      CALL message(TRIM(routine), 'Experimental pond scheme is disabled.')
    END IF

    SELECT CASE(TRIM(tolower(soilhydmodel)))
    CASE("vangenuchten")
      config%soilhydmodel = VanGenuchten_
      CALL message(TRIM(routine), 'Soil hydrology uses the Van Genuchten model')
    CASE("brookscorey")
      config%soilhydmodel = BrooksCorey_
      CALL message(TRIM(routine), 'Soil hydrology uses the Brooks & Corey model')
    CASE("campbell")
      config%soilhydmodel = Campbell_
      CALL message(TRIM(routine), 'Soil hydrology uses the Campbell model')
    CASE default
      CALL finish(TRIM(routine), 'Soil hydrology model '//TRIM(soilhydmodel)//' not implemented')
    END SELECT

    SELECT CASE(TRIM(tolower(interpol_mean)))
    CASE("upstream")
      config%interpol_mean = Upstream_
      CALL message(TRIM(routine), 'Soil diffusivity/conductivity interpolation uses upstream means')
    CASE("arithmetic")
      config%interpol_mean = Arithmetic_
      CALL message(TRIM(routine), 'Soil diffusivity/conductivity interpolation uses arithmetic means')
    CASE default
      CALL finish(TRIM(routine), 'Soil diffusivity/conductivity interpolation method '//TRIM(interpol_mean)//' not implemented')
    END SELECT

    SELECT CASE(TRIM(tolower(pond_dynamics)))
    CASE("quadratic")
      config%pond_dynamics = Quad_
      CALL message(TRIM(routine), 'Pond dynamics scheme uses quadratic scaling')
    CASE("tanh")
      config%pond_dynamics = Tanh_
      CALL message(TRIM(routine), 'Pond dynamics scheme uses tanh scaling')
    CASE default
      CALL finish(TRIM(routine), 'Pond dynamics scheme '//TRIM(pond_dynamics)//' not implemented')
    END SELECT

    SELECT CASE(TRIM(tolower(hydro_scale)))
    CASE("semi_distributed")
      config%hydro_scale = Semi_Distributed_
      CALL message(TRIM(routine), 'Using semi-distributed scale parametrization (ARNO scheme) for soil hydrological fluxes')
    CASE("uniform")
      config%hydro_scale = Uniform_
      CALL message(TRIM(routine), 'Using uniform scale process implementation for hydrological fluxes (still experimental)')
      CALL message(TRIM(routine), 'Runoff retention times [s]:'//TRIM(real2string(ret_macro_blg))// &
        &                         ' (below ground) and '//TRIM(real2string(ret_macro_srf))//' (surface)')
    CASE default
      CALL finish(TRIM(routine), 'Soil hydrology scale '//TRIM(hydro_scale)//' not implemented')
    END SELECT

    ! quincy standalone
    config%qs_nsoil_water = qs_nsoil_water

    IF (.NOT. active) RETURN

#ifndef __QUINCY_STANDALONE__
    input_file = jsb_netcdf_open_input(ic_filename)

    ptr_1D => input_file%Read_1d(variable_name='soillev')    ! Depth of layer bottom
    nsoil = SIZE(ptr_1D)

    IF (l_organic .AND. nsoil < 3) &
      & CALL finish(TRIM(routine), 'At least three soil layers required for using organic soil component')

    ! calculate mid-points of each soil layer
    ALLOCATE(depths(nsoil+1))
    ALLOCATE(mids(nsoil))
    depths(1) = 0._wp
    depths(2:nsoil+1) = ptr_1D(1:nsoil)
    mids(1:nsoil) = (depths(1:nsoil) + depths(2:nsoil+1)) / 2._wp
    DEALLOCATE(ptr_1D)

    CALL input_file%Close()
#else
    ! quincy standalone
    !
    ! docu of values used for 5 soil layers
    !   jsbach:
    !     soillev = 0.065, 0.319, 1.232, 4.134, 9.834
    !   quincy:
    !     src/q_soil_phys/mo_spq_config_class.f90
    !     dz_water (1:5)                = (/0.065_wp,0.254_wp,0.913_wp,2.902_wp,5.700_wp/)

    ! temporary solution: set consistently with SPQ_ config via namelist
    nsoil = qs_nsoil_water

    ALLOCATE(depths(nsoil+1))
    ALLOCATE(mids(nsoil))
    depths(1) = 0._wp
    ALLOCATE(dz_water(nsoil))

    ! slightly modified copy/paste from SPQ_ config
    ! If not using 5 layers, estimate soil profile (i.e., dz == layer thicknesses) to mimick standard JSB4 profile
    !  assuming dz_water(:) = dz_energy(:)
    IF (nsoil > 5) THEN
      ! initial calculation of soil profile (for soil moisture calculations) calculating the upper bound of each layer
      DO isoil = 1,nsoil
        soil_layer_profile_ubound_estimate(isoil) = min_layer_depth * &
          &                                         EXP(k_soil_profile * isoil * REAL(max_soil_layers, wp) / REAL(nsoil, wp)) - &
          &                                         min_layer_depth
      END DO
      ! final calculation of soil profile, calculating the layer thickness (dz(:)) and correcting for minimum soil-layer thickness
      dz_water(1) = min_layer_depth
      DO isoil = 2,nsoil
        dz_water(isoil) = MAX(min_layer_depth, &
          &                   soil_layer_profile_ubound_estimate(isoil) - soil_layer_profile_ubound_estimate(isoil-1))
      END DO
      ! temporary: for compatibility with "jsbach new_vgrid('soil_depth_water', ...)" below
      depths(2:nsoil+1) = dz_water(1:nsoil)
    ELSE
      depths(2:nsoil+1) = (/0.065_wp,0.254_wp,0.913_wp,2.902_wp,5.700_wp/)
    END IF
    mids(1:nsoil) = (depths(1:nsoil) + depths(2:nsoil+1)) / 2._wp
    DEALLOCATE(dz_water)
#endif

    soil_w  => new_vgrid('soil_depth_water', ZAXIS_DEPTH_BELOW_LAND, nsoil, &
      & levels  = mids                 (1:nsoil  ),                         &
      & lbounds = depths               (1:nsoil  ),                         &
      & ubounds = depths               (2:nsoil+1),                         &
      & units='m')
    CALL register_vgrid(soil_w)
    WRITE(message_text, *) 'Soil levels in hydrology (upper) [m]: ', soil_w%lbounds
    CALL message(TRIM(routine), message_text)
    WRITE(message_text, *) 'Soil levels in hydrology (mid)   [m]: ', soil_w%levels
    CALL message(TRIM(routine), message_text)
    WRITE(message_text, *) 'Soil levels in hydrology (lower) [m]: ', soil_w%ubounds
    CALL message(TRIM(routine), message_text)
    WRITE(message_text, *) 'Soil level depths in hydrology   [m]: ', soil_w%dz
    CALL message(TRIM(routine), message_text)
    DEALLOCATE(depths, mids)

  END SUBROUTINE Init_hydro_config

#endif
END MODULE mo_hydro_config_class
