# Soil carbon dynamics

This project studies soil carbon storage and replacement using decomposition-rate distributions and isotope information.

## Language

**Soil column**:
The soil from the surface to 100 cm depth at a site, composed of ten adjacent 10 cm layers in the proposed layered model.

**Layer carbon stock**:
The mass of carbon per unit ground area within one depth interval, distinct from the cumulative stock between the surface and a given depth.

**Complete calibration profile**:
A soil profile with ten finite, positive layer carbon stocks, ten finite layer radiocarbon values, and finite, positive site NPP. Observed new-carbon fractions are evaluation data and are not required to fit the profile.

**Calibration profile**:
A named Balesdent soil profile with its own layer stocks, labeling duration, and fitted layer parameters. Profiles at the same geographic coordinates remain distinct and can share gridded NPP and radiocarbon targets.
_Avoid_: Unique coordinate pair as a synonym for profile

**Vertical redistribution**:
Transfer of existing soil carbon between layers, conserving the total column stock when considered separately from external inputs and decomposition.

**Closed transport boundary**:
A column boundary across which no carbon passes through vertical redistribution; external carbon inputs and decomposition remain possible within the column.

**Effective downward velocity**:
The velocity parameter representing downward redistribution of bulk soil carbon.
_Avoid_: Pore-water velocity

**Input e-folding depth**:
The depth increment over which the external carbon input density decreases by a factor of e in an exponential depth profile; the proposed layered model uses the same value at every site.

**Layer external input**:
The part of site NPP entering a layer directly, allocated by integrating an exponential depth profile over that layer and normalizing over 0–100 cm. The ten layer external inputs sum to site NPP and exclude carbon transferred from other layers.

**Shared hyperparameters**:
The diffusion coefficient D, effective downward velocity v, and input e-folding depth h, each taking one common value across all modeled sites.

**Input decomposition-rate distribution**:
The distribution of first-order decomposition rates among newly entering external carbon; each layer's mu and sigma describe the natural logarithm of this rate. The distribution of carbon already resident in a layer need not be log-normal.

**Energy class**:
A carbon class identified by its first-order decomposition rate k, which remains unchanged when the carbon moves between layers.
_Avoid_: Layer-relative rate class

**Coupled carbon steady state**:
The layer and energy-class carbon stocks that remain constant under fixed external inputs, decomposition rates, and vertical redistribution. Each layer's balance includes imports and exports as well as its external input and decomposition.

**New-carbon fraction (f_new)**:
The fraction of a layer's current carbon that entered the column as external input after labeling began, starting from the coupled carbon steady state with unchanged inputs and parameters. Carbon retains its new or old status when it moves between layers.
_Avoid_: Fraction newly arrived in the layer
