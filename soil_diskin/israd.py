"""Bulk 0-100 cm soil profiles from the International Soil Radiocarbon Database (ISRaD).

Reads the flat layer table of ISRaD (data/ISRaD/ISRaD_data_flat_layer_*.csv) and
combines each profile's layers into one pseudo-bulk soil column, as needed to
compare ISRaD with the bulk Balesdent analysis. Used by
notebooks/01_preprocess_israd.py and notebooks/experimental/calibrate_lognormal_israd.py.

Units follow the Balesdent tables: stocks in kg C/m², depths in cm, and
radiocarbon as R = Delta14C/1000 + 1, the convention of data/14C_atm_annot.csv.
ISRaD's own units (from its template documentation): lyr_soc in g C/cm², bulk
density in g/cm³, carbon in weight %, lyr_14c in per mil, and fraction modern
relative to 95% of the 1950 NBS oxalic acid standard.
"""
import numpy as np
import pandas as pd

from soil_diskin.constants import LAMBDA_14C

DEPTH = 100.  # cm
PROFILE = ['entry_name', 'site_name', 'pro_name', 'lyr_obs_date_y']  # one pseudo-bulk column each
# Profile fields reported by the data contributors that mark permafrost or peat soils.
SOIL_FLAGS = ['pro_permafrost', 'pro_peatland', 'pro_thaw_depth', 'pro_usda_soil_order']
G_CM2_TO_KG_M2 = 10.  # 1 g/cm² = 10 kg/m²


def load_layers(path):
    """ISRaD layers with radiocarbon (R) and a carbon stock (stock_kg_m2), from measured values only.

    R is lyr_14c/1000 + 1, or, where only fraction modern is reported, Fm
    exp(-lambda (year - 1950)): ISRaD's Fm is decay corrected to 1950, while R
    (like Delta14C) refers to the sampling year. The stock is the reported
    lyr_soc, or else carbon % x bulk density x thickness, which is how ISRaD
    computes lyr_soc (no coarse-fragment correction). Carbon % is lyr_c_org, else
    lyr_c_tot; bulk density is lyr_bd_samp (coarse fragments removed), else
    lyr_bd_tot. `fallback` marks layers that used lyr_c_tot or lyr_bd_tot.
    Layers without R, a stock, bounds, a sampling year or coordinates, or with
    zero thickness or a negative stock, are dropped. The SOIL_FLAGS profile
    fields are kept, for permafrost_or_peat.
    """
    numeric = ['site_lat', 'site_long', 'lyr_top', 'lyr_bot', 'lyr_obs_date_y', 'lyr_c_org',
               'lyr_c_tot', 'lyr_soc', 'lyr_bd_samp', 'lyr_bd_tot', 'lyr_14c', 'lyr_fraction_modern']
    # Some text fields are not valid UTF-8, so undecodable bytes are replaced.
    layers = pd.read_csv(path, usecols=PROFILE[:3]+['lyr_name']+numeric+SOIL_FLAGS,
                         encoding_errors='replace', low_memory=False)
    # Non-numeric entries (e.g. notes in a numeric column) become NaN.
    layers[numeric] = layers[numeric].apply(pd.to_numeric, errors='coerce')

    # Radiocarbon: Delta14C if reported, else fraction modern decayed from 1950 to the sampling year.
    year = layers.lyr_obs_date_y
    layers['R'] = (layers.lyr_14c/1000 + 1).fillna(
        layers.lyr_fraction_modern*np.exp(-(year - 1950)*LAMBDA_14C))

    # Carbon stock: reported lyr_soc, else carbon % x bulk density x thickness (g C/cm²),
    # preferring organic carbon and fine-earth bulk density where both are reported.
    thickness = layers.lyr_bot - layers.lyr_top
    carbon = layers.lyr_c_org.fillna(layers.lyr_c_tot)
    density = layers.lyr_bd_samp.fillna(layers.lyr_bd_tot)
    stock_g_cm2 = layers.lyr_soc.fillna(carbon/100*density*thickness)
    # Flag computed stocks that needed total carbon or whole-soil bulk density.
    layers['fallback'] = layers.lyr_soc.isna() & (
        (layers.lyr_c_org.isna() & layers.lyr_c_tot.notna())
        | (layers.lyr_bd_samp.isna() & layers.lyr_bd_tot.notna()))
    layers['stock_kg_m2'] = stock_g_cm2*G_CM2_TO_KG_M2

    # Keep layers with everything pseudo_bulk needs, a positive thickness and a non-negative stock.
    valid = layers[['R', 'stock_kg_m2', 'lyr_top', 'lyr_bot', 'lyr_obs_date_y', 'site_lat', 'site_long']].notna().all(axis=1)
    return layers[valid & (thickness > 0) & (layers.stock_kg_m2 >= 0)]


def permafrost_or_peat(layers):
    """Profile-years whose contributors report permafrost or peat, with the reason.

    A profile-year is flagged if any of its layers has pro_permafrost or
    pro_peatland 'yes', a reported thaw depth (the depth to frozen soil), or a
    USDA soil order of Gelisols (permafrost soils) or Histosols (peat). Only the
    contributors' fields are used, not ISRaD's map-derived soil orders.

    Returns:
        dict mapping PROFILE tuples to a description of the evidence.
    """
    orders = {'gelisols': 'Gelisols (permafrost)', 'histosols': 'Histosols (peat)'}
    reasons = {}
    for key, profile in layers.groupby(PROFILE, dropna=False):
        evidence = [f'{column} = yes' for column in ['pro_permafrost', 'pro_peatland']
                    if profile[column].astype(str).str.strip().str.lower().eq('yes').any()]
        thaw = pd.to_numeric(profile.pro_thaw_depth, errors='coerce').dropna()
        if len(thaw):
            evidence.append(f'thaw depth {thaw.iloc[0]:g} cm')
        reported = profile.pro_usda_soil_order.dropna().astype(str).str.strip().str.lower()
        evidence += [f'soil order {orders[o]}' for o in sorted(set(reported) & set(orders))]
        if evidence:
            reasons[key] = '; '.join(evidence)
    return reasons


def pseudo_bulk(layers, depth=DEPTH):
    """Stock and stock-weighted R over [0, depth] cm for each completely covered profile-year.

    This turns a depth-resolved profile into the bulk 0-100 cm observation the
    Balesdent analysis uses: one carbon stock and one radiocarbon value for the
    whole mineral soil column. For each profile-year (PROFILE: entry, site,
    profile and sampling year, so resampled profiles give one row per year):

      1. Clip every layer to [0, depth]. ISRaD measures depth from the top of the
         mineral soil, so organic horizons lie at negative depths and drop out;
         a layer straddling 0 or `depth` keeps only its part inside.
      2. Require complete coverage: the clipped layers, sorted by depth, must
         abut from 0 to `depth` with no gap and no overlap. A gap (e.g. an
         unsampled interval, or a layer load_layers dropped for lack of
         radiocarbon or a stock) would leave part of the column unaccounted for;
         an overlap (e.g. replicate cores of the same interval reported as
         separate layers) would count it twice. Either way the profile-year is
         rejected rather than repaired.
      3. Prorate each layer's stock by the share of its thickness inside
         [0, depth], assuming carbon is spread evenly through the layer.
      4. Sum the prorated stocks, and average R weighted by them. R is a ratio
         of 14C to total carbon, so the bulk value is the stock-weighted mean:
         what a single sample homogenized over 0-`depth` would measure.

    Profile-years with no stock in [0, depth] are dropped.

    Parameters
    ----------
    layers : pd.DataFrame
        Layers from load_layers, with R, stock_kg_m2 and fallback.
    depth : float
        Bottom of the bulk column in cm (default DEPTH, 100 cm, as in Balesdent).

    Returns
    -------
    pd.DataFrame
        One row per profile-year: the PROFILE columns; latitude and longitude
        (site_lat and site_long of the profile's first layer); n_layers, the
        layers contributing to [0, depth]; n_fallback_layers, those whose stock
        used total carbon or bulk density with coarse fragments (see
        load_layers); stock_kg_m2, the total stock over [0, depth] in kg C/m²;
        and R, the stock-weighted mean radiocarbon ratio. Empty, with these
        columns, if no profile-year is complete.
    """
    rows = []
    # Loop over individual profile-years, identified by the PROFILE columns.
    # dropna=False: keep profiles missing an identifier, e.g. a sampling year.
    for key, profile in layers.groupby(PROFILE, dropna=False):
        # 1. Clip to [0, depth] and keep the layers with some thickness left inside.
        top, bottom = profile.lyr_top.clip(lower=0), profile.lyr_bot.clip(upper=depth)
        inside = bottom > top
        if not inside.any():
            continue
        profile, top, bottom = profile[inside], top[inside], bottom[inside]
        # Sort by depth; `order` also reorders every per-layer array below.
        order = np.argsort(top.to_numpy())
        top, bottom = top.to_numpy()[order], bottom.to_numpy()[order]

        # 2. Complete coverage without overlap: layers abut from 0 to depth. Contributors
        # report adjacent layers with matching bounds, so an exact comparison suffices.
        if top[0] > 0 or bottom[-1] < depth or np.any(top[1:] != bottom[:-1]):
            continue # Reject profile-years with a gap or overlap in [0, depth].

        # 3. Each layer's share of its own thickness inside [0, depth] (1 unless clipped).
        share = (bottom - top)/(profile.lyr_bot - profile.lyr_top).to_numpy()[order]
        stock = profile.stock_kg_m2.to_numpy()[order]*share
        if stock.sum() <= 0:
            continue # Reject profile-years with no stock in [0, depth].

        # 4. Total stock and stock-weighted R over the column.
        stock_kg_m2 = stock.sum()
        stock_weighted_R = (stock*profile.R.to_numpy()[order]).sum()/stock.sum()
        rows.append(dict(zip(PROFILE, key), latitude=profile.site_lat.iloc[0],
                         longitude=profile.site_long.iloc[0], n_layers=len(profile),
                         n_fallback_layers=int(profile.fallback.sum()),
                         stock_kg_m2=stock_kg_m2, R=stock_weighted_R))
    columns = PROFILE + ['latitude', 'longitude', 'n_layers', 'n_fallback_layers', 'stock_kg_m2', 'R']
    return pd.DataFrame(rows, columns=columns)
