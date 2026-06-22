#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ===================================================================*)
(* LOGNORMAL AGE DISTRIBUTION SCAN FOR SOIL RADIOCARBON MODELING     *)
(* ===================================================================*)
(* This script performs age distribution scans for soil radiocarbon   *)
(* using a lognormal model. It:                                       *)
(*   1. Loads atmospheric 14C data and site turnover measurements     *)
(*   2. Scans across a range of mean ages for each site               *)
(*   3. Computes predicted radiocarbon values for each (tau, age)     *)
(*   4. Exports results for main estimates and confidence intervals   *)
(* ===================================================================*)

(* ===================================================================*)
(* SECTION 1: DATA LOADING AND INITIALIZATION                         *)
(* ===================================================================*)

(* Load atmospheric 14C data from annotated CSV file *)
C14DATADIR = "data/14C_atm_annot.csv";
C14DATA = Import[C14DATADIR, "HeaderLines" -> 1];
C14DATA = C14DATA[[All, {4, 5}]];  (* Extract columns 4 and 5: year and 14C ratio *)

(* Create interpolation function for atmospheric 14C *)
(* Use mean ratio from recent data for extrapolation beyond data range *)
MeanR = Last[Mean[C14DATA[[-50000 ;;]]]];
atm14C = 
  Interpolation[C14DATA, InterpolationOrder -> 0, 
   "ExtrapolationHandler" -> {(MeanR &)}];

(* ===================================================================*)
(* SECTION 2: LOGNORMAL RADIOCARBON MODEL FUNCTION                    *)
(* ===================================================================*)

(* Calculate radiocarbon signal from turnover time and mean age *)
(* Parameters:                                                   *)
(*   tau  - turnover time (years)                               *)
(*   age  - mean age of the system (years)                      *)
(* Returns: predicted fraction modern (fm) value                *)
(* Note: Uses 8267 years as the radiocarbon decay constant      *)
LognormalRadiocarbon[tau_, age_] := (
  (* Calculate lognormal distribution parameters *)
  sig = Sqrt[Log[age/tau]];  (* Standard deviation of log-ages *)
  mu = -Log[Sqrt[tau^3/age]];  (* Mean of log-ages *)
  
  (* Integrate over age distribution and atmospheric 14C history *)
  r = NIntegrate[ 
    atm14C[a]*Exp[-a/8267]* PDF[LogNormalDistribution[mu, sig], x] * 
     Exp[-x*a] / Exp[-mu + (sig^2)/2], {x, 0, Infinity}, {a, 0, 
     Infinity}, PrecisionGoal -> 3];
  r
  )

(* ===================================================================*)
(* SECTION 3: SITE DATA LOADING AND PREPROCESSING                     *)
(* ===================================================================*)

csvFilePath = "results/all_sites_14C_turnover.csv";
Print["Importing data from: ", csvFilePath];

(* Import site data with turnover times and observed 14C measurements *)
data = Import[csvFilePath, "Dataset", HeaderLines -> 1];
data = Values[data[[All, {"turnover", "fm"}]]] // Normal;

(* Clean data: convert strings to numbers and filter invalid entries *)
data = Cases[data, {x_, y_} /; NumericQ[ToExpression[x]] :> {ToExpression[x], ToExpression[y]}];

(* ===================================================================*)
(* SECTION 4: PARALLEL COMPUTATION SETUP                              *)
(* ===================================================================*)

(* Initialize parallel kernels for faster computation *)
LaunchKernels[];

(* Distribute necessary functions and data to all parallel kernels *)
DistributeDefinitions[LognormalRadiocarbon, atm14C];

(* ===================================================================*)
(* SECTION 5: AGE SCAN FUNCTION DEFINITIONS                           *)
(* ===================================================================*)

(* Define logarithmic age range for scanning (1000 to ~316,000 years) *)
agelist = 10^Range[3, 5.5, (5.5 - 3)/100];

(* Scan ages for a given turnover time *)
(* For each age in agelist, compute predicted radiocarbon value *)
ScanAges[tau_] := 
 Module[{ages}, 
  ages = Table[Quiet[LognormalRadiocarbon[tau, i]], {i, agelist}];
  Return[ages]
 ]

(* Export scan results to HDF5 format *)
(* Parameters:                                                        *)
(*   filePath     - output HDF5 file path                            *)
(*   results      - matrix of radiocarbon predictions                *)
(*   ages         - list of ages used in the scan                    *)
(*   siteIndices  - indices of sites corresponding to result rows    *)
ExportScanResults[filePath_, results_, ages_, siteIndices_] := Module[{},
  Export[filePath, {
    "/results" -> results,
    "/ages" -> ages,
    "/site_indices" -> siteIndices
  }, {"Datasets", "Rules"}]
]

(* ===================================================================*)
(* SECTION 6: MAIN AGE SCAN (CENTRAL ESTIMATES)                       *)
(* ===================================================================*)

Print["Starting main age scan with ", Length[data], " sites..."];
{mainScanTime, results} = AbsoluteTiming[ParallelMap[ScanAges[#] &, data[[All, 1]]]];
Print["Main age scan finished in ", mainScanTime, " seconds. Starting export..."];

(* Export main results to HDF5 *)
{mainExportTime, _} = AbsoluteTiming[
  ExportScanResults["results/03_calibrate_models/03b_lognormal_model_age_scan.h5", 
    results, agelist, Range[Length[data]]]
];
Print["Main age scan export complete in ", mainExportTime, " seconds."];

(* Reset parallel kernels to free memory before next computation *)
CloseKernels[];
LaunchKernels[];
DistributeDefinitions[LognormalRadiocarbon, atm14C];

(* ===================================================================*)
(* SECTION 7: UNCERTAINTY BOUNDS DATA PREPARATION                     *)
(* ===================================================================*)

(* Load raw data to access confidence interval columns *)
rawData = Import[csvFilePath, "Data"];

(* Separate header and data rows *)
header = First[rawData];
body = Rest[rawData];

(* Find column indices for 5th and 95th percentile turnover times *)
colIndex05 = Position[header, "turnover_q05"][[1, 1]];
colIndex95 = Position[header, "turnover_q95"][[1, 1]];

(* Filter to sites with complete confidence interval data *)
cleanBody = Select[body, #[[colIndex05]] != "" && #[[colIndex95]] != "" &];

(* ===================================================================*)
(* SECTION 8: LOWER BOUND (5TH PERCENTILE) AGE SCAN                   *)
(* ===================================================================*)

Print["Starting 5% confidence interval scan with ", Length[cleanBody], " sites..."];
{lowScanTime, lowresults} = AbsoluteTiming[ParallelMap[ScanAges[#] &, cleanBody[[All, colIndex05]]]];
Print["5% confidence interval scan finished in ", lowScanTime, " seconds. Starting export..."];

(* Export lower bound results *)
{lowExportTime, _} = AbsoluteTiming[
  ExportScanResults["results/03_calibrate_models/03b_lognormal_model_age_scan_05.h5", 
    lowresults, agelist, Range[Length[cleanBody]]]
];
Print["5% confidence interval scan export complete in ", lowExportTime, " seconds."];

(* Reset parallel kernels *)
CloseKernels[];
LaunchKernels[];
DistributeDefinitions[LognormalRadiocarbon, atm14C];

(* ===================================================================*)
(* SECTION 9: UPPER BOUND (95TH PERCENTILE) AGE SCAN                  *)
(* ===================================================================*)

Print["Starting 95% confidence interval scan with ", Length[cleanBody], " sites..."];
{highScanTime, highresults} = AbsoluteTiming[ParallelMap[ScanAges[#] &, cleanBody[[All, colIndex95]]]];
Print["95% confidence interval scan finished in ", highScanTime, " seconds. Starting export..."];

(* Export upper bound results *)
{highExportTime, _} = AbsoluteTiming[
  ExportScanResults["results/03_calibrate_models/03b_lognormal_model_age_scan_95.h5", 
    highresults, agelist, Range[Length[cleanBody]]]
];
Print["95% confidence interval scan export complete in ", highExportTime, " seconds."];

(* ===================================================================*)
(* SECTION 10: CLEANUP AND COMPLETION                                *)
(* ===================================================================*)

(* Close all parallel kernels to release resources *)
CloseKernels[];

Print["All scans complete. Results exported to results/03_calibrate_models/"];




