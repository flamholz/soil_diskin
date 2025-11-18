(*============================================================*)
(*1. SETUP:Data Loading and Function Definitions*)
(* ===================================================================*)
(* Read atmospheric 14C data*)
C14DATADIR = "data/14C_atm_annot.csv";
C14DATA = Import[C14DATADIR, "HeaderLines" -> 1];
C14DATA = C14DATA[[All, {4, 5}]];
(*Interpolate the 14C data, use the mean ratio for extrapolation*)
MeanR = Last[Mean[C14DATA[[-50000 ;;]]]];
atm14C = 
  Interpolation[C14DATA, InterpolationOrder -> 0, 
   "ExtrapolationHandler" -> {(MeanR &)}];

(* Plot the results of the interpolation to see it makes sense *)
Plot[atm14C[x], {x, 0, 550000}, 
 Epilog -> {PointSize[Medium], Red, Point[C14DATA]}]

(* Define the function that calculates the radiocarbon signal from the\
 inputs of the turnover time and mean age of the system *)
(* TODO to avoid confusion, replace tau with T for turnover *)
LognormalRadiocarbon[tau_, age_] := (
  sig = Sqrt[Log[age/tau]];
  mu = -Log[Sqrt[tau^3/age]];
  r = NIntegrate[ 
    atm14C[a]*Exp[-a/8267]* PDF[LogNormalDistribution[mu, sig], x] * 
     Exp[-x*a] / Exp[-mu + (sig^2)/2], {x, 0, Infinity}, {a, 0, 
     Infinity}, PrecisionGoal -> 3];
  r
  )

csvFilePath = "results/all_sites_14C_turnover.csv"
Print["Importing data from: ", csvFilePath];
data = Import[csvFilePath, "Dataset", HeaderLines -> 1];
data = Values[data[[All, {"turnover", "fm"}]]] // Normal;

LaunchKernels[];

(* 3. Distribute definitions to parallel kernels*)
(* Make sure all symbols that LognormalRadiocarbon d\
epends on are distributed.*)
(* atm14C is already defined globally.\
If it were a more complex structure or function,\
you'd want to be sure it's available.*)
DistributeDefinitions[LognormalRadiocarbon, atm14C];

(* TODO save the ages in the output file *)
agelist = 10^Range[3, 5.5, (5.5 - 3)/100];
ScanAges[tau_] := 
 Module[{ages}, 
  ages = Table[Quiet[LognormalRadiocarbon[tau, i]], {i, agelist}];
  Return[ages]]

results = ParallelMap[ScanAges[#] &, data[[All, 1]]]
Export["results/03_calibrate_models/03b_lognormal_model_age_scan.csv", results, "CSV"]