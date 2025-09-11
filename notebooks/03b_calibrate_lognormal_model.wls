(* Load the 14C data *)
C14DATADIR = "data/14C_atm_annot.csv";
C14DATA = Import[C14DATADIR, "HeaderLines" -> 1];
C14DATA = C14DATA[[All, {4, 5}]];

(* Interpolate the 14C data, use the mean ratio for extrapolation *)
MeanR = Last[Mean[C14DATA[[-50000 ;;]]]];
atm14C = 
  Interpolation[C14DATA, InterpolationOrder -> 0, 
   "ExtrapolationHandler" -> {(MeanR &)}];

(* Define the function that calculates the radiocarbon signal from the\
 inputs of the turnover time and mean age of the system *)
LognormalRadiocarbon[tau_, age_] := (
  sig = Sqrt[Log[age/tau]];
  mu = -Log[Sqrt[tau^3/age]];
  r = NIntegrate[ 
    atm14C[a]*Exp[-a/8267]* PDF[LogNormalDistribution[mu, sig], x] * 
     Exp[-x*a] / Exp[-mu + (sig^2)/2], {x, 0, Infinity}, {a, 0, 
     Infinity}, PrecisionGoal -> 3];
  r
  )

(* Define the function that optimizes the mean age for a given turnove\
r time and observed radiocarbon signal *)
FindOptimalAge[tau_?NumericQ, rtrue_?NumericQ, 
  InitialAge_?NumericQ] := (
  Module[{objectiveFunction, solution},
   (*Define the objective function to minimize:
   the squared difference between the function output and r_true*)
   objectiveFunction[
     age_?NumericQ] := (LognormalRadiocarbon[tau, age] - rtrue)^2;
   (*Use FindMinimum to find the age that minimizes the objective func\
tion*)
   (*FindMinimum requires a starting point for'age'*)
   solution = 
    Quiet[FindMinimum[objectiveFunction[age], {age, InitialAge}, 
      AccuracyGoal -> 8, MaxIterations -> 100]];
   (*solution=FindMinimum[objectiveFunction[age],{age,InitialAge}, 
   AccuracyGoal\[Rule]8, MaxIterations->100];*)
   (*Return the value of age that minimizes the difference*)
   Return[age /. Last[solution]]]
  )

  (* Run code to optimize all of the site data in parallel *)

(*Ensure parallel kernels are launched (adjust number if needed,\
0 means all available)*)
LaunchKernels[];

(*3. Distribute definitions to parallel kernels*)
(*Make sure all symbols that FindOptimalAge and LognormalRadiocarbon d\
epend on are distributed.*)
(*atm14C is already defined globally.\
If it were a more complex structure or function,\
you'd want to be sure it's available.*)
DistributeDefinitions[LognormalRadiocarbon, FindOptimalAge, atm14C];

(*4. Function to process a single row*)
ProcessRow[{tau_, rtrue_}, defaultInitialAge_] := 
 Module[{optimalAge}, 
  Print["Processing: tau = ", tau, ", r_true = ", rtrue];
  optimalAge = FindOptimalAge[tau, rtrue, defaultInitialAge];
  Return[optimalAge];
  ]

(*5. Main script*)
csvFilePath = "results/all_sites_14C_turnover.csv"
Print["Importing data from: ", csvFilePath];
data = Import[csvFilePath, "CSV", HeaderLines -> 1];
(*data=data[{1},{"turnover","fm"}];*)
defaultInitialAge = 1000;
data = data[[{1, 2, 3}, {7, 6}]];
data = Import[csvFilePath, "Dataset", HeaderLines -> 1];
data = Values[data[[All, {"turnover", "fm"}]]] // Normal;
results = ParallelMap[ProcessRow[#, defaultInitialAge] &, data]

Export["results/03b_lognormal_model_predictions_14C.csv", predictions, "CSV"]

Export["results/03b_lognormal_site_parameters.csv", results, "CSV"]

CloseKernels[];
