#!/usr/bin/env wolframscript
(* Small lognormal age-scan smoke script for tests
        - Reads data/14C_atm_annot.csv
        - Builds a piecewise-constant atmospheric interpolation
        - Runs a short age scan (5 points) for a fixed tau
        - Writes results as JSON to a specified output file or stdout
*)

(* Load atmospheric 14C data from annotated CSV file *)
C14DATADIR = "data/14C_atm_annot.csv";
raw = Import[C14DATADIR, "CSV"];
(* If file has a header, drop a single leading header row if non-numeric *)
If[! NumericQ[ToExpression[raw[[1, 1]]]], raw = Rest[raw]];
(* Extract columns: years_before_2000 (col 4) and R_14C (col 5) if present; fallback to first two cols *)
cols = If[Length[raw[[1]]] >= 5, {4, 5}, {1, 2}];
C14DATA = raw[[All, cols]];

(* Determine mean for tail using up to last 50000 rows *)
n = Min[50000, Length[C14DATA]];
MeanR = If[n > 0, Last[Mean[Take[C14DATA, -n]]], 1.0];

(* Build piecewise-constant interpolation with constant tail *)
atm14C = Interpolation[C14DATA, InterpolationOrder -> 0, "ExtrapolationHandler" -> {(MeanR &)}];

(* Radiocarbon mean life constant *)
meanLife = 8267.0;

(* Lognormal radiocarbon evaluator (keeps same formula as main script) *)
LognormalRadiocarbon[tau_, age_] := Module[{sig, mu, r},
    sig = Sqrt[Log[age/tau]];
    mu = -Log[Sqrt[tau^3/age]];
    r = NIntegrate[
        atm14C[a] * Exp[-a/meanLife] * PDF[LogNormalDistribution[mu, sig], x] *
            Exp[-x*a] / Exp[-mu + (sig^2)/2],
        {x, 0, Infinity}, {a, 0, Infinity}, PrecisionGoal -> 4, MaxRecursion -> 12
    ];
    N[r, 12]
];

(* Small age scan (5 points) and fixed tau for a test run *)
agelist = {10., 100., 1000., 10000., 100000.};
tau = 8.0;

results = Table[LognormalRadiocarbon[tau, a], {a, agelist}];

(* Convert results to JSON string *)
json = ExportString[<|"tau" -> tau, "ages" -> agelist, "fm" -> results|>, "JSON"];

(* Write JSON to a temporary file in the working directory for test consumption *)
stream = OpenWrite["tmp_age_scan.json"];
WriteString[stream, json];
Close[stream];

(* Exit with success *)
Exit[0];