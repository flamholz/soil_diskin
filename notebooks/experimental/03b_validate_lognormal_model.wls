#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Recompute fm from the fitted lognormal age and original turnover time *)

C14DATADIR = "data/14C_atm_annot.csv";
C14DATA = Import[C14DATADIR, "HeaderLines" -> 1];
C14DATA = C14DATA[[All, {4, 5}]];

(*MeanR = Last[Mean[C14DATA[[-50000 ;;]]]];*)
MeanR = 10000;
atm14C =
  Interpolation[C14DATA, InterpolationOrder -> 0,
   "ExtrapolationHandler" -> {(MeanR &)}];

LognormalRadiocarbon[tau_, age_] := (
  sig = Sqrt[Log[age/tau]];
  mu = -Log[Sqrt[tau^3/age]];
  r = NIntegrate[
    atm14C[a]*Exp[-a/8267]*PDF[LogNormalDistribution[mu, sig], x]*
     Exp[-x*a]/Exp[-mu + (sig^2)/2], {x, 0, Infinity}, {a, 0,
     Infinity}, PrecisionGoal -> 3];
  r
  );

predictionsFilePath =
  "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv";
outputFilePath =
  "results/03_calibrate_models/03b_lognormal_validation.csv";

Print["Importing data from: ", predictionsFilePath];
rawData = Import[predictionsFilePath, "CSV"];
header = First[rawData];
body = Rest[rawData];

turnoverIndex = Position[header, "turnover"][[1, 1]];
fmIndex = Position[header, "fm"][[1, 1]];
predIndex = Position[header, "pred"][[1, 1]];

LaunchKernels[];
DistributeDefinitions[LognormalRadiocarbon, atm14C];

ValidateRow[row_] :=
 Module[{turnover, observedFm, predAge, predictedFm},
  turnover = row[[turnoverIndex]];
  observedFm = row[[fmIndex]];
  predAge = row[[predIndex]];
  predictedFm = Quiet[LognormalRadiocarbon[turnover, predAge]];
  Join[row, {predictedFm, predictedFm - observedFm, Abs[predictedFm - observedFm]}]
  ];

validatedRows = ParallelMap[ValidateRow, body];

Export[
 outputFilePath,
 Prepend[
  validatedRows,
  Join[header, {"fm_predicted", "fm_difference", "fm_abs_difference"}]
  ],
 "CSV"
 ];

Print["Validation results exported to: ", outputFilePath];
