import argparse
import pandas as pd


"""
Annotates and preprocesses historical atmospheric 14C levels
for the convenience of downstream analysis.
"""


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Annotate and preprocess atmospheric 14C history data."
	)
	parser.add_argument(
		"--input",
		default="data/14C_atm.csv",
		help="Path to input CSV containing year and Delta_14C columns.",
	)
	parser.add_argument(
		"--output",
		default="data/14C_atm_annot.csv",
		help="Path to output annotated CSV.",
	)
	args = parser.parse_args()

	# Load the 14C data
	# TODO: document the provenance of this data
	c14_data = pd.read_csv(args.input)
	c14_data.columns = ["year", "Delta_14C"]
	c14_data = c14_data.dropna().sort_values("year", ascending=True)

	# Annotate the 14C data with years before present and years before 2000
	c14_data["years_before_present"] = (c14_data.year - c14_data.year.max()).abs()
	c14_data["years_before_2000"] = 2000 - c14_data.year
	# Calculate R_14C, which is the ratio of 14C/12C relative in per mil
	c14_data["R_14C"] = c14_data.Delta_14C / 1000 + 1
	c14_data = c14_data.sort_values("year", ascending=False)
	c14_data.to_csv(args.output, index=False)


if __name__ == "__main__":
	main()