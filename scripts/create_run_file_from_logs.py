import os
import argparse


def convert_to_qrel(input_file, output_file, run_tag="bm25_duckDB"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line_num, line in enumerate(infile, start=1):
                try:
                    # Remove parentheses and split by commas
                    data = line.strip()[1:-1].split(", ")

                    if len(data) != 4:
                        raise ValueError(
                            f"Invalid format on line {line_num}: {line.strip()}"
                        )

                    # Extract the values from the line
                    doc_id = data[0]
                    entity_name = data[1].strip("'")  # Remove single quotes
                    score = data[2]
                    rank = data[3]

                    # Format the line in TREC qrel format
                    qrel_line = f"{doc_id} Q0 {entity_name} {rank} {score} {run_tag}\n"
                    outfile.write(qrel_line)

                except ValueError as ve:
                    print(ve)
                    continue  # Skip invalid lines

        print(f"Converted {input_file} to {output_file} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert a file to TREC qrel format.")
    parser.add_argument("--input-file", required=True, help="Path to the input file.")
    parser.add_argument("--output-file", required=True, help="Path to the output file.")
    parser.add_argument(
        "--run-tag",
        default="bm25_duckDB",
        help="Run tag to use in the output file (default: bm25_duckDB).",
    )

    args = parser.parse_args()
    convert_to_qrel(args.input_file, args.output_file, args.run_tag)


if __name__ == "__main__":
    main()
