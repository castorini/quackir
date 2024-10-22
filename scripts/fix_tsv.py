import csv
import sys

def fix_tsv_quotes(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile, delimiter='\t', quotechar='"')
        writer = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)

        for row in reader:
            escaped_row = [field.replace('"', '""') for field in row]
            writer.writerow(escaped_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_tsv_quotes.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    fix_tsv_quotes(input_file, output_file)
    print(f"Processed {input_file} and saved to {output_file}")