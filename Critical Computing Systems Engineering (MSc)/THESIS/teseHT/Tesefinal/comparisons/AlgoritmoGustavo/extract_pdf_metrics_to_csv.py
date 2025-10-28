#!/usr/bin/env python3
# extract_first_page_metrics.py

import sys
import re
import pdfplumber   # pip install pdfplumber
import pandas as pd

"""
Script: extract_pdf_first_page_metrics_to_csv.py
Purpose:
    Parse the first page of a specific PDF report and extract 9 sail metrics
    (CAMBER/TWIST at TOP/MID/BOT) across 20 frames (PHOTO 1..20), then save
    them to a CSV file.

How it works:
    1) Opens the PDF and extracts text from page 1.
    2) Finds the header line that starts with "PHOTO" and includes 1..20.
    3) For each metric name, finds the corresponding row and parses exactly
       20 floating-point values.
    4) Builds a DataFrame with metrics as rows and Frame_1..Frame_20 as columns.
    5) Writes the CSV.

Inputs (edit the CONFIG paths below):
    PDF_PATH : path to the input PDF file
    CSV_OUT  : path to the output CSV file

Outputs:
    CSV file with shape (9 metrics x 20 frames).

Notes:
    - Expects a table-like text layout on page 1 with the "PHOTO 1 ... 20" header.
    - Stops with a clear error if any required row or values are missing.
    - Requires: pdfplumber, pandas
      pip install pdfplumber pandas
"""

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PDF_PATH = r"C:\Users\asus\OneDrive - Instituto Superior de Engenharia do Porto\Desktop\teseHT\Tesefinal\comparisons\AlgoritmoGustavo\reportsgustavo\video_23.pdf"
CSV_OUT  = r"C:\Users\asus\OneDrive - Instituto Superior de Engenharia do Porto\Desktop\teseHT\Tesefinal\comparisons\AlgoritmoGustavo\parametrospdfsGustavo\video23_parameters.csv"
# ────────────────────────────────────────────────────────────────────────────────

METRICS = [
    "CAMBER TOP",
    "CAMBER MID",
    "CAMBER BOT",
    "DRAFT TOP",
    "DRAFT MID",
    "DRAFT BOT",
    "TWIST TOP",
    "TWIST MID",
    "TWIST BOT",
]

def main():
    # 1) extract all text from page 1
    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            txt = pdf.pages[0].extract_text()
    except Exception as e:
        sys.exit(f"ERROR: could not open or parse PDF:\n  {e}")

    if not txt:
        sys.exit("ERROR: page 1 is empty or unreadable.")

    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]

    # 2) find the “PHOTO 1 2 … 20” header line to know where the table starts
    hdr_idx = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("PHOTO"):
            # ensure it has numbers 1–20
            if re.search(r"\b1\b.*\b20\b", ln):
                hdr_idx = i
                break
    if hdr_idx is None:
        sys.exit("ERROR: could not find the PHOTO header row.")

    # 3) parse the next 9 lines for the metrics
    data = {}
    for metric in METRICS:
        # look for a line that starts with this metric
        found = False
        pattern = metric + r"\s+([0-9]+\.[0-9]+(?:\s+[0-9]+\.[0-9]+){19})"
        for ln in lines[hdr_idx+1:]:
            m = re.match(pattern, ln)
            if m:
                vals = m.group(1).split()
                data[metric.replace(" ", "_").title()] = [float(v) for v in vals]
                found = True
                break
        if not found:
            sys.exit(f"ERROR: could not find data row for '{metric}'.")

    # 4) build DataFrame
    df = pd.DataFrame(
        data,
        index=[f"Frame_{i+1}" for i in range(20)]
    ).T
    df.index.name = "Metric"

    # 5) write CSV
    df.to_csv(
        CSV_OUT,
        index=True,
        index_label="Metric",
        float_format="%.2f"
    )
    print(f"Extracted metrics → {CSV_OUT}")

if __name__ == "__main__":
    main()
