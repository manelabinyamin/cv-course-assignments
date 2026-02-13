import argparse
import os
import subprocess

try:
    from PyPDF2 import PdfMerger

    MERGE = True
except ImportError:
    print("Could not find PyPDF2. Leaving pdf files unmerged.")
    MERGE = False


def notebook_to_pdf(notebook_file):
    base = os.path.splitext(notebook_file)[0]
    tex_file = base + ".tex"

    # Step 1: Export to LaTeX
    subprocess.run([
        "jupyter", "nbconvert",
        "--log-level", "CRITICAL",
        "--to", "latex",
        notebook_file,
    ])

    if not os.path.exists(tex_file):
        return False

    # Step 2: Fix TeX Live 2025 incompatibility — pandoc generates
    # \def\LTcaptype{none} but newer longtable/hyperref expects a valid
    # counter name, causing "No counter 'none' defined".
    with open(tex_file, "r") as f:
        tex = f.read()
    tex = tex.replace(r"\def\LTcaptype{none}", r"\def\LTcaptype{table}")
    with open(tex_file, "w") as f:
        f.write(tex)

    # Step 3: Compile with xelatex (twice for cross-references)
    xelatex_args = ["xelatex", "-interaction=nonstopmode", "-quiet", tex_file]
    subprocess.run(xelatex_args, capture_output=True)
    subprocess.run(xelatex_args, capture_output=True)

    # Step 4: Clean up auxiliary files
    for ext in [".tex", ".aux", ".log", ".out"]:
        aux = base + ext
        if os.path.exists(aux):
            os.remove(aux)

    return os.path.exists(base + ".pdf")


def main(files, pdf_name):
    for f in files:
        notebook_to_pdf(f)
        print("Created PDF {}.".format(f))
    if MERGE:
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(pdf_name)
        merger.close()
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We pass in a explicit notebook arg so that we can provide an ordered list
    # and produce an ordered PDF.
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename)
