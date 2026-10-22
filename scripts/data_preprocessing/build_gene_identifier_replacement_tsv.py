#!/usr/bin/env python
import argparse
import csv
import re
from pathlib import Path

def nk(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def find_col(cols, names):
    m = {nk(c): c for c in cols}
    for n in names:
        if nk(n) in m:
            return m[nk(n)]
    for c in cols:
        ck = nk(c)
        for n in names:
            if nk(n) in ck:
                return c
    raise RuntimeError("missing column %s in %s" % (names, cols))

def replace_gene(text, gene, repl):
    text = "" if text is None else str(text)
    gene = "" if gene is None else str(gene)
    repl = "" if repl is None else str(repl)
    if not gene:
        return text
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(gene) + r"(?![A-Za-z0-9_])", re.I)
    return pat.sub(repl, text)

def load_map(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        gene_col = find_col(cols, ["gene", "symbol", "gene_symbol"])
        id_col = find_col(cols, ["entrez", "entrezid", "entrez_id", "geneid", "gene_id"])
        d = {}
        for row in r:
            g = str(row.get(gene_col, "")).strip()
            e = str(row.get(id_col, "")).strip()
            if g and e:
                d[g] = e
        return d

def write_tsv(path, cols, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--entrez-map", required=True)
    ap.add_argument("--output-entrez-tsv", required=True)
    ap.add_argument("--output-random-tsv", required=True)
    ap.add_argument("--random-map-csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    entrez = load_map(a.entrez_map)

    with open(a.input_tsv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols = r.fieldnames or []
        gene_col = find_col(cols, ["gene"])
        sent_col = find_col(cols, ["sentence", "text"])
        rows = list(r)

    genes = sorted({str(x.get(gene_col, "")).strip() for x in rows if str(x.get(gene_col, "")).strip()})
    random_map = {g: "gene_%05d" % (i + 1) for i, g in enumerate(genes)}

    entrez_rows = []
    random_rows = []
    entrez_replaced = 0
    random_replaced = 0

    for row in rows:
        g = str(row.get(gene_col, "")).strip()

        er = dict(row)
        old = str(er.get(sent_col, ""))
        new = replace_gene(old, g, entrez.get(g, g))
        if new != old:
            entrez_replaced += 1
        er[sent_col] = new
        entrez_rows.append(er)

        rr = dict(row)
        old = str(rr.get(sent_col, ""))
        new = replace_gene(old, g, random_map.get(g, g))
        if new != old:
            random_replaced += 1
        rr[sent_col] = new
        random_rows.append(rr)

    write_tsv(a.output_entrez_tsv, cols, entrez_rows)
    write_tsv(a.output_random_tsv, cols, random_rows)

    Path(a.random_map_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.random_map_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "random_alias"])
        for g in genes:
            w.writerow([g, random_map[g]])

    missing = [g for g in genes if g not in entrez]
    print("genes", len(genes))
    print("rows", len(rows))
    print("missing_entrez", len(missing))
    print("entrez_replaced_rows", entrez_replaced)
    print("random_replaced_rows", random_replaced)
    print("wrote", a.output_entrez_tsv)
    print("wrote", a.output_random_tsv)
    print("wrote", a.random_map_csv)

if __name__ == "__main__":
    main()

