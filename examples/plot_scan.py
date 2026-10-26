"""Plot scan.csv from GratingBEM. Requires optional matplotlib: pip install '.[plot]'"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('scan_csv',help='scan.csv created by .save()')
parser.add_argument('--out',default='scan_efficiencies.png')
args=parser.parse_args()
with Path(args.scan_csv).open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
for pol in ('TE','TM'):
    valid=[(i,float(row[pol+'_target_pct'])) for i,row in enumerate(rows)
           if row.get(pol+'_target_pct') not in (None,'')]
    if valid:plt.plot([k for k,v in valid],[v for k,v in valid],'-o',label=pol)
plt.xticks(range(len(rows)),[str(row.get('scan_value',i)) for i,row in enumerate(rows)],rotation=45)
plt.ylabel('Target-order efficiency (%)')
plt.xlabel(rows[0].get('scan_parameter','Scan value') if rows else 'Scan value')
plt.legend();plt.tight_layout();plt.savefig(args.out,dpi=160)
print(args.out)
