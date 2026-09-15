#!/usr/bin/env python
from pathlib import Path
import os, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'data'/'medical_mcq.csv'
print('Repository:',ROOT)
print('OPENAI_API_KEY configured:',bool(os.getenv('OPENAI_API_KEY')))
print('Dataset exists:',path.exists(),path)
if path.exists():
    df=pd.read_csv(path)
    print('Rows:',len(df),'Columns:',list(df.columns))
    missing={'instruction','input','output'}-set(df.columns)
    if missing: print('ERROR missing columns:',sorted(missing)); sys.exit(2)
else:
    print('Place your study CSV at data/medical_mcq.csv before running the experiment.')
    sys.exit(1)
print('Setup looks ready.')
