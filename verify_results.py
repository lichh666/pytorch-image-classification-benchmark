"""Recompute published classification metrics without PyTorch or dataset access."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify():
    rows=list(csv.DictReader((ROOT/'results/final_test/comparison.csv').open(encoding='utf-8-sig',newline='')))
    assert len(rows)==11
    for row in rows:
        name=row['experiment']; folder=ROOT/'results/final_test'/name
        record=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        cm=[[0]*10 for _ in range(10)]
        predictions=list(csv.DictReader((folder/'predictions.csv').open(encoding='utf-8-sig',newline='')))
        assert len(predictions)==3929 and len({r['path'] for r in predictions})==3929
        for r in predictions: cm[int(r['target'])][int(r['prediction'])]+=1
        assert cm==record['official_test']['confusion_matrix']
        total=sum(map(sum,cm)); accuracy=sum(cm[i][i] for i in range(10))/total
        precision=[cm[i][i]/max(sum(cm[j][i] for j in range(10)),1) for i in range(10)]
        recall=[cm[i][i]/max(sum(cm[i]),1) for i in range(10)]
        f1=[2*p*r/(p+r) if p+r else 0 for p,r in zip(precision,recall)]
        for key,value in [('accuracy',accuracy),('precision_macro',sum(precision)/10),('recall_macro',sum(recall)/10),('f1_macro',sum(f1)/10)]:
            assert math.isclose(value,float(row['test_'+key]),rel_tol=1e-12,abs_tol=1e-12),(name,key)
            assert math.isclose(value,record['official_test'][key],rel_tol=1e-12,abs_tol=1e-12)
        assert record['config']['pretrained'] is False
        print(f'{name}: n={total}, accuracy={accuracy:.8%}, macro_F1={sum(f1)/10:.8%}')
    assert [r['experiment'] for r in rows if r['selected_primary_model']=='True']==['S1']
    print('PASS: all 11 prediction files, confusion matrices and four aggregate metrics agree.')

if __name__=='__main__':verify()
