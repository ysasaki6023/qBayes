import openpyxl as px
import naiveBayes
import numpy as np
import pandas as pd
import csv

def loadOne(fpath):
    d = pd.read_csv(fpath,encoding="shift-jis")
    return d[["code","infer","truth"]]

if __name__=="__main__":
    columns_to_use = naiveBayes.getColumns()
    result_infer = {}
    result_truth = {}
    header = ["code"]
    for key in sorted(columns_to_use):
        print key
        try:
            ret = loadOne("analysis/%s.csv"%key)
        except:
            print "skipped:%s"%key
            continue
        header.append(columns_to_use[key])
        for i,row in ret.iterrows():
            code,truth,infer = row["code"],row["truth"],row["infer"]
            if not code in result_truth:
                result_truth[code], result_infer[code] = [code], [code]
            result_truth[code].append(truth)
            result_infer[code].append(infer)

    #del result_infer[float("nan")]
    #del result_truth[float("nan")]

    # output
    ofile_truth = open("analysis/truth.csv","w")
    ofile_infer = open("analysis/infer.csv","w")
    csv_truth   = csv.writer(ofile_truth,lineterminator='\n')
    csv_infer   = csv.writer(ofile_infer,lineterminator='\n')
    csv_truth.writerow([x.decode("utf-8").encode("shift-jis") for x in header])
    csv_infer.writerow([x.decode("utf-8").encode("shift-jis") for x in header])
    for code in sorted(result_truth):
        try: code = int(code)#float("nan"): print "error!!!"
        except: continue
        result_truth[code][0] = code
        result_infer[code][0] = code
        csv_truth.writerow([unicode(x).encode("shift-jis") for x in result_truth[code]])
        csv_infer.writerow([unicode(x).encode("shift-jis") for x in result_infer[code]])
    ofile_truth.close()
    ofile_infer.close()
