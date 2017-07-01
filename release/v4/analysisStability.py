# -*- coding: utf-8 -*-
from naiveBayes import NaiveBayes

nb = NaiveBayes()
nb.addTextFile    (fpath="data/20170531_bunrui_Kari.csv",index_column="銘柄コード",columns_to_use=['売上モデル', '営業利益モデル'], applyMecab=False, minFreq=None, codec="utf-8")
nb.addTextFile    (fpath="data/UFO.csv",index_column="code4",columns_to_use=['situation', 'risk'], minFreq=100,codec="utf-8")
nb.loadCategoryFile(fpath="data/AIdata.csv" ,index_column="証券コード",column_to_use="安定性")
nb.train(minFreq=100)
print nb
nb.save("data/trainedModel_stability.pickle")

nb = NaiveBayes()

nb.load("data/trainedModel_stability.pickle")
wd0 = nb.wordInfo(fpath="analysis/words_all_stability.csv")
wd1 = nb.cleanupWords(wd0,exclude_path="data/excludeWords.csv")
wd2 = nb.wordInfo(fpath="analysis/words_to_use_stability.csv",wordFilter=wd1,minFreq=100)

nb.TestByCompany(fpath="analysis/result_stability.csv",topn=25,wordFilter=wd2,verbose=True)
nb.nFoldTest(fpath="analysis/5FoldTest_stability.csv",nFold=5,wordFilter=wd2)
#nb.nFoldTest(fpath="analysis/5FoldTest_stability_countOne.csv",nFold=5,wordFilter=wd2,countOne=True)
