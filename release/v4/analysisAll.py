# -*- coding: utf-8 -*-
from naiveBayes import NaiveBayes

nb = NaiveBayes()
nb.addTextFile    (fpath="data/UFO.csv",index_column="code4",columns_to_use=['situation', 'issue', 'risk', 'financial'], minFreq=100,codec="utf-8")
nb.loadCategoryFile(fpath="data/AIdata.csv" ,index_column="証券コード",column_to_use="名称")
nb.train(minFreq=100)
print nb
nb.save("data/trainedModel_all.pickle")

nb = NaiveBayes()

nb.load("data/trainedModel_all.pickle")
wd0 = nb.wordInfo(fpath="analysis/words_all_all.csv")
wd1 = nb.cleanupWords(wd0,exclude_path="data/excludeWords.csv")
wd2 = nb.wordInfo(fpath="analysis/words_to_use_all.csv",wordFilter=wd1,minFreq=100)

nb.TestByCompany(fpath="analysis/result_all.csv",topn=25,wordFilter=wd2,verbose=True)
#nb.nFoldTest(fpath="analysis/5FoldTest_all.csv",nFold=5,wordFilter=wd2)
#nb.nFoldTest(fpath="analysis/5FoldTest_all_countOne.csv",nFold=5,wordFilter=wd2,countOne=True)
