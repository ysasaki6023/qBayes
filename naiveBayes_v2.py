#coding:utf-8
## cited from http://aidiary.hatenablog.com/entry/20100613/1276389337
## 2017/05/14 v1.1 中分類を再現するよう学習・評価
import math,sys,csv,re
import numpy as np
import MeCab
from collections import defaultdict
import os
import cPickle as pickle
import pandas as pd

class NaiveBayes:
    """Multinomial Naive Bayes"""
    def __init__(self):
        self.wordcount = None         # wordcount[cat][word] カテゴリでの単語の出現回数
        self.catcount = None          # catcount[cat] カテゴリの出現回数
        self.denominator = None       # denominator[cat] P(word|cat)の分母の値
        self.textData = {}
        self.cateData = {}
        self.cacheDir = "cache"
        self.data = None
    
    def __str__(self):
        return "vocabularies: %d, categories: %d" % (len(self.vocabularies), len(self.categories))

    def save(self,fpath):
        p = {}
        p["vocabularies"] = self.vocabularies
        p["categories"]   = self.categories
        p["vocab_lookup"] = self.vocab_lookup
        p["np_words"]     = self.np_words
        p["np_categ"]     = self.np_categ
        p["textFile_fpath"] = self.textFile_fpath
        p["textFile_index_column"]  = self.textFile_index_column
        p["textFile_columns_to_use"] = self.textFile_columns_to_use
        p["cateFile_fpath"] = self.cateFile_fpath
        p["cateFile_index_column"]  = self.cateFile_index_column
        p["cateFile_column_to_use"] = self.cateFile_column_to_use

        with open(fpath,"wb") as f:
            pickle.dump(p,f,protocol=2)
        return

    def load(self,fpath):
        with open(fpath,"rb") as f:
            p = pickle.load(f)
        self.vocabularies = p["vocabularies"]
        self.categories   = p["categories"]
        self.vocab_lookup = p["vocab_lookup"]
        self.np_words     = p["np_words"]
        self.np_categ     = p["np_categ"]
        self.textFile_fpath         = p["textFile_fpath"]
        self.textFile_index_column  = p["textFile_index_column"]
        self.textFile_columns_to_use = p["textFile_columns_to_use"]
        self.cateFile_fpath         = p["cateFile_fpath"]
        self.cateFile_index_column  = p["cateFile_index_column"]
        self.cateFile_column_to_use = p["cateFile_column_to_use"]

        return

    def loadTextFile(self,fpath,index_column,columns_to_use=None):

        self.textFile_fpath          = fpath
        self.textFile_index_column   = index_column
        self.textFile_columns_to_use = columns_to_use

        cacheName = fpath+index_column.encode("utf-8")+"".join([x.encode("utf-8") for x in columns_to_use])+"_text"
        cacheName = cacheName.replace("/","").replace(".","").replace("[","").replace("]","").replace(",","").replace("'","").replace(" ","")
        cacheFname = os.path.join(self.cacheDir,cacheName+".pickle")
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        if os.path.exists(cacheFname):
            with open(cacheFname,"rb") as f:
                self.textData = pickle.load(f)
            return

        self.textFile_index_column   = index_column
        self.textFile_columns_to_use = columns_to_use
        tagger = MeCab.Tagger("-Ochasen")

        print "loading text file:%s"%fpath
        with open(fpath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_index  = header.index(index_column)
            if columns_to_use:
                idx_to_use = [header.index(x) for x in columns_to_use]
            else:
                idx_to_use = [i for i in range(len(row)) if not i == idx_index]

            for i,row in enumerate(reader):
                if i%1000==0: print "..loading: %d"%i
                index = row[idx_index]
                line  = ""
                for i in idx_to_use:
                    line += row[i]+" "

                ## Wakachi-gaki
                line = tagger.parseToNode(line)
                # Extract word and class
                words = []
                while line:
                    word = line.surface.decode("utf-8", "ignore")
                    clazz = line.feature.split(',')[0].decode('utf-8', 'ignore')
                    if clazz==u"名詞" and clazz != u'BOS/EOS':
                        if word.isdigit(): continue
                        words.append(word)

                    line = line.next

                self.textData[index] = words
        with open(cacheFname,"wb") as f:
            pickle.dump(self.textData,f,protocol=2)
        return

    def loadCategoryFile(self,fpath,index_column,column_to_use):
        self.cateFile_fpath          = fpath
        self.cateFile_index_column   = index_column
        self.cateFile_column_to_use  = column_to_use

        print "loading category file:%s, categoryColumn:%s"%(fpath,column_to_use)
        self.catefile_index_column   = index_column
        self.catefile_column_to_use = column_to_use

        cacheName = fpath+index_column.encode("utf-8")+column_to_use.encode("utf-8")+"_cate"
        cacheName = cacheName.replace("/","").replace(".","").replace("[","").replace("]","").replace(",","").replace("'","").replace(" ","")
        cacheFname = os.path.join(self.cacheDir,cacheName+".pickle")
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        if os.path.exists(cacheFname):
            with open(cacheFname,"rb") as f:
                self.cateData = pickle.load(f)
            return

        with open(fpath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [x.decode("utf-8") for x in header]
            idx_index  = header.index(index_column)
            idx_to_use = header.index(column_to_use)

            for row in reader:
                index = row[idx_index]
                cate  = row[idx_to_use]

                self.cateData[index] = cate

        with open(cacheFname,"wb") as f:
            pickle.dump(self.cateData,f,protocol=2)

        return

    def buildData(self,verbose=False):
        # データを作成
        self.data = []
        for index in self.cateData:
            if not index in self.textData:
                if verbose: print "%s is not in self.textData. Skip"%index
                continue
            self.data.append([self.cateData[index],self.textData[index]])
        return

    def train(self,minFreq=None):
        """ナイーブベイズ分類器の訓練"""
        print "train()"
        self.buildData(verbose=True)
        
        # 文書集合からカテゴリを抽出して辞書を初期化
        self.categories = set()
        for cat,d in self.data:
            self.categories.add(cat)
        self.categories = list(self.categories)

        # 全単語数を計算
        print
        self.vocabcount   = {}
        for cnt,catdog in enumerate(self.data):
            if cnt%1000==0: print "..counting words %d/%d"%(cnt,len(self.data))
            _,doc = catdog
            for word in doc:
                if not word in self.vocabcount: self.vocabcount[word] = 0
                self.vocabcount[word] += 1
        origNum = len(self.vocabcount)
        print
        if minFreq:
            dictList = self.vocabcount.keys()
            for w in dictList:
                if self.vocabcount[w]<minFreq:
                    del self.vocabcount[w]
            print "..found %d words originally, which is reduced to %d under minFreq=%d"%(origNum,len(self.vocabcount),minFreq)
        else:
            print "..found %d words"%origNum
        self.vocabularies = list(self.vocabcount.keys())
        self.vocab_lookup = {w:i for i,w in enumerate(self.vocabularies)}

        # vocabulary 配列を作成
        self.np_words = np.zeros((len(self.categories),len(self.vocabularies)),dtype=np.int32)
        self.np_categ = np.zeros(len(self.categories),dtype=np.int32)

        # 文書集合からカテゴリと単語をカウント
        print
        for cnt,catdog in enumerate(self.data):
            if cnt%100==0: print "..building word count map: %d/%d"%(cnt,len(self.data))
            cat,doc = catdog
            idx_cat = self.categories.index(cat)
            self.np_categ[idx_cat] += 1
            for word in doc:
                if not word in self.vocab_lookup: continue
                idx_word = self.vocab_lookup[word]
                self.np_words[idx_cat][idx_word] += 1
        return
    
    def classify(self, doc):
        """事後確率の対数 log(P(cat|doc)) がもっとも大きなカテゴリを返す"""
        best = None
        mmax = -sys.maxint
        for cat in self.catcount.keys():
            p = self.score(doc, cat)
            if p > mmax:
                mmax = p
                best = cat
        return best
    
    def wordProb(self, word, cat, wordFilter):
        """単語の条件付き確率 P(word|cat) を求める"""
        # 単語の条件付き確率の分母の値をあらかじめ一括計算
        oneNumFlag = False
        if not type(word) == type([]): word,oneNumFlag = [word],True
        if type(self.denominator) == type(None):
            self.denominator = np.zeros(len(self.categories),dtype=np.float32)
            for cat in self.categories:
                idx_cat = self.categories.index(cat)
                self.denominator[idx_cat] = np.sum(self.np_words[idx_cat]) + len(self.vocabularies)
        # ラプラススムージング
        wd_idx = np.zeros(len(word),dtype=np.int32)
        mk_idx = np.zeros(len(word),dtype=np.bool)
        if wordFilter: wordFilter_lookup = {w:True for w in wordFilter}
        else         : wordFilter_lookup = None
        for i,w in enumerate(word):
            idx = 0 # dummy
            mk_idx[i] = True
            if w in self.vocab_lookup:
                if (not wordFilter_lookup) or (wordFilter_lookup and w in wordFilter_lookup):
                    idx = self.vocab_lookup[w]
                    mk_idx[i] = False
            wd_idx[i] = idx

        nominator = self.np_words[self.categories.index(cat),wd_idx]
        nominator[mk_idx] = 0.
        val =  (nominator + 1.) / self.denominator[self.categories.index(cat)]

        if oneNumFlag: return val[0]
        else         : return val

    def scoreDict(self,doc,wordFilter=None):
        scoreCat = self.scoreList(doc,wordFilter)
        scoreDict = {}
        for i, cat in enumerate(self.categories):
            scoreDict[cat] = scoreCat[i]
        return scoreDict

    def probDict(self,doc,wordFilter=None):
        probCat = self.probList(doc,wordFilter)
        probDict = {}
        for i, cat in enumerate(self.categories):
            probDict[cat] = probCat[i]
        return probDict

    def scoreList(self,doc,wordFilter=None):
        scoreCat = []
        for c in self.categories:
            scoreCat.append(self.score(doc,c,wordFilter))
        return scoreCat

    def probList(self,doc,wordFilter=None):
        probCat = []
        for c in self.categories:
            probCat.append(math.exp(self.score(doc,c,wordFilter)))
        total = sum(probCat)
        for i,c in enumerate(self.categories):
            probCat[i] /= total
        return probCat
    
    def score(self, doc, cat, wordFilter=None):
        """文書が与えられたときのカテゴリの事後確率の対数 log(P(cat|doc)) を求める"""
        #doc = self.applyFilter(doc, wordFilter)
        score  = np.log(self.np_categ[self.categories.index(cat)])/np.sum(self.np_categ)  # log P(cat)
        score += np.sum( np.log(self.wordProb( doc, cat, wordFilter ) ) ) # sum log P(doc|cat)
        return score

    def applyFilter(self, doc, wordFilter=None):
        if not wordFilter: return doc
        dd = []
        for w in doc:
            if w in wordFilter: dd.append(w)
        return dd

    def test(self,x,wordFilter):
        sd = self.scoreDict(x,wordFilter)
        y  = max(sd.items(), key=lambda x:x[1])[0]
        return y
    
    def evaluate(self, wordFilter=None, fpath=None):
        print "evaluate()"
        print "..load data"
        if not self.textData: self.loadTextFile(self.textFile_fpath,self.textFile_index_column,self.textFile_columns_to_use)
        if not self.cateData: self.loadCategoryFile(self.cateFile_fpath,self.cateFile_index_column,self.cateFile_column_to_use)
        if not self.data    : self.buildData()

        nCat = len(self.categories)
        mat = np.zeros((nCat,nCat),dtype=np.int32)

        print
        for cnt, catdog in enumerate(self.data):
            if cnt%100==0: print "..%d"%cnt
            t, x = catdog
            y = self.test(x,wordFilter)
            mat[self.categories.index(t),self.categories.index(y)] += 1

        print
        print mat

        if fpath:
            with open(fpath,"w") as f:
                w = csv.writer(f)
                line = [""] + [self.categories[x] for x in range(nCat)]
                w.writerow(line)
                for k in range(nCat):
                    line = [self.categories[k]] + [mat[k,j] for j in range(nCat)]
                    w.writerow(line)

        pass

    
    def wordInfo(self,fpath=None,topn=None,minFreq=None,maxEntropy=None, wordFilter=None):
        """単語の性質を出力"""
        prob1 = self.np_words.astype(np.float32) / np.sum(self.np_words,axis=0)
        entropy1 = - prob1 * np.ma.log(prob1)
        entropy1 = np.sum(entropy1, axis=0)
        # カテゴリレベルでそもそも発生しているエントロピーの補正
        prob0 = self.np_categ.astype(np.float32) / np.sum(self.np_categ)
        entropy0 = - prob0 * np.ma.log(prob0)
        entropy0 = np.sum(entropy0)

        entropy  = entropy1 - entropy0

        if fpath:
            f = open(fpath, 'w')
            writer = csv.writer(f, lineterminator='\n')
            line = ["index","word","entropy"] # vocabularies -> need to sort, thus use wordidx
            for j in range(len(self.categories)): line.append(self.categories[j])
            for j in range(len(self.categories)): line.append("P("+self.categories[j]+"|word)")
            writer.writerow(line)
            line = ["","category",entropy0] # vocabularies -> need to sort, thus use wordidx
            for j in range(len(self.categories)): line.append(self.np_categ[j])
            writer.writerow(line)

        mask = np.zeros(len(self.vocabularies),dtype=bool)
        mask[:] = True
        if minFreq: mask[ np.sum(self.np_words,axis=0) < minFreq ] = False
        if maxEntropy: mask[ entropy > maxEntropy ] = False

        wcounts = self.np_words[:,mask]
        wordidx = np.linspace(0,self.np_words.shape[1]-1,num=self.np_words.shape[1],dtype=np.int32)[mask]
        entropy = entropy[mask]

        idx_sorted = np.argsort(entropy)
        wcounts = wcounts[:,idx_sorted]
        wordidx = wordidx[idx_sorted]
        entropy = entropy[idx_sorted]

        if topn:
            wcounts = wcounts[:,:topn]
            wordidx = wordidx[:topn]
            entropy = entropy[:topn]

        cnt = -1
        for i in range(len(wordidx)):
            if wordFilter and (not self.vocabularies[wordidx[i]] in wordFilter): continue
            cnt += 1
            if fpath:
                line = [cnt,self.vocabularies[wordidx[i]].encode("utf-8")] # vocabularies -> need to sort, thus use wordidx
                line.append(entropy[i]) # entropy -> already sorted
                for j in range(len(self.categories)): line.append(wcounts[j,i]) # wcounts -> already sorted
                for j in range(len(self.categories)): line.append(wcounts[j,i].astype(np.float32)/np.sum(wcounts[:,i]))
                writer.writerow(line)
            else:
                print "%8d  H(%20s) = %.3e"%(cnt,self.vocabularies[wordidx[i]],entropy[i])

        if fpath: f.close()

        return [self.vocabularies[wordidx[i]] for i in range(len(wordidx))]

    def CountWords(self,index,topn=None,wordFilter=None,verbose=True):
        if not self.textData:
            print "please execute loadTextFile() first"
            return
        if not type(index)==type([]):
            index = [index]
        # 全単語数を計算
        self.vocabcount   = {}
        for idx in index:
            if not idx in self.textData:
                print "%s not in self.textData. Skip"%idx
                continue
            for word in self.textData[idx]:
                if wordFilter:
                    if not word in wordFilter: continue
                if not word in self.vocabcount: self.vocabcount[word] = 0
                self.vocabcount[word] += 1
        origNum = len(self.vocabcount)

        ret = []
        count = -1
        for w,num in sorted(self.vocabcount.items(), key=lambda x: x[1])[::-1]:
            count += 1
            if topn and count>=topn: break
            ret.append( (w,num) )

        if verbose:
            for w,n in ret:
                print w,n

        return ret

    def TestByCompany(self,fpath,topn=10,wordFilter=None):
        print "TestByCompany()"
        print "load data.."
        if not self.textData: self.loadTextFile(self.textFile_fpath,self.textFile_index_column,self.textFile_columns_to_use)
        if not self.cateData: self.loadCategoryFile(self.cateFile_fpath,self.cateFile_index_column,self.cateFile_column_to_use)
        if not self.data    : self.buildData()
        print "..done"

        f = open(fpath,"w")
        writer = csv.writer(f, lineterminator='\n')
        line = ["code"] + ["truth"] + ["infer"] + ["word%d"%i for i in range(topn)] + ["count%d"%i for i in range(topn)]
        writer.writerow(line)


        targetIndex = []
        for index in self.textData:
            targetIndex.append([index,[index]])

        cnt = -1
        for title,index in targetIndex:
            cnt += 1
            if cnt%10 == 0: print "%d/%d"%(cnt,len(targetIndex))
            truth = self.data[cnt][0]
            infer = self.test(self.textData[index[0]],wordFilter=wordFilter)
            words = self.CountWords(index,topn=topn,wordFilter=wordFilter,verbose=False)
            line = [title,truth,infer]

            for w in words: line.append(w[0].encode("utf-8"))
            if len(words)<topn:
                for i in range(topn-len(words)): line.append("")

            for w in words: line.append(w[1])
            if len(words)<topn:
                for i in range(topn-len(words)): line.append(0)

            writer.writerow(line)
        f.close()
        return

    def dumpWordsByCompany(self,fpath,classList=None,topn=10,wordFilter=None):
        print "TestByCompany()"
        print "load data.."
        if not self.textData: self.loadTextFile(self.textFile_fpath,self.textFile_index_column,self.textFile_columns_to_use)
        print "..done"

        f = open(fpath,"w")
        writer = csv.writer(f, lineterminator='\n')
        line = ["code"] + ["word%d"%i for i in range(topn)] + ["count%d"%i for i in range(topn)]
        writer.writerow(line)

        cnt = -1

        targetIndex = []
        if not classList:
            for index in self.textData:
                targetIndex.append([index,[index]])
        else:
            targetIndex = classList

        for title,index in targetIndex:
            cnt += 1
            if cnt%10 == 0: print "%d/%d"%(cnt,len(targetIndex))
            words = self.CountWords(index,topn=topn,wordFilter=wordFilter,verbose=False)
            line = [title]

            for w in words: line.append(w[0].encode("utf-8"))
            if len(words)<topn:
                for i in range(topn-len(words)): line.append("")

            for w in words: line.append(w[1])
            if len(words)<topn:
                for i in range(topn-len(words)): line.append(0)

            writer.writerow(line)
        f.close()
        return

    def cleanupWords(self,wordFilter):
        symbolReg1 = re.compile(u'[︰-＠①-⑨Ⅰ-Ⅹ]|△|㈱|－|＜|＞|;|・|ｍ') # 全角記号
        symbolReg2 = re.compile(r'[!-/:-@\[-`{-~]|&|\*|#|;') # 半角記号
        symbolReg3 = re.compile(r'^[a-zA-Z]$') # 半角英数一文字
        goodWords = []
        for w in wordFilter:
            if symbolReg1.search(w): continue
            if symbolReg2.search(w): continue
            if symbolReg3.search(w): continue
            if w in [u"万",u"億",u"兆",u"付",u"買" ]: continue
            goodWords.append(w)
        return goodWords

def loadClass(fpath,index):
    d = pd.read_csv(fpath)
    cls = {}
    for i, row in d.iterrows():
        c = row["分類"]
        if not c in cls:
            cls[c] = []
        v = row[index]
        cls[c].append(str(v))
    for c in cls:
        print c,cls[c]

    return cls.items()


if __name__ == "__main__":
    #nb = NaiveBayes()
    #nb.loadTextFile    (fpath="data/mergedEDINET.csv",index_column=u"code4",columns_to_use=[u'situation', u'issue', u'risk', u'financial'])
    #nb.loadCategoryFile(fpath="data/category.csv" ,index_column=u"銘柄コード",column_to_use=u"NEEDS業種名-中分類･主業種")
    #nb.train(minFreq=100)
    #print nb
    #nb.save("data/analysis_textcate_basic_v2.pickle")

    nb = NaiveBayes()
    nb.load("data/analysis_textcate_basic_v2.pickle")
    wd = nb.wordInfo(fpath="analysis/v2/wordInfo_maxEntropy-0.4490_beforeCleaning.csv")
    wd = nb.cleanupWords(wd)
    wd = nb.wordInfo(fpath="analysis/v2/wordInfo_maxEntropy-0.4490_afterCleaning.csv" , maxEntropy=-0.4490, wordFilter=wd)
    w = nb.dumpWordsByCompany("analysis/v2/wordsByCompany_maxEntropy-0.4490.csv" ,topn=25,wordFilter=wd)
    w = nb.dumpWordsByCompany("analysis/v2/wordsByCategory_maxEntropy-0.4490.csv",topn=25,wordFilter=wd,classList=loadClass("data/company_class.csv",index="分類"))
    nb.TestByCompany(fpath="analysis/v2/test_by_company.csv",wordFilter=None)
    #nb.evaluate(fpath="analysis/v2/mat_basic_maxEntropy-0.4490.csv",wordFilter=wd)
