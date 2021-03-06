#coding:utf-8
## cited from http://aidiary.hatenablog.com/entry/20100613/1276389337
import math,sys,csv
import numpy as np
import MeCab
from collections import defaultdict
import os
import cPickle as pickle

class NaiveBayes:
    """Multinomial Naive Bayes"""
    def __init__(self):
        self.categories = set()     # カテゴリの集合
        self.vocabularies = set()   # ボキャブラリの集合
        self.wordcount = {}         # wordcount[cat][word] カテゴリでの単語の出現回数
        self.catcount = {}          # catcount[cat] カテゴリの出現回数
        self.denominator = {}       # denominator[cat] P(word|cat)の分母の値
        self.textData = {}
        self.cateData = {}
        self.cacheDir = "cache"
    
    def __str__(self):
        total = sum(self.catcount.values())
        return "documents: %d, vocabularies: %d, categories: %d" % (total, len(self.vocabularies), len(self.categories))

    def loadTextFile(self,fpath,index_column,columns_to_use=None):
        cacheName = fpath+str(index_column)+str(columns_to_use)
        cacheName = cacheName.replace("/","").replace(".","").replace("[","").replace("]","").replace(",","").replace("'","").replace(" ","")
        cacheFname = os.path.join(self.cacheDir,cacheName+".pickle")
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        if os.path.exists(cacheFname):
            with open(cacheFname,"rb") as f:
                self.textData = pickle.load(f)
            return

        self.textfile_index_column   = index_column
        self.textfile_columns_to_use = columns_to_use
        tagger = MeCab.Tagger("-Owakati") # -Ochasen -Owakati

        print "loading:%s"%fpath
        with open(fpath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_index  = header.index(index_column)
            if columns_to_use:
                idx_to_use = [header.index(x) for x in columns_to_use]
            else:
                idx_to_use = [i for i in range(len(row)) if not i == idx_index]

            for i,row in enumerate(reader):
                if i%1000==0: print "processing: %d"%i
                index = row[idx_index]
                line  = ""
                for i in idx_to_use:
                    line += row[i]+" "

                ## Wakachi-gaki
                line = tagger.parse(line)
                line = line.decode('utf-8')
                words = line.split()

                self.textData[index] = words
        with open(cacheFname,"wb") as f:
            pickle.dump(self.textData,f,protocol=2)
        return

    def loadCategoryFile(self,fpath,index_column,column_to_use):
        self.catefile_index_column   = index_column
        self.catefile_column_to_use = column_to_use

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
        return

    def train(self,wordFilter=None):
        """ナイーブベイズ分類器の訓練"""
        # データを作成
        self.data = []
        for index in self.cateData:
            if not index in self.textData:
                print "%s is not in self.textData. Skip"%index
                continue
            self.data.append([index,self.textData[index]])
        
        # 文書集合からカテゴリを抽出して辞書を初期化
        for cat,d in self.data:
            self.categories.add(cat)
        for cat in self.categories:
            self.wordcount[cat] = defaultdict(int)
            self.catcount[cat] = 0
        # 文書集合からカテゴリと単語をカウント
        for cnt,catdog in enumerate(self.data):
            if cnt%1000==0: print "train(): processing %d/%d"%(cnt,len(self.data))
            cat,doc = catdog
            self.catcount[cat] += 1
            for word in doc:
                if wordFilter and (not word in wordFilter): continue
                self.vocabularies.add(word)
                self.wordcount[cat][word] += 1
        # 単語の条件付き確率の分母の値をあらかじめ一括計算
        for cat in self.categories:
            self.denominator[cat] = sum(self.wordcount[cat].values()) + len(self.vocabularies)
    
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
    
    def wordProb(self, word, cat):
        """単語の条件付き確率 P(word|cat) を求める"""
        # ラプラススムージングを適用
        # wordcount[cat]はdefaultdict(int)なのでカテゴリに存在しなかった単語はデフォルトの0を返す
        # 分母はtrain()の最後で一括計算済み
        return float(self.wordcount[cat][word] + 1) / float(self.denominator[cat])

    def scoreDict(self,doc):
        probCat = self.scoreList(doc)
        probDict = {}
        for i, cat in enumerate(self.categories):
            probDict[cat] = probCat[i]
        return probDict

    def scoreList(self,doc):
        probCat = []
        for c in self.categories:
            probCat.append(math.exp(self.score(doc,c)))
        total = sum(probCat)
        for i,c in enumerate(self.categories):
            probCat[i] /= total
        return probCat
    
    def score(self, doc, cat):
        """文書が与えられたときのカテゴリの事後確率の対数 log(P(cat|doc)) を求める"""
        total = sum(self.catcount.values())
        score = math.log(float(self.catcount[cat]) / total)  # log P(cat)
        for word in doc:
            score += math.log(self.wordProb(word, cat))  # log P(word|cat)
        return score
    
    def wordInfo(self,fpath=None,topn=None):
        """単語の性質を出力"""
        words = set()
        for cat in self.categories:
            for w in self.wordcount[cat]:
                words.add(w)

        wordNum = {}
        wordEnt = {}
        for cnt,w in enumerate(words):
            if cnt%10000==0: print "wordInfo(): processing %d/%d"%(cnt,len(words))
            wordNum[w] = {}
            for cat in self.categories:
                wordNum[w][cat] = self.wordcount[cat][w]
            entropy = 0.
            for cat in self.categories:
                prob = float(wordNum[w][cat])/sum(wordNum[w].values())
                if prob > 0.:
                    entropy -= prob * math.log(prob)
            wordEnt[w] = entropy

        if fpath:
            f = open(fpath, 'w')
            writer = csv.writer(f, lineterminator='\n')

        cnt = 1
        for word,entropy in sorted(wordEnt.items(), key=lambda x:x[1]):
            if topn and cnt>topn: break
            if fpath:
                line = [cnt,word.encode("utf-8")]
                for c in self.categories: line.append(wordNum[word][c])
                line.append(entropy)
                writer.writerow(line)
            else:
                print "%3d  H(%20s) = %.3e"%(cnt,word,entropy)
            cnt += 1

        if fpath: f.close()

if __name__ == "__main__":
    nb = NaiveBayes()
    print "load files..."
    nb.loadTextFile    (fpath="data/mergedEDINET.csv",index_column=u"code4",columns_to_use=[u'situation', u'issue', u'risk', u'contract', u'research',u'financial'])
    print "load files..."
    nb.loadCategoryFile(fpath="data/category.csv" ,index_column=u"銘柄コード",column_to_use=u"結論")
    print "...done"
    #nb.train(wordFilter=[u"は",u"ため",u"ゴム",u"以来"])
    print "training..."
    nb.train()
    print "...done"
    print nb
    nb.wordInfo(topn=10)
    nb.wordInfo(fpath="word.csv")
    #print nb.categories
    print "log P(1|は, ため) =", nb.score([u"は",u"ため"], u"1")
    print "log P(2|は, ため) =", nb.score([u"は",u"ため"], u"2")
    print "log P(X|は, ため) =", nb.scoreDict([u"は",u"ため"])
    print
    print "log P(1|は, ため, ゴム) =", nb.score([u"は",u"ため",u"ゴム"], u"1")
    print "log P(2|は, ため, ゴム) =", nb.score([u"は",u"ため",u"ゴム"], u"2")
    print "log P(X|は, ため, ゴム) =", nb.scoreDict([u"は",u"ため",u"ゴム"])
    print "log P(X|は, ため, 以来, が) =", nb.scoreDict([u"は",u"ため",u"以来",u"が"])

    """
    print "P(Chinese|yes) = ", nb.wordProb("Chinese", "yes")
    print "P(Tokyo|yes) = ", nb.wordProb("Tokyo", "yes")
    print "P(Japan|yes) = ", nb.wordProb("Japan", "yes")
    print "P(Chinese|no) = ", nb.wordProb("Chinese", "no")
    print "P(Tokyo|no) = ", nb.wordProb("Tokyo", "no")
    print "P(Japan|no) = ", nb.wordProb("Japan", "no")
    
    # テストデータのカテゴリを予測
    test = ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]
    print "log P(yes|test) =", nb.score(test, "yes")
    print "log P(no|test) =", nb.score(test, "no")
    print nb.classify(test)
    """
