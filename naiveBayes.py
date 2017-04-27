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

        return

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
                        if not word.isdigit():
                            words.append(word)
                    line = line.next

                self.textData[index] = words
        with open(cacheFname,"wb") as f:
            pickle.dump(self.textData,f,protocol=2)
        return

    def loadCategoryFile(self,fpath,index_column,column_to_use):
        print "loading category file:%s, categoryColumn:%s"%(fpath,column_to_use)
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

    def train(self,minfreq=None,wordFilter=None):
        """ナイーブベイズ分類器の訓練"""
        print "train()"
        # データを作成
        self.data = []
        for index in self.cateData:
            if not index in self.textData:
                print "%s is not in self.textData. Skip"%index
                continue
            self.data.append([self.cateData[index],self.textData[index]])
        
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
                if wordFilter and (not word in wordFilter): continue
                if not word in self.vocabcount: self.vocabcount[word] = 0
                self.vocabcount[word] += 1
        origNum = len(self.vocabcount)
        print
        if minfreq:
            dictList = self.vocabcount.keys()
            for w in dictList:
                if self.vocabcount[w]<minfreq:
                    del self.vocabcount[w]
            print "..found %d words originally, which is reduced to %d under minfreq=%d"%(origNum,len(self.vocabcount),minfreq)
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
    
    def wordProb(self, word, cat):
        """単語の条件付き確率 P(word|cat) を求める"""
        # 単語の条件付き確率の分母の値をあらかじめ一括計算
        oneNumFlag = False
        if not type(word) == type([]): word,oneNumFlag = [word],True
        if type(self.denominator) == type(None):
            self.denominator = np.zeros(len(self.categories),dtype=np.float32)
            for cat in self.categories:
                idx_cat = self.categories.index(cat)
                self.denominator[idx_cat] = np.sum(self.np_words[idx_cat]) + len(self.vocabularies)
        # ラプラススムージングを適用
        # wordcount[cat]はdefaultdict(int)なのでカテゴリに存在しなかった単語はデフォルトの0を返す
        wd_idx = np.zeros(len(word),dtype=np.int32)
        mk_idx = np.zeros(len(word),dtype=np.bool)
        for i,w in enumerate(word):
            if w in self.vocab_lookup:
                idx = self.vocab_lookup[w]
                mk_idx[i] = False
            else:
                idx = 0 # dummy
                mk_idx[i] = True
            wd_idx[i] = idx

        nominator = self.np_words[self.categories.index(cat),wd_idx]
        nominator[mk_idx] = 0.
        val =  (nominator + 1.) / self.denominator[self.categories.index(cat)]

        if oneNumFlag: return val[0]
        else         : return val


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
        score  = np.log(self.np_categ[self.categories.index(cat)])/np.sum(self.np_categ)  # log P(cat)
        score += np.sum( np.log(self.wordProb( doc, cat ) ) ) # sum log P(doc|cat)
        return score
    
    def wordInfo(self,fpath=None,topn=None,minfreq=None):
        """単語の性質を出力"""
        prob = self.np_words.astype(np.float32) / np.sum(self.np_words,axis=0)
        entropy = - prob * np.ma.log(prob)
        entropy = np.sum(entropy, axis=0)

        if fpath:
            f = open(fpath, 'w')
            writer = csv.writer(f, lineterminator='\n')

        mask = np.zeros(len(self.vocabularies),dtype=bool)
        mask[:] = True
        if minfreq: mask[ np.sum(self.np_words,axis=0) < minfreq ] = False

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

        for i in range(len(wordidx)):
            if fpath:
                line = [i,self.vocabularies[wordidx[i]].encode("utf-8")] # vocabularies -> need to sort, thus use wordidx
                for j in range(len(self.categories)): line.append(wcounts[j,i]) # wcounts -> already sorted
                line.append(entropy[i]) # entropy -> already sorted
                writer.writerow(line)
            else:
                print "%8d  H(%20s) = %.3e"%(i,self.vocabularies[wordidx[i]],entropy[i])

        if fpath: f.close()

if __name__ == "__main__":
    """
    nb = NaiveBayes()
    nb.loadTextFile    (fpath="data/mergedEDINET.csv",index_column=u"code4",columns_to_use=[u'situation', u'issue', u'risk', u'contract', u'research',u'financial'])
    nb.loadCategoryFile(fpath="data/category.csv" ,index_column=u"銘柄コード",column_to_use=u"結論")
    print "...done"
    #nb.train(wordFilter=[u"は",u"ため",u"ゴム",u"以来"])
    nb.train(minfreq=100)
    print nb
    nb.save("test.pickle")
    """

    nb2 = NaiveBayes()
    nb2.load("test.pickle")
    print nb2
    nb2.wordInfo(topn=10)
    nb2.wordInfo(fpath="word.csv",minfreq=100)
    #print nb.categories
    print "log P(1|暖冬) =", nb2.score([u"暖冬"], u"1")
    print "log P(X|暖冬) =", nb2.scoreDict([u"暖冬"])
    #print "log P(2|は, ため) =", nb.score([u"は",u"ため"], u"2")
    #print "log P(X|は, ため) =", nb.scoreDict([u"は",u"ため"])
    #print
    #print "log P(1|は, ため, ゴム) =", nb.score([u"は",u"ため",u"ゴム"], u"1")
    #print "log P(2|は, ため, ゴム) =", nb.score([u"は",u"ため",u"ゴム"], u"2")
    #print "log P(X|は, ため, ゴム) =", nb.scoreDict([u"は",u"ため",u"ゴム"])
    #print "log P(X|は, ため, 以来, が) =", nb.scoreDict([u"は",u"ため",u"以来",u"が"])
