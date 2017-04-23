#coding:utf-8
## cited from http://aidiary.hatenablog.com/entry/20100613/1276389337
import math
import sys
import numpy as np
from collections import defaultdict

class NaiveBayes:
    """Multinomial Naive Bayes"""
    def __init__(self):
        self.categories = set()     # カテゴリの集合
        self.vocabularies = set()   # ボキャブラリの集合
        self.wordcount = {}         # wordcount[cat][word] カテゴリでの単語の出現回数
        self.catcount = {}          # catcount[cat] カテゴリの出現回数
        self.denominator = {}       # denominator[cat] P(word|cat)の分母の値
    
    def train(self, data):
        """ナイーブベイズ分類器の訓練"""
        # 文書集合からカテゴリを抽出して辞書を初期化
        for cat,d in data:
            self.categories.add(cat)
        for cat in self.categories:
            self.wordcount[cat] = defaultdict(int)
            self.catcount[cat] = 0
        # 文書集合からカテゴリと単語をカウント
        for cat,doc in data:
            self.catcount[cat] += 1
            for word in doc:
                self.vocabularies.add(word)
                self.wordcount[cat][word] += 1
        # 単語の条件付き確率の分母の値をあらかじめ一括計算しておく（高速化のため）
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
    
    def score(self, doc, cat):
        """文書が与えられたときのカテゴリの事後確率の対数 log(P(cat|doc)) を求める"""
        total = sum(self.catcount.values())  # 総文書数
        score = math.log(float(self.catcount[cat]) / total)  # log P(cat)
        for word in doc:
            # logをとるとかけ算は足し算になる
            score += math.log(self.wordProb(word, cat))  # log P(word|cat)
        return score
    
    def __str__(self):
        total = sum(self.catcount.values())  # 総文書数
        return "documents: %d, vocabularies: %d, categories: %d" % (total, len(self.vocabularies), len(self.categories))

if __name__ == "__main__":
    # Introduction to Information Retrieval 13.2の例題
    data = [["yes", ["Chinese", "Beijing", "Chinese"]],
            ["yes", ["Chinese", "Chinese", "Shanghai"]],
            ["yes", ["Chinese", "Macao"]],
            ["no",  ["Tokyo", "Japan", "Chinese"]]]
    
    # ナイーブベイズ分類器を訓練
    nb = NaiveBayes()
    nb.train(data)
    print nb
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
