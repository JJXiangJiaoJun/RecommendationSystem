import jieba
import math
import jieba.analyse


class TF_IDF(object):
    def __init__(self, file, stop_file):
        self.file = file
        self.stop_file = stop_file
        self.stop_words = self.getStopWords()

    def getStopWords(self):
        """获取停用词表"""
        swlist = list()
        for line in open(self.stop_file, "r", encoding="utf-8").readlines():
            swlist.append(line.strip())
        print("加载停用词完成.....")
        return swlist

    def loadData(self):
        """加载商品和其对应的短标题，使用jieba进行分词并去除停用词"""
        dMap = dict()
        for line in open(self.file, "r", encoding="utf-8").readlines():
            id, title = line.strip().split("\t")
            dMap.setdefault(id, [])
            for word in list(jieba.cut(str(title).replace(" ", ""), cut_all=False)):
                if word not in self.stop_words:
                    dMap[id].append(word)
        print("加载商品和对应的短标题，并使用jieba分词和去除停用词完成....")
        return dMap

    def getFreqWord(self, words):
        """获取一个短标题中的词频"""
        freqWord = dict()
        for word in words:
            freqWord.setdefault(word, 0)
            freqWord[word] += 1
        return freqWord

    def getCountWordInFile(self, word, dMap):
        count = 0
        for key in dMap.keys():
            if word in dMap[key]:
                count += 1
        return count

    def getTFIDF(self, words, dMap):
        """计算TFIDF值，记录单词关键词和对应的tf-idf值"""
        outDic = dict()
        freqWord = self.getFreqWord(words)
        for word in words:
            # 计算TF值，即单个word在整句中出现的次数
            tf = freqWord[word] * 1.0 / len(words)
            idf = math.log(len(dMap) / self.getCountWordInFile(word, dMap) + 1)
            tfidf = tf * idf
            outDic[word] = tfidf
        # 给字典排序
        orderDic = sorted(outDic.items(), key=lambda x: x[1], reverse=True)
        return orderDic

    def getTag(self, words):
        print(jieba.analyse.extract_tags(words, topK=20, withWeight=True))


if __name__ == "__main__":
    # 数据集
    file = "../data/phone-title/id_title.txt"
    stop_file = "../data/phone-title/stop_words.txt"

    tfidf = TF_IDF(file, stop_file)
    tfidf.getTag("小米 红米6Pro 异形全面屏， 后置1200万双摄， 4000mAh超大电池")

    dMap = tfidf.loadData()
    for id in dMap.keys():
        tfIdfDic = tfidf.getTFIDF(dMap[id], dMap)
        print(id, tfIdfDic)
