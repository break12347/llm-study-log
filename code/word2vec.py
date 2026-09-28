from gensim.models import Word2Vec
import jieba
# 1. 原始中文文章（你可以替换成自己的长文本）
raw_text = """
人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
大语言模型是人工智能自然语言处理方向的重要成果,基于Transformer架构,可以理解人类语言,生成文本,进行逻辑推理。
词向量是自然语言处理的基础技术,Word2Vec可以把词语映射到低维向量空间,语义相近的词向量距离更近。
"""
#jieba分词，切整篇文章，过滤空白字符
words=jieba.lcut(raw_text)
sentences=[words]
#训练
model=Word2Vec(
    sentences=sentences,
    vector_size=50,# 词向量维度
    window=3,#上下文窗口大小
    min_count=1,# 最低出现次数，小语料设1
    epochs=20,# 迭代轮数
    sg=0 # sg=0 CBOW()小语料；sg=1 Skip-gram（大数据）
    )
#测试
print("=== 和【人工智能】最相似的词 ===")
print(model.wv.most_similar("人工智能"))
print("\n=== 查看【机器】的词向量 ===")
print(model.wv["机器"])
print("\n=== 两个词相似度 ===")
print(model.wv.similarity("人工智能","机器"))

import jieba
from gensim.models import Word2Vec

# 读取本地txt文本
# with open("article.txt", "r", encoding="utf-8") as f:
#     content = f.read()

# words = jieba.lcut(content)
# sentences = [words]

# model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=20)
# print(model.wv.most_similar("人工智能"))
