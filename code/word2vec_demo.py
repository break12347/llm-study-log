from gensim.models import Word2Vec

# 1. 准备语料（每句话 = 一个词的列表）
sentences = [
    ["我", "爱", "北京", "天安门"],
    ["我", "爱", "上海", "外滩"],
    ["北京", "天安门", "故宫", "长城"],
    ["上海", "外滩", "东方明珠"],
]

# 2. 训练（10 秒跑完）
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, epochs=20)

# 3. 找相似词
print(model.wv.most_similar("北京"))
# 输出类似: [('天安门', 0.99), ('长城', 0.85), ('上海', 0.6), ...]
