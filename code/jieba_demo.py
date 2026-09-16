import jieba
#歧义句子
sen='乒乓球拍卖完了'
print("全模式:",jieba.lcut(sen, cut_all=True))  # 全模式
print("精确模式:",jieba.lcut(sen, cut_all=False))  # 精确模式
print("搜索引擎模式:",jieba.lcut_for_search(sen))  # 搜索引擎模式
sen1="""当我们在谈论世面时，是在谈论世界的尺寸，还是在丈量生活的刻度？世面是什么？是高楼大厦，是霓虹璀璨，还是异国他乡？又或者是那些我们从未见过，
却始终在别人口中流传的远方？世面，是舞台到观众席的距离，是卧室到巨星的经历，是山脚到山顶的风景，原来长大就是带更多人看见“我”的世界。
见天地之道，阅众生之相，世面——就是世界的每一面。已识乾坤大，犹怜草木青。高考在即，祝愿所有考生笔锋所至，皆成坦途。无论笔下答卷还是远方山海，
都是等待你们解锁的“世界另一面”。"""
print("全模式:",jieba.lcut(sen1, cut_all=True))  # 全模式
print("精确模式:",jieba.lcut(sen1, cut_all=False))  # 精确模式
print("搜索引擎模式:",jieba.lcut_for_search(sen1))  # 搜索引擎模式