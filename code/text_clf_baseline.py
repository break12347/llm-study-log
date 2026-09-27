# 导入 jieba 中文分词库，用于对中文句子切分词语
import jieba
# pandas用于读取本地csv数据集
import pandas as pd
# sklearn 数据集划分工具
from sklearn.model_selection import train_test_split
# 导入TF‑IDF文本向量化工具，把分词后的文本转为向量特征
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入逻辑回归分类器
from sklearn.linear_model import LogisticRegression
# 导入评估指标：准确率、分类报告
from sklearn.metrics import accuracy_score, classification_report

# ---------------------- 1. 加载本地 ChnSentiCorp 数据集 ----------------------
# csv放在py脚本同一个文件夹
df = pd.read_csv(r"E:\project important\github code\llm-study-log\code\ChnSentiCorp_htl_all.csv")
# df两列：review评论文本，label标签 0=负面，1=正面

# 分层划分：train80%，val10%，test10%；stratify保证正负标签比例不变
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# ---------------------- 2. 定义停用词集合 + 分词函数（增加停用词过滤） ----------------------
# 简易中文停用词：虚词、助词，无情感信息，过滤掉
stopwords = {"的", "了", "是", "就", "都", "而", "及", "与", "也", "和", "在", "有", "我", "他", "这", "那"}

def tokenize(text):
    # str(text)防止有空值；strip去除首尾空格；jieba.lcut精确分词返回词语列表
    words = jieba.lcut(str(text).strip())
    # 过滤停用词、过滤空字符串
    words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
    # 返回空格分隔字符串，给TF‑IDF输入
    return " ".join(words)

# ---------------------- 3. 全部文本执行分词处理 ----------------------
print("开始分词...")
train_texts = [tokenize(item) for item in train_df["review"]]
train_labels = train_df["label"].tolist()

test_texts = [tokenize(item) for item in test_df["review"]]
test_labels = test_df["label"].tolist()

# ---------------------- 4. TF‑IDF特征提取器初始化 ----------------------
tfidf = TfidfVectorizer(
    max_features=6000,        # 保留词频最高前6000个特征，控制维度爆炸
    min_df=2,                 # 词语至少出现2篇文档才保留，过滤罕见噪声词
    ngram_range=(1,2)         # 同时提取单字、双字短语，捕获中文短语语义
)

# fit_transform：训练集构建词表+转为向量
X_train = tfidf.fit_transform(train_texts)
# transform：测试集**只转换，不重新fit**，防止数据泄露
X_test = tfidf.transform(test_texts)

# ---------------------- 5. 初始化训练逻辑回归模型 ----------------------
print("开始训练模型...")
clf = LogisticRegression(
    max_iter=200,             # 增大迭代次数，保证模型收敛
    C=3.0                     # C越大正则越弱，拟合能力更强
)
clf.fit(X_train, train_labels)

# ---------------------- 6. 预测与评估 ----------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(test_labels, y_pred)
print(f"测试集准确率: {acc:.4f}")
print(classification_report(test_labels, y_pred))
