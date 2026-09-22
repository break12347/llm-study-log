from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 数据（先用 sklearn 自带的"20 分类新闻"代替中文，省掉加载步骤）
from sklearn.datasets import fetch_20newsgroups#从sklearn.datasets库中导入fetch_20newsgroups
data = fetch_20newsgroups(subset='train',
                          categories=['sci.med','rec.autos'],
                          shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# 2. 向量化：文本 → 数字
vec = TfidfVectorizer(max_features=5000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec  = vec.transform(X_test)#将X_test转换

# 3. 训练：分类器学"什么词属于哪一类"
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)#clf学习X_train_vec,y_train

# 4. 考试
y_pred = clf.predict(X_test_vec)#clf根据X_test_vec回答
print("准确率:", accuracy_score(y_test, y_pred))#得到准确率

# 5. 试玩
test_text = ["This is about a patient with diabetes and new treatment."]
print("预测类别:", data.target_names[clf.predict(vec.transform(test_text))[0]])