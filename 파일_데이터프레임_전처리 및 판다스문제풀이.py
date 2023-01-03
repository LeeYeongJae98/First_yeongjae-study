import pandas as pd

df = pd.read_csv("auto-mpg.csv", header =0)
print(df.head())

#%%
#print(df.info())
#%%
import numpy as np
df.describe(include=[np.number],
            percentiles = [.01,.05,.10,.25,.5,.75,.9,.95,.99]).T

#print(df.describe)

df.describe(include='all')
print(df.describe)
#%%
dataFrame1 =  pd.DataFrame({ 'StudentID': [1, 3, 5, 7, 9, 11, 13, 15, 
                                           17, 19, 21, 23, 25, 27, 29], 
                            'Score' : [89, 39, 50, 97, 22, 66, 31, 51, 
                                       71, 91, 56, 32, 52, 73, 92]})
dataFrame2 =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 
                                          18, 20, 22, 24, 26, 28, 30], 
                            'Score': [98, 93, 44, 77, 69, 56, 31, 53, 
                                      78, 93, 56, 77, 33, 56, 27]})
dataFrame = pd.concat([dataFrame1, dataFrame2], ignore_index= True)
print(dataFrame)
#%%
df1SE =  pd.DataFrame({ 'StudentID': [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], 
                       'ScoreSE' : [22, 66, 31, 51, 71, 91, 56, 32, 52, 73, 92]})
df2SE =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 
                                     18, 20, 22, 24, 26, 28, 30], 
                       'ScoreSE': [98, 93, 44, 77, 69, 56, 31, 53, 
                                   78, 93, 56, 77, 33, 56, 27]})

df1ML =  pd.DataFrame({ 'StudentID': [1, 3, 5, 7, 9, 11, 13, 15, 17, 
                                      19, 21, 23, 25, 27, 29], 
                       'ScoreML' : [39, 49, 55, 77, 52, 86, 41, 77, 
                                    73, 51, 86, 82, 92, 23, 49]})
df2ML =  pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 
                       'ScoreML': [93, 44, 78, 97, 87, 89, 39, 43, 88, 78]})
#%%
data = np.arange(15).reshape((3,5))
indexers = ['Rainfall', 'Humidity', 'Wind']
dframe1 = pd.DataFrame(data, index=indexers, 
                       columns=['Bergen', 'Oslo', 'Trondheim', 
                                'Stavanger', 'Kristiansand'])
print(dframe1)
print(dframe1.unstack())
#%%
df = pd.read_csv("apple_stock_nan_example.csv")
#print(df.head())
print(df.isnull().sum().sum())

df.fillna(df.mean())

import matplotlib.pyplot as plt
from IPython import display
import matplotlib as mpl

mpl.rc('font', family = 'Malgun Gothic')
mpl.rc('axes', unicode_minus= False)

#df.unstack()

df.plot()
plt.title('구글 주가')
plt.xlabel("주가")
plt.ylabel("차트")
plt.show()
#%%
df.isnull().sum()
#??↓
df.replace(-9999.000000, np.nan, inplace = True)
df.replace(-999.000000, np.nan, inplace = True)
df.replace("nan", np.nan, inplace = True)

df['High'].fillna(df['High'].mean(),inplace = True)
df['Low'].fillna(df['Low'].mean(),inplace = True)
df['Open'].fillna(df['Open'].mean(),inplace = True)
df['Close'].fillna(df['Close'].mean(),inplace = True)
df['Volume'].fillna(df['Volume'].mean(),inplace = True)
df['Adj Close'].fillna(df['Adj Close'].mean(),inplace = True)

df.plot()
plt.title('구글 주가')
plt.xlabel("주가")
plt.ylabel("차트")
plt.show()
#%%
df = pd.read_csv("wine.csv")
#print(df.count())
#print(df.isnull().sum())
#print(df.describe()) #주로 활용되는 통계값 한번에 출력
#print(df.std()) # 표준편차 구하기
print(df['wine'].corr(df['death'])) # 특정컬럼 두가지의 상관계수 계산
print(df.corrwith(df['wine'])) #특정 컬럼과 나머지 컬럼간의 상관게수 구하기

plt.scatter(df['wine'],df['death'],alpha = 1)
plt.title('포도주 소비량과 심장병 사망')
plt.xlabel("포도주 소비량")
plt.ylabel("심장병에 기인한 사망")
plt.shcow()
#%% bin의 갯수를 정수로 지정

pd.cut(np.random.rand(40), 5, precision= 2)
randomNumbers = np.random.rand(2000)
category3 =pd.qcut(randomNumbers,4)
print(category3)
print(pd.value_counts(category3))
pd.qcut(randomNumbers, [0,0.3,0.5,0.7,1.0])
#%%
import pandas as pd

height =  [120, 122, 125, 127, 121, 123, 137, 131, 161, 145, 141, 132]

bins = [118, 125, 135, 160, 200]

category = pd.cut(height, bins)

category
#%% 문제 2번
df_score2 = pd.DataFrame({
 '반' : ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
 '번호' : [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
 '국어' : [90, 80, 90, 70, 100, 80, 90, 100, 70, 80],
 '영어' : [100, 90, 100, 80, 70, 90, 100, 70, 80, 90],
 '수학' : [80, 100, 80, 90, 80, 100, 70, 80, 90, 100]},
 columns = ["반","번호","국어","영어","수학"] 
 )
#print(df_score2)
#print(df_score2.describe())
#print(df_score2.isnull().sum())
#print(df_score2.take(np.permutation(df_score2[1,2])))
#df_score4 = df_score2.reindex(columns= ['번호','반','국어','영어','수학']).T
#print(df_score4)
# df_score2를 변형하여 1차 행 인덱스로 반을 2차행 인덱스로 번호를 가지는
#데이터 프레임 df_score4를 만든다.
df_score4 = df_score2.set_index(["반","번호"])
print(df_score4)
#각 학생의 평균을 나타내는 행을 오른쪽에 추가한다
df_score4["평균"] = df_score4.mean(axis = 1)
print(df_score4)

# df_score2을 변형하여 행 인덱스로 번호 를 1차 열 인덱스로
# 국어 영어 수학을 2차열 인덱스로 반 을 가지는 데이터프레임 을 만든다

df2 =df_score2.set_index(["반","번호"])
df_score4 = df2.unstack("반")
print(df_score4)

#데이터 프레임 df_score4에 각 반별 각 과목의 평균을 나타내는 행을 아래에 추가한다.
df_score4.loc["평균", :] = df_score4.mean()
print(df_score4)
#%% 3번문제 name컬럼을 가지고 두 데이터프레임 합치기
df1 = pd.DataFrame({
 'Name' : ['Morning', 'K3', 'K5', 'K8', 'K9'],
 'Segment' : ['Mini', 'Small', 'Mid', 'Sport', 'Large'],
 'Engine' : ['1.0L', '1.6', '2.0', '3.0', '5.0'],
 'Fuel' : ['14km', '18km', '16km', '12km', '10km'],
 'Price' : [1000, 2000, 3000, 4000, 5000]},
 columns = ["Name", "Segment", "Engine", "Fuel", "Price"])
df1

df2 = pd.DataFrame({
 '이름' : ['Morning', 'K3', 'K5', 'K8', 'K9'],
 '출시년도' : ['2000', '2005', '2005', '2016', '2010'],
 '연료' : ['휘발유', '경유', '경유', '휘발유', '휘발유'],
 '마력' : [50, 100, 150, 250, 300],
 '탑승인원' : [4, 5, 5, 4, 4]},
 columns = ["이름", "출시년도"," 연료"," 마력","탑승인원"])


df3 = pd.merge(df1, df2, left_on='Name', right_on='이름')
print(df3)

#%% 4번 문제
# 다음 두 데이터프레임을 합치기
profit1 = {'매출' : [1000, 1500, 3000, 4000, 5000, 6000],
 '비용' : [1500, 2000, 2500, 2700, 3000, 3200]}
columns1 = ['매출', '비용']
index1 = ['1월', '2월', '3월', '4월', '5월', '6월'] 
df1 = pd.DataFrame(profit1, index=index1, columns=columns1)
df1['이익'] = df1['매출'] - df1['비용']
df1

profit2 = {'매출' : [4500, 4000, 5000, 6000, 3000, 2000],
 '비용' : [2800, 2700, 3000, 3200, 2500, 2000]}
columns2 = ['매출', '비용']
index2 = ['7월', '8월', '9월', '10월', '11월', '12월']
df2 = pd.DataFrame(profit2, index=index2, columns=columns2)
df2['이익'] = df2['매출'] - df2['비용']
df2
# 답
df3 = pd.concat([df1, df2]) 
df3 = pd.concat([df3, pd.DataFrame({'매출': df3['매출'].sum(), '비용': df3['비용'].sum(), 
                                    '이익': df3['이익'].sum()}, index = ['총실적'])])
print(df3)

#%%
titanic = sns.load_dataset("titanic")

import seaborn as sns
titanic = sns.load_dataset("titanic")
titanic
bins = [1, 20, 60, 100]
labels = ["미성년", "성년", "노년"]
titanic_age = pd.cut(titanic["age"], bins, labels = labels)
titanic["age_class"] = titanic_age
titanic
df1 = titanic.groupby(['sex', 'age_class', 'class'])["survived"].mean()
df2 = df1.unstack("class")
print(df2)
titanic.pivot_table(['survived'], index=['sex', 'class'], aggfunc='mean')