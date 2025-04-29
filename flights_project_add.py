from matplotlib import font_manager, rc

### 운영체제 확인 라이브러리
import platform

### 시각화 시 마이너스(-, 음수) 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

### OS별 한글처리
# - 윈도우 운영체게
if platform.system() == "Windows" :
    # path = "c:/Windows/Fonts/malgun.ttf"
    # font_name = font_manager.FontProperties(fname=path).get_name()
    # rc("font", family = font_name)
    
    ### 또는 아래처럼 한줄로도 가능 (아래 한글처리를 주로 사용합니다.)
    plt.rc("font", family = "Malgun Gothic")

# - Mac 운영체제
elif platform.system() == "Darwin" :
    rc("font", family = "AppleGothic")
    
else :
    print("넌 누구?")





import numpy as np
import pandas as pd
nycflights = pd.read_csv('../data/nycflights.csv')
import matplotlib.pyplot as plt
import seaborn as sns
from nycflights13 import flights,planes
del planes['speed']
flights.isna().sum()
planes.isna().sum()

#flights 결측치 싹다 제거 
f = flights.dropna()
f.isna().sum()
#planes 결측치 제거 
p = planes.dropna()
p.isna().sum()

#merge 및 중복열 삭제, 변수명 변경
df =pd.merge(f,p,on='tailnum',how='inner')
df.isna().sum()
df = df.drop(columns=['sched_dep_time','time_hour'])
df = df.rename(columns = {'year_x':'year','year_y':'man_year','hour':'sched_dep_hour','minute':'sched_dep_minute'})



########################################################################################################
# 데이터 현황 분석
########################################################################################################
df.info()
df.describe()
df["dep_delay"].describe()


# 출발 지연 시간 분포
plt.figure(figsize=(12, 6))
sns.histplot(df["dep_delay"], bins=50, kde=True, color="blue")
plt.title("dep_delay_distribution")
plt.xlabel("dep_delay_minute")
plt.ylabel("항공편 수")
plt.xlim(-10, 300)  # 극단적인 이상치는 제외하고 가시성 높이기
plt.grid()
plt.show()




########################################################################################################
# 일별 지연시간 분석
########################################################################################################


df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['day_of_week'] = df['date'].dt.dayofweek

# 요일별 평균 출발 지연시간 계산
weekday_delays = df.groupby('day_of_week')['dep_delay'].mean().reset_index()


# 요일 이름 매핑
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_delays['day_of_week'] = weekday_delays['day_of_week'].map(lambda x: weekday_labels[x])

# 시각화
plt.figure(figsize=(10, 5))
plt.bar(weekday_delays['day_of_week'], weekday_delays['dep_delay'], color='skyblue', alpha=0.8)
plt.xlabel('Day of the Week')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Day of the Week')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


########################################################################################################
# 월별 지연 트렌드
########################################################################################################

mean_month=df.groupby("month")["dep_delay"].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(mean_month['month'], mean_month['dep_delay'], color='skyblue', alpha=0.8)
plt.xlabel('month')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by month')
plt.xticks(ticks=range(1, 13), labels=range(1, 13))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


########################################################################################################
#항공사별 dep_delay 열의 평균/중앙값 파악
########################################################################################################


carrier_mean = df.groupby('carrier')['dep_delay'].mean().reset_index().sort_values('dep_delay',ascending=False)
df.groupby('carrier')['dep_delay'].median().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
plt.bar(carrier_mean['carrier'], carrier_mean['dep_delay'], color='skyblue', alpha=0.8)
plt.xlabel('carrier')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by carrier')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# 평균 지연 시간 계산
carrier_mean = df.groupby('carrier')['dep_delay'].mean().reset_index().sort_values('dep_delay', ascending=False)

# 강조할 항공사
highlight = ['B6', 'VX']

# 색상 설정 (강조할 항공사만 파란색, 나머지는 회색)
colors = ['blue' if carrier in highlight else 'lightgray' for carrier in carrier_mean['carrier']]

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.bar(carrier_mean['carrier'], carrier_mean['dep_delay'], color=colors, alpha=0.8)
plt.xlabel('Carrier')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Carrier')

# 강조한 막대에 라벨 추가
for i, carrier in enumerate(carrier_mean['carrier']):
    if carrier in highlight:
        plt.text(i, carrier_mean['dep_delay'].iloc[i] + 0.5, f"{carrier_mean['dep_delay'].iloc[i]:.1f}", 
                 ha='center', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


############################################################################################
delay_over_180 = df.loc[df['dep_delay'] >= 180]
df_count = df['carrier'].value_counts().reset_index()


# 180 이상
delay_over_180_count = delay_over_180['carrier'].value_counts().reset_index()
delay_over_180_count_merge = pd.merge(delay_over_180_count,df_count,on='carrier',how='outer')
delay_over_180_ratio = pd.DataFrame((delay_over_180_count_merge.iloc[:,1] / delay_over_180_count_merge.iloc[:,2]).sort_index(ascending=True))
ratio_over_180 = pd.concat([delay_over_180_count_merge,delay_over_180_ratio],axis=1)
ratio_over_180 = ratio_over_180.rename(columns = {'count_x':'dep_delay_count','count_y': 'total_count',0:'ratio'})
ratio_over_180.sort_values('ratio',ascending=False)

over_180_df = pd.merge(ratio_over_180,delay_over_180.groupby('carrier')['dep_delay'].mean(),on='carrier',how='outer')
over_180_df = over_180_df.fillna(0)

#180 이하
delay_under_180 = df.loc[(df['dep_delay'] > 0) & (df['dep_delay'] <180)]
delay_under_180_count = delay_under_180['carrier'].value_counts().reset_index()
delay_under_180_count_merge = pd.merge(delay_under_180_count,df_count,on='carrier',how='outer')
delay_under_180_ratio = pd.DataFrame((delay_under_180_count_merge.iloc[:,1] / delay_under_180_count_merge.iloc[:,2]).sort_index(ascending=True))
ratio_under_180 = pd.concat([delay_under_180_count_merge,delay_under_180_ratio],axis=1)
ratio_under_180 = ratio_under_180.rename(columns = {'count_x':'dep_delay_count','count_y': 'total_count',0:'ratio'})
ratio_under_180.sort_values('ratio',ascending=False)

under_180_df = pd.merge(ratio_under_180,delay_under_180.groupby('carrier')['dep_delay'].mean(),on='carrier',how='outer')

merged_df = pd.concat([over_180_df,under_180_df], axis=0).reset_index(drop=True)
merged_df['minmax_ratio'] = (merged_df['ratio'] - merged_df['ratio'].min()) / (merged_df['ratio'].max() - merged_df['ratio'].min())
merged_df['minmax_dep_delay'] = (merged_df['dep_delay'] - merged_df['dep_delay'].min()) / (merged_df['dep_delay'].max() - merged_df['dep_delay'].min())

over_180_df['minmax_ratio'] = (over_180_df['ratio'])/0.524012   #전체 ratio 중에서 가장 큰 값/작은값은 0
over_180_df['minmax_dep_delay'] = (over_180_df['dep_delay'])/743.5  #전체 dep_delay중에서 가장 큰 값/작은값은 0

under_180_df['minmax_ratio'] = (under_180_df['ratio'])/0.524012
under_180_df['minmax_dep_delay'] = (under_180_df['dep_delay'])/743.5

weight_total_score = ((over_180_df['minmax_ratio'] * 0.3 + over_180_df['minmax_dep_delay'] * 0.7) + 
 (under_180_df['minmax_ratio'] * 0.5 + under_180_df['minmax_dep_delay'] * 0.5))
weight_total_score_df = pd.concat([over_180_df['carrier'], weight_total_score], axis=1).sort_values(0,ascending=False)
weight_total_score_df[0]

# HA, FL, WN, F9 의 순으로 높음

df.loc[(df['carrier'] == 'HA'),'distance'].mean()
df.loc[(df['carrier'] == 'FL'),'distance'].mean()
df.loc[(df['carrier'] == 'WN'),'distance'].mean()
df.loc[(df['carrier'] == 'F9'),'distance'].mean()

df['distance'].mean()

df['engine'].value_counts()
##!!!!!!!!!!!!!!!!표만들어!!!!!!!!!!!!!!!!!!!!!!##
df.loc[(df['carrier'] == 'HA'),'model'].value_counts()
df.loc[(df['carrier'] == 'HA'),'engines'].value_counts()
df.loc[(df['carrier'] == 'HA'),'engine'].value_counts()


df.loc[(df['carrier'] == 'FL'),'model'].value_counts()
df.loc[(df['carrier'] == 'FL'),'engines'].value_counts()
df.loc[(df['carrier'] == 'FL'),'engine'].value_counts()

df.loc[(df['carrier'] == 'WN'),'model'].value_counts()
df.loc[(df['carrier'] == 'WN'),'engines'].value_counts()
df.loc[(df['carrier'] == 'WN'),'engine'].value_counts()

df.loc[(df['carrier'] == 'F9'),'model'].value_counts()
df.loc[(df['carrier'] == 'F9'),'engines'].value_counts()
df.loc[(df['carrier'] == 'F9'),'engine'].value_counts()


#시각화 
df.loc[(df['carrier'] == 'HA') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'FL') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'WN') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'F9') & (df['dep_delay']>0)]['month'].value_counts()



df.loc[(df['carrier'] == 'HA') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'FL') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'WN') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'F9') & (df['dep_delay']>0)]['day_of_week'].value_counts()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance', y='dep_delay', alpha=0.5)
plt.title("비행 거리 vs. 출발 지연 시간")
plt.xlabel("비행 거리 (마일)")
plt.ylabel("출발 지연 시간 (분)")
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
top_4_carriers = ['HA', 'FL', 'WN', 'F9']
df_top_4 = df[df['carrier'].isin(top_4_carriers)]  # 상위 4개 항공사 데이터만 필터링

sns.scatterplot(data=df_top_4, x='distance', y='dep_delay', hue='carrier', alpha=0.5, palette='tab10')

plt.title("비행 거리 vs. 출발 지연 시간 (항공사별 비교)")
plt.xlabel("비행 거리 (마일)")
plt.ylabel("출발 지연 시간 (분)")
plt.legend(title="항공사")
plt.grid()
plt.show()



#발표자+
import numpy as np
np.random.seed(1011)
np.random.randint(1,5)


df.groupby('carrier')['dep_delay'].mean()

