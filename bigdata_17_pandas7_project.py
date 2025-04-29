import numpy as np
import pandas as pd
nycflights = pd.read_csv('../data/nycflights.csv')

from nycflights13 import flights,planes

flights.info()
planes.isna().sum()
flights.isna().sum()


#제출하는건 보고서 하나
#merge 사용해서 flights와 planes 병합한 데이터로
#각 데이터 변수 최소 하나씩 선택 후 분석할 것.
#날짜&시간 전처리 코드 들어갈 것.
#문자 전처리 코드 들어갈 것.
#시각화 종류 최소 3개(배우지 않은 것도 할 수 있으면 넣어도 됨.)


# speed열 삭제
#del planes['speed']
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
########################################################################
#항공사별 dep_delay 열의 평균/중앙값 파악
df.groupby('carrier')['dep_delay'].mean().sort_values(ascending=False)
df.groupby('carrier')['dep_delay'].median().sort_values(ascending=False)

dep_delay_df = df.loc[df['dep_delay'] > 0]

dep_delay_count = dep_delay_df['carrier'].value_counts().reset_index()
df_count = df['carrier'].value_counts().reset_index()
delay_count_merge = pd.merge(dep_delay_count,df_count,on='carrier')
delay_ratio_series = pd.DataFrame((dep_delay_count.iloc[:,1] / df_count.iloc[:,1]).sort_index(ascending=True))

ratio_total = pd.concat([delay_count_merge,delay_ratio_series],axis=1)

ratio_total = ratio_total.rename(columns = {'count_x':'dep_delay_count','count_y': 'total_count','count':'ratio'})
ratio_total.sort_values('ratio',ascending=False)

#가중치 어째 둘거?????
# ratio의 단위, mean의 단위가 다름 -> 어떻게 해결해서 가중치 맞출거?

#평균 지연 시간(분)
#(dep_delay_df.groupby('carrier')['dep_delay'].sum().reset_index().iloc[:,1]) / (dep_delay_count.iloc[:,1])
dep_delay_df.groupby('carrier')['dep_delay'].mean()
weight_df = pd.merge(ratio_total,dep_delay_df.groupby('carrier')['dep_delay'].mean(),on='carrier')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
weight_df['minmac_ratio'] = scaler.fit_transform(weight_df[['ratio']])
weight_df['minmax_dep_delay'] = scaler.fit_transform(weight_df[['dep_delay']])

#주말/주중/시간대별 항공사 지연 파악










df['date']=pd.to_datetime(df['year'].astype('str') + df['month'].astype('str') +df['day'].astype('str'),format='%Y%m%d')
#출발 지연이 된 데이터 프레임만 볼거임
dep_delay_df = df.loc[df['dep_delay'] > 0]












dep_delay_df.groupby('carrier',as_index=False)['arr_delay'].mean().sort_values('arr_delay',ascending=False)
# oo 항공사의 지연이 가장 높음음

dep_delay_df.groupby('carrier',as_index=False)['arr_delay'].median().sort_values('arr_delay',ascending=False)
# 중앙값으로 비교했을 때도 oo가 가장 높음.


dep_delay_df.pivot_table(
    index='carrier',
    values='arr_delay',
    aggfunc='max'
)
# 항공사별, 도착 지연이 가장 큰 값을 찍어봄. HA가 1272분이 나옴



#항공기 종류별 평균 출발 지연
dep_delay_df.groupby('type')['dep_delay'].mean()

#항공기 종류별 평균 비행거리
df.groupby('type')['distance'].mean()



#a = pd.pivot_table(dep_delay_df,
#               values='dep_delay',
#               index=['carrier','sched_dep_hour'],
#               #columns='sched_dep_hour',
#               aggfunc='mean').reset_index()
#
#a.loc[a.groupby('carrier')['dep_delay'].idxmax()].sort_values('dep_delay',ascending=False)











df = pd.merge(flights, planes,on='tailnum', how='inner')
df.info()
df.isna().sum()
df.head()
pd.set_option('display.max_columns', None) 
df['tailnum'].nunique()

df['time_hour'] = pd.to_datetime(df['time_hour'])

# 문제가 있던 항공기들의 결함 및 날씨 영향 파악
# 항공사들의 항공기 선호도
# 문제가 있던 항공기들의 연식, 엔진 종류 등 파악
# 특정 항공기가 지연이 많이 되는가?
# 항공사별로 어느 시간대에 지연이 많이 되는지 파악 
# 항공사별 선호하는 기종 / 제조사 파악
# 지연이 많이 되는 비행기 기종 파악

# 항공사별 지연 원인 분석
# 출발지연 (날씨영향 및 결함이 있기에 출발 지연이 있었을 것이다.)
# 그리고 출발에 지연하면 도착에도 지연이니까까

# 주말, 주중, 시간대별 항공사의 지연 현황 파악

# 지연이 많이 되는 항공사가 많이 이용하는 기종 및 제조사가 무엇인지.

# 지연이 많이 되는 비행기 기종 / 엔진 종류 파악

# 지연이 많이 되는 비행기 기종은 연식이 오래 됐을 것이다
# / 엔진 양이 적을 것이다 -> 오랜 거리 비행을 하지 않을 것이다. 

# 비행기 모델(최신식, 옛날식) 별로도 보면 좋을거 같다. 
df.info()

df.isna().sum()
planes.isna().sum()
df['tailnum']

df = df.drop(columns = ['year_x','speed'])
df=df.columns.rename(columns={'year_y:'})

# fligths의 nan 변수 확인 -> 둘이 같은 행에 nan위치치
flights[(flights['dep_time'].isna()) & (flights['dep_delay'].isna())]