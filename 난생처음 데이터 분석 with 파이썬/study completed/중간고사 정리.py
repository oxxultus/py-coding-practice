= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# 문자열 포맷팅
print("{} * {} = {}".format(a, b, c))
print("{0} * {1} = {2}".format(a, b, c))
print(f"{a} * {b} = {c}")

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# 리스트 라이브러리 함수
append(), insert(), remove(), sort(), reverse(), clear()
max(), sum(), min(), len()

[10, 20, 200, 40, 50, 60, 70, 80] + [5, 9, 12, 8, 30]
= [10, 20, 200, 40, 50, 60, 70, 80, 5, 9, 12, 8, 30]
listC = listA + listB

#행: A ~ B-1 Index
#열: C ~ C-1 Index
listC[A:B, C:D]

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# 딕셔너리 라이브러리 함수
update(), pop(), keys(), values(), items(), clear() 
len(), del

dict_1 = {'name': '홍길동', 'birth': 1990, 'addr': 'KR'}
dict_1['weight'] = 60.5
dict_1['family'] = ['아빠', '엄마', '여동생']

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# NumPy 라이브러리 함수
import numpy as np
from numpy import*
import numpy

# ndarray 생성
npy = np.arange(12).reshape(4, 3)
np.array([LIST], dtype=np.int8)
np.zeros((4, 3))  # 요소값이 모두 0
np.ones((4, 3))   # 요소값이 모두 1
np.full((4, 3), 2)  # 요소값이 모두 2
np.eye(3)          # 단위행렬

# 배열 속성 및 연산
npy.size           # 요소개수
npy.dtype          # 데이터타입
npy.shape          # 모양
npy.ndim           # 차원 수
npy.flat           # 이터레이터
npy.nbytes         # 바이트 크기
npy.T              # 전치행렬

# 배열 변환
npy.astype(np.float64)
npy.flatten()
npy.reshape((2, 6))

# 배열 요소 합과 평균
npy.sum()          # 전체요소의 합
npy.sum(axis=0 or 1)  # 특정 축(axis)을 기준으로 합을 계산
npy.mean()         # 전체요소의 평균
npy.mean(axis=0 or 1)  # 특정 축(axis)을 기준으로 평균을 계산

= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Pandas 시리즈 라이브러리 함수
pd.Series([리스트],index=[리스트]) 
pd.Series({딕셔너리}) #key => index, value=>value

시리즈.unique() 고유값
시리즈.count_values(sort=True or False) 고유값의 개수 , 오름차순 정렬할지 말지
시리즈.rank(ascending=True or False,method='max' or 'min' or 'first')  
시리즈[시리즈.isin(['value'])] # 해당값만 출력 boolean

시리즈 = pd.Series([1, 2, 3, 4])
시리즈 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
시리즈 = pd.Series({'city': '서울', 'year': 2018})
시리즈7 = pd.Series({'city': ['서울', '부산', '대전', '대구', '광주'],
         'year': [2017, 2017, 2018, 2018, 2018],
         'temp': [18, 20, 19, 21, 20]})
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Pandas 데이터프레임 라이브러리 함수
pd.DataFrame({딕셔너리}) #키값이 열제목, 값이 해당열의 value로 들어간다
pd.DataFrame([리스트])  
#([리스트],[인덱스],[열제목]) => 1열으로 생성
#([[리스트],[리스트2],[리스트3]])=> ('리스트의 개수'행 생성) * ('리스트의 요소 개수' 열으로 생성)
pd.DataFrame(np.arange(12).reshape(4,3),columns=[리스트],index=[리스트])

시리즈 = 데이터프레임['column'] # 하나의 열은 시리즈
데이터프레임 = 데이터프레임[['column','column']] # 두개 이상의 열은 데이터프레임

데이터프레임.index = [리스트]
데이터프레임.columns = [리스트]
데이터프레임.loc[index_name]
데이터프레임.iloc[index_number]
데이터프레임.set_index('column',inplace=True or False)
데이터프레임.fillna(value) # 결측값 value값으로 바꾸기
데이터프레임['key'] = value #value의 값을 가진 key라는 열을 추가 하거나 있으면 변경한다.

#기준으로 연산
데이터프레임.drop(index or key,axis=0 or 1,inplace=True or False)
데이터프레임.dropna(axis=0 or 1,inplace=True or False)

#방향으로 연산
데이터프레임.sort_index(axis 0 or 1, ascending=True or False,inplace=True or False)
데이터프레임.sort_values(by=['b', 'c'],ascending=True or False,inplace=True or False)
데이터프레임.rank(axis=0 or 1,ascending=True or False,method='first','max','min')
데이터프레임.sum(axis=0 or 1,skipna=True or False)
데이터프레임.mean(axis=0 or 1,skipna=True or False)
데이터프레임.idxmax(axis=0 or 1)
데이터프레임.idxmin(axis=0 or 1)

# 조건에 따라 새로운 열 'high1'을 DataFrame '데이터프레임'에 추가합니다.
# car의 값보다 크거나 같으면 True를 작으면 False를 저장한다.
데이터프레임['high1'] = 데이터프레임.car >= 30

# 시리즈를 CSV 파일로 저장합니다.
# './file2.csv'는 저장할 파일의 경로와 이름을 나타냅니다.
# header=True는 열 이름을 포함하여 저장하고, index=False는 인덱스를 저장하지 않습니다.
# encoding='utf-8'은 UTF-8 인코딩을 사용하여 파일을 저장합니다.
데이터프레임.to_csv('./file2.csv', header=True, index=False, encoding='utf-8')

# CSV 파일을 읽어서 데이터프레임으로 변환합니다.
# './file2.csv'에서 파일을 읽어오며, sep=','는 구분자로 쉼표를 사용합니다.
데이터프레임 = pd.read_csv('./file2.csv', sep=',')
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =











