
# '-' 부호를 제대로 표시할 수 있도록 설정
plt.rcParams["axes.unicode_minus"] = False

# pandas를 사용하여 데이터 프레임 생성
df = pd.DataFrame(np.random.rand(6,4), columns=pd.Index(['A','B','C','D'],name='genus'), index=['one','two','three','four','five','six'])

# numpy 및 matplotlib을 사용하여 여러 그래프 그리기
np.random.seed(10)
data = np.random.randn(2,100)
fig, axs = plt.subplots(2,2, figsize=(5,5))

# 각 축에 그래프를 그리기
axs[0,0].hist(data[0])
axs[1,0].scatter(data[0], data[1])
axs[0,1].plot(data[0], data[1])
axs[1,1].hist2d(data[0], data[1])


import matplotlib.pyplot as plt
plt.figure(): 새로운 피규어를 생성하는 함수.
# 예시: plt.figure(figsize=(8, 6))
# figsize: 피규어의 크기를 (너비, 높이)로 지정합니다.

plt.title(): 그래프의 제목을 설정하는 함수.
# 예시: plt.title('Graph Title')
# Graph Title: 그래프의 제목 문자열을 지정합니다.

plt.xlim(): x축의 범위를 설정하는 함수.
# 예시: plt.xlim(-10, 10)
# -10: x축의 최소 값.
# 10: x축의 최대 값.
plt.ylim(): y축의 범위를 설정하는 함수입니다.
# 예시: plt.ylim(-5, 5)
# -5: y축의 최소 값.
# 5: y축의 최대 값.


plt.xticks(): x축의 눈금을 설정하는 함수입니다.
# 예시: plt.xticks(np.arange(-10, 11, 1), labels=['-10', '0', '10'])
# np.arange(-10, 11, 1): x축 눈금의 위치를 배열로 지정합니다.
# labels: x축 눈금의 라벨을 배열로 지정합니다.
plt.yticks(): y축의 눈금을 설정하는 함수입니다.
# 예시: plt.yticks(np.arange(-5, 6, 1), labels=['-5', '0', '5'])
# np.arange(-5, 6, 1): y축 눈금의 위치를 배열로 지정합니다.
# labels: y축 눈금의 라벨을 배열로 지정합니다.


plt.legend(): 그래프의 범례를 표시하는 함수입니다.
# 예시: plt.legend(loc='upper right')
# loc: 범례의 위치를 지정합니다. 예를 들어, 'upper right'는 우측 상단을 의미합니다.

plt.subplot(): 여러 개의 그래프를 한 피규어에 그리기 위한 함수입니다.
# 예시: plt.subplot(2, 2, 1)
# 2, 2, 1: 피규어를 2x2 격자형으로 나누고 첫 번째 격자에 플롯을 그립니다.


plt.plot(): 선 그래프를 그리는 함수입니다.
#예시: plt.plot([1, 2, 3], [4, 5, 6], 'r--', label='Line')
# [1, 2, 3]: x 좌표 값.
# [4, 5, 6]: y 좌표 값.
# 'r--': 선 스타일. 빨간색(r)과 점선(--) 스타일을 지정합니다.
# label: 범례에 사용할 이름을 지정합니다.
# kind='bar': 세로 막대 그레프
# kind='barh': 가로 막대 그레프
# stacked=True: 누적 막대 그래프
# c="": 선 색
# lw="": 선 굵기
# ls="": 선 스타일 
# maker="": 마커 종류
# ms="": 마커 크기
# mec="": 마커 선 색
# mew="": 마커 선 굻기
# mfc="": 마커 내부 색


plt.hist(): 히스토그램을 그리는 함수입니다.
# 예시: plt.hist(np.random.randn(100), bins=10, color='blue')
# np.random.randn(100): 정규 분포를 따르는 난수 100개.
# bins=10: 히스토그램의 막대 수.
# color='blue': 막대 색상.
# 막대 사이 간격 있음 rwidth=0.5, 없음 width=1


plt.pie(): 파이 차트를 그리는 함수입니다.
# 예시: plt.pie([30, 20, 50], labels=['A', 'B', 'C'], autopct='%.1f%%')
# [30, 20, 50]: 각 파이 조각의 크기.
# labels: 각 조각의 라벨.
# autopct='%.1f%%': 각 조각의 비율을 표시하는 포맷. %.1f%%는 소수점 첫째 자리까지 표시합니다.

plt.boxplot(): 상자 그림(Boxplot)을 그리는 함수입니다.
# 예시: plt.boxplot(np.random.randn(100)): 정규 분포(평균 0, 표준 편차 1)를 따르는 난수 100개를 생성합니다.
# np.random.randn(100): 상자 그림을 그릴 데이터를 지정합니다.
#이상치 : 희귀 값, 무시 X
#최대
# 75% : 3사분기 : 편차값
# 50% : 2사분기
# 25% : 1사분기 : 편차값
#최소
#이싱치 : 상위3퍼 하위4퍼

plt.scatter(): 산점도를 그리는 함수입니다.
# 예시: plt.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30), c='g')
# np.arange(30): x 좌표 값.
# np.arange(30) + 3 * np.random.randn(30): y 좌표 값.
# c='g': 점의 색상을 녹색(green)으로 지정합니다.

plt.tight_layout(): 그래프의 레이아웃을 자동으로 조정하는 함수.

plt.show(): 그래프를 화면에 표시하는 함수.

plt.savefig(): 그래프를 파일로 저장하는 함수입니다.
# 예시: plt.savefig("graph.png", dpi=300)
# "graph.png": 저장할 파일의 경로와 이름을 지정합니다.
# dpi=300: 이미지의 해상도를 설정합니다.


np.random.seed(): 난수 생성의 시드를 설정하는 함수.
# seed (optional): 난수 생성의 시드 값.

np.random.randn(): 표준 정규 분포를 따르는 난수 배열을 생성하는 함수.
# (3,4) 3행4열 과 같은 형식
# d0, d1, ..., dn (optional): 난수 배열의 차원 크기를 지정하는 매개변수. 
# 여러 개의 차원 크기를 지정하면 해당 크기의 다차원 배열을 생성합니다.

np.random.randint(): 지정된 범위에서 정수 난수를 생성하는 함수.
# low: 난수를 생성할 최소값을 지정합니다.
# high (optional): 난수를 생성할 최대값을 지정합니다. 이 값은 low 이상이어야 합니다.
# size (4,3): 생성할 난수 배열의 크기를 지정합니다.

np.random.rand(): 0에서 1 사이의 난수를 생성하는 함수.
# (3,4) 3행4열 과 같은 형식
# d0, d1, ..., dn (optional): 난수 배열의 차원 크기를 지정하는 매개변수. 
# 여러 개의 차원 크기를 지정하면 해당 크기의 다차원 배열을 생성합니다.

np.linspace(): 지정된 범위에서 일정 간격의 숫자 배열을 생성하는 함수.
# start: 배열의 시작 값.
# stop: 배열의 종료 값.
# num (optional): 배열에 포함될 숫자의 개수. 기본값은 50입니다.
# endpoint (optional): stop 값을 배열에 포함할지 여부를 지정합니다. 
# 기본값은 True로, stop 값이 배열의 마지막 숫자로 포함됩니다.

np.cos(): 코사인 함수를 계산하는 함수.
# x = np.linspace(-np.pi, np.pi, num=100)
# cos_values = np.cos(x)

np.arange(): 지정된 범위에서 일정 간격의 숫자 배열을 생성하는 함수.
# start (optional): 배열의 시작 값. 지정하지 않으면 기본값은 0입니다.
# stop: 배열의 종료 값. 이 값은 배열에 포함되지 않습니다.
# step (optional): 배열의 간격을 지정합니다. 기본값은 1입니다.
