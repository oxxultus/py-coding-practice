#= = = = = = = = = = = = = = = = = = = = = = = = = = 

                   # 연관 : 판다스 #

#= = = = = = = = = = = = = = = = = = = = = = = = = = 

                   # 딕셔너리의 생성 #

#= = = = = = = = = = = = = = = = = = = = = = = = = = 

# 딕셔너리를 생성합니다. 키와 값의 쌍으로 이루어집니다.
# 키 'name'에 해당하는 값은 '홍길동', 키 'birth'에 해당하는 값은 1990, 키 'addr'에 해당하는 값은 'KR'입니다.
dict_1 = {'name': '홍길동', 'birth': 1990, 'addr': 'KR'}


# 생성된 딕셔너리를 출력합니다.
print(dict_1)


#= = = = = = = = = = = = = = = = = = = = = = = = = = 

                # 딕셔너리의 요소 수정 #

#= = = = = = = = = = = = = = = = = = = = = = = = = = 

# 딕셔너리에서 'birth' 키에 해당하는 값을 출력합니다.
# 결과: 1990
print(dict_1['birth'])


# 딕셔너리에 새로운 키와 값을 추가합니다.
# 'weight' 키에는 60.5라는 값을 할당하고, 'family' 키에는 리스트 ['아빠', '엄마', '여동생']를 할당합니다.
dict_1['weight'] = 60.5
dict_1['family'] = ['아빠', '엄마', '여동생']


#= = = = = = = = = = = = = = = = = = = = = = = = = = 

                # 딕셔너리 라이브러리 함수 #

#= = = = = = = = = = = = = = = = = = = = = = = = = = 

# 딕셔너리 dict_1에 새로운 키와 값을 업데이트합니다.
# 'weight' 키에는 67.8이라는 값을 할당하고, 'hobby' 키에는 리스트 ['게임', '독서']를 할당합니다.
dict_1.update({'weight': 67.8, 'hobby': ['게임', '독서']})


# 딕셔너리 dict_1에서 'hobby' 키에 해당하는 값을 새로운 값으로 변경합니다.
# 'hobby' 키에는 리스트 ['축구', '등산']를 할당합니다.
dict_1['hobby'] = ['축구', '등산']


# 딕셔너리 dict_1에서 'weight' 키와 해당하는 값을 삭제합니다.
del dict_1['weight']


# 'hobby' 키에 해당하는 값을 반환하고 해당 키-값 쌍을 딕셔너리에서 제거합니다.
removed_value = dict_1.pop('hobby')


# 딕셔너리의 모든 요소를 제거합니다.
dict_1.clear()


# 예제 딕셔너리 생성
my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}


# keys() 함수를 사용하여 딕셔너리의 키(key)들을 가져옵니다.
keys = my_dict.keys()
print("Keys:", keys)  # 출력: Keys: dict_keys(['name', 'age', 'city'])


# values() 함수를 사용하여 딕셔너리의 값(value)들을 가져옵니다.
values = my_dict.values()
print("Values:", values)  # 출력: Values: dict_values(['John', 30, 'New York'])


# items() 함수를 사용하여 딕셔너리의 (키, 값) 쌍을 가져옵니다.
items = my_dict.items()
print("Items:", items)  # 출력: Items: dict_items([('name', 'John'), ('age', 30), ('city', 'New York')])


# len() 함수를 사용하여 딕셔너리의 길이(키-값 쌍의 개수)를 가져옵니다.
length = len(my_dict)
print("Length:", length)  # 출력: Length: 3


# type() 함수를 사용하여 딕셔너리의 타입을 확인합니다.
dict_type = type(my_dict)
print("Type:", dict_type)  # 출력: Type: <class 'dict'>
