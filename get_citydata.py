import pandas as pd
import requests
from xml.etree import ElementTree as ET
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import traceback

# In[2]:

places = ['강남 MICE 관광특구',
 '동대문 관광특구',
 '명동 관광특구',
 '이태원 관광특구',
 '잠실 관광특구',
 '종로·청계 관광특구',
 '홍대 관광특구',
 '경복궁·서촌마을',
 '광화문·덕수궁',
 '창덕궁·종묘',
 '가산디지털단지역',
 '강남역',
 '건대입구역',
 '고속터미널역',
 '교대역',
 '구로디지털단지역',
 '서울역',
 '선릉역',
 '신도림역',
 '신림역',
 '신촌·이대역',
 '역삼역',
 '연신내역',
 '용산역',
 '왕십리역',
 'DMC(디지털미디어시티)',
 '창동 신경제 중심지',
 '노량진',
 '낙산공원·이화마을',
 '북촌한옥마을',
 '가로수길',
 '성수카페거리',
 '수유리 먹자골목',
 '쌍문동 맛집거리',
 '압구정로데오거리',
 '여의도',
 '영등포 타임스퀘어',
 '인사동·익선동',
 '국립중앙박물관·용산가족공원',
 '남산공원',
 '뚝섬한강공원',
 '망원한강공원',
 '반포한강공원',
 '북서울꿈의숲',
 '서울대공원',
 '서울숲공원',
 '월드컵공원',
 '이촌한강공원',
 '잠실종합운동장',
 '잠실한강공원']


# In[3]:


# csv_file = pd.read_csv("C:\\Users\\nws71\\seoul_city_data.csv")
# sheet_name = 'city_pop'


# In[55]:


def city_pop(update): # 인구 혼잡도 dataframe of places
  for index, place in enumerate(places):
    url = f'http://openapi.seoul.go.kr:8088/796e7558466e777334394148697a62/xml/citydata/1/5/{place}'
    print(f'{index}번째 for문입니다. 값은 {place}입니다.')
    response = requests.get(url).text
    tree = ET.fromstring(response)
    ET.indent(tree, space="  ")
    update.append(tree[2][1][0][17].text)
    if len(update) == 1:
      return city_pop(update)
    else:
      if update[-1] == update[-2]:
        listOf = []
        listOf.append(tree[2][0].text)
        listOf.append(tree[2][1][0][0].text)
        listOf.append(tree[2][1][0][2].text)
        listOf.append(tree[2][1][0][3].text)
        listOf.append(tree[2][1][0][4].text)
        listOf.append(tree[2][1][0][5].text)
        listOf.append(tree[2][1][0][6].text)
        listOf.append(tree[2][1][0][7].text)
        listOf.append(tree[2][1][0][8].text)
        listOf.append(tree[2][1][0][9].text)
        listOf.append(tree[2][1][0][10].text)
        listOf.append(tree[2][1][0][11].text)
        listOf.append(tree[2][1][0][12].text)
        listOf.append(tree[2][1][0][13].text)
        listOf.append(tree[2][1][0][14].text)
        listOf.append(tree[2][1][0][15].text)
        listOf.append(tree[2][1][0][17].text)
        listOf.append(tree[2][8][0][0].text)
        listOf.append(tree[2][8][0][1].text)
        listOf.append(tree[2][8][0][2].text)
        listOf.append(tree[2][8][0][5].text)
        listOf.append(tree[2][8][0][7].text)
        listOf.append(tree[2][8][0][8].text)
        listOf.append(tree[2][8][0][9].text)
        listOf.append(tree[2][8][0][13].text)
        listOf.append(tree[2][8][0][16].text)
        listOf.append(tree[2][8][0][18].text)
        listOf.append(tree[2][8][0][20].text)
        listOf.append(tree[2][8][0][24][-1][0].text)
        listOf.append(tree[2][8][0][24][-1][5].text)
        listOf.append(tree[2][2][0][3].text)
        listOf.append(tree[2][2][0][2].text)

        columns = []
        columns.append(tree[2][0].tag)
        columns.append(tree[2][1][0][0].tag)
        columns.append(tree[2][1][0][2].tag)
        columns.append(tree[2][1][0][3].tag)
        columns.append(tree[2][1][0][4].tag)
        columns.append(tree[2][1][0][5].tag)
        columns.append(tree[2][1][0][6].tag)
        columns.append(tree[2][1][0][7].tag)
        columns.append(tree[2][1][0][8].tag)
        columns.append(tree[2][1][0][9].tag)
        columns.append(tree[2][1][0][10].tag)
        columns.append(tree[2][1][0][11].tag)
        columns.append(tree[2][1][0][12].tag)
        columns.append(tree[2][1][0][13].tag)
        columns.append(tree[2][1][0][14].tag)
        columns.append(tree[2][1][0][15].tag)
        columns.append(tree[2][1][0][17].tag)
        columns.append(tree[2][8][0][0].tag)
        columns.append(tree[2][8][0][1].tag)
        columns.append(tree[2][8][0][2].tag)
        columns.append(tree[2][8][0][5].tag)
        columns.append(tree[2][8][0][7].tag)
        columns.append(tree[2][8][0][8].tag)
        columns.append(tree[2][8][0][9].tag)
        columns.append(tree[2][8][0][13].tag)
        columns.append(tree[2][8][0][16].tag)
        columns.append(tree[2][8][0][18].tag)
        columns.append(tree[2][8][0][20].tag)
        columns.append(tree[2][8][0][24][-1][0].tag)
        columns.append(tree[2][8][0][24][-1][5].tag)
        columns.append(tree[2][2][0][3].tag)
        columns.append(tree[2][2][0][2].tag)

        if index == 0:
          city = pd.DataFrame([tuple(listOf)], columns=columns) # listOf를 transpose한 후에 tuple로 바꿔주기 때문에 원하는대로 18개의 element
        else:
          city.loc[len(city)] = listOf
      else:
        return city_pop(update)
  return city, update

# In[59]:
def runtime_taken(start, end):
    time_diff = end - start
    print("saveD", end, "| 소요시간: ", time_diff)


# In[ ]:
update = []
z = 1
while True:
    try:
      if z == 1:
        start = datetime.now()
        save, update = city_pop(update)
        save.to_csv("seoul_citydata_2 (17).csv", mode='a', header=False, index=False) # csv파일에 튜플들 누적 저장
        end = datetime.now()
        runtime_taken(start, end) # 소요시간 함수
        z += 1
      else:
        url = 'http://openapi.seoul.go.kr:8088/796e7558466e777334394148697a62/xml/citydata/1/5/강남역'
        response = requests.get(url).text
        tree = ET.fromstring(response)
        ET.indent(tree, space="  ")
        if update[-1] == tree[2][1][0][17].text:
            print("서버 갱신 안됨")
            time.sleep(30) # 갱신 1초 전에 실행 -> 갱신된지 29초 후에 실행 -> 3,4분 소요 => 총 4분 30초(최대 5분)
                            #     continue #다시 처음부터 while문 시작
        else:
            update.append(tree[2][1][0][17].text) #서버갱신시간 time 리스트에 추가
            start = datetime.now()
            print("서버갱신: " , update[-1], "| 현재 시각: ",start) #서버갱신 알림출력
            save, update = city_pop(update) #population 데이터프레임 추출 함수 -> population
            save.to_csv("seoul_citydata_2 (17).csv", mode='a', header=False, index=False) # csv파일에 튜플들 누적 저장
            end = datetime.now()
            runtime_taken(start, end) # 소요시간 함수
            update = [update[-1]]
    #             time.sleep(60) # 1분 대기 후

    except Exception as e:
        # 예외 발생 시 로그 출력
        print(f"Error occurred: {e}")
        # 30초간 대기 후 다시 실행
        time.sleep(30)
