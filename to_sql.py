import pandas as pd
import requests
from xml.etree import ElementTree as ET
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import traceback
import pandas as pd
import mysql.connector

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


def city_to_db(update): # 인구 혼잡도 dataframe of places
  for index, place in enumerate(places):
    url = f'http://openapi.seoul.go.kr:8088/796e7558466e777334394148697a62/xml/citydata/1/5/{place}'
    print(f'{index}번째 for문입니다. 값은 {place}입니다.')
    response = requests.get(url).text
    tree = ET.fromstring(response)
    ET.indent(tree, space="  ")
    update.append(tree[2][1][0][17].text)
    if len(update) == 1:
      return city_to_db(update)
    else:
      if update[-1] == update[-2]:
        listOf = []
        listOf.append(tree[2][0].text)
        listOf.append(tree[2][1][0][0].text)
        listOf.append(tree[2][2][0][3].text)
        listOf.append(tree[2][8][0][19].text)
        listOf.append(tree[2][8][0][24][-1][5].text)

        columns = []
        columns.append('hotspot_name')
        columns.append('congestion_level')
        columns.append('road_traffic_spd')
        columns.append('pm_10')
        columns.append('sky_status_level')


        if index == 0:
          city = pd.DataFrame([tuple(listOf)], columns=columns) # listOf를 transpose한 후에 tuple로 바꿔주기 때문에 원하는대로 18개의 element
        else:
          city.loc[len(city)] = listOf
      else:
        return city_to_db(update)
  city.loc[:, 'congestion_level'] = city['congestion_level'].replace({'여유': 1, '보통': 2, '약간 붐빔': 3, '붐빔': 4})
  city.loc[:, 'sky_status_level'] = city['sky_status_level'].replace({'맑음': 1, '구름많음': 2, '흐림': 3})
  city['pm_10'] = city['pm_10'].fillna('NULL')
  return city, update

def runtime_taken(start, end, mid):
    time_diff = end - start
    time_diff_2 = end - mid
    print("saveD", end, "| 소요시간: ", time_diff, "| DB저장 소요시간: ", time_diff_2)

def make_tree():
    url = 'http://openapi.seoul.go.kr:8088/796e7558466e777334394148697a62/xml/citydata/1/5/강남역'
    response = requests.get(url).text
    tree = ET.fromstring(response)
    ET.indent(tree, space="  ")
    return tree

def to_sql(city):
    cnx = mysql.connector.connect(user='dbid231', password='dbpass231',
                                  host='localhost',
                                  database='db23103')
    cursor = cnx.cursor()

    for i, row in city.iterrows():
        update_query = f"UPDATE hotspots_tb SET hotspot_name = '{row['hotspot_name']}', congestion_level = {row['congestion_level']}, road_traffic_spd = {row['road_traffic_spd']}, pm_10 = {row['pm_10']}, sky_status_level = {row['sky_status_level']} WHERE hotspot_id = {i}"
        cursor.execute(update_query)

    # 데이터베이스 연결 종료
    cnx.commit()
    cursor.close()
    cnx.close()

update = []
z = 1
while True:
    try:
      if z == 1: #update
        tree = make_tree()
        update.append(tree[2][1][0][17].text)
        print("프로그램 시작(서버갱신):", update[-1])
        z += 1
      else:
        tree = make_tree()
        if (z != 2) & (update[-1] == tree[2][1][0][17].text):
            print("서버 갱신 안됨")
            time.sleep(30) # 갱신 1초 전에 실행 -> 갱신된지 29초 후에 실행 -> 3,4분 소요 => 총 4분 30초(최대 5분)
                            #     continue #다시 처음부터 while문 시작
        else:
            update.append(tree[2][1][0][17].text) #서버갱신시간 time 리스트에 추가
            start = datetime.now()
            print("서버갱신: " , update[-1], "| 현재 시각: ",start) #서버갱신 알림출력
            city, update = city_to_db(update) #population 데이터프레임 추출 함수 -> population
            mid = datetime.now()
            print("DB저장시작: ", mid)
            to_sql(city)
            end = datetime.now()
            runtime_taken(start, end, mid) # 소요시간 함수
            update = [update[-1]]
    #             time.sleep(60) # 1분 대기 후
        z +=1
    except Exception as e:
        # 예외 발생 시 로그 출력
        print(f"Error occurred: {e}")
        # 30초간 대기 후 다시 실행
        time.sleep(30)
