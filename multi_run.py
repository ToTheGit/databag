#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
from functools import partial

# In[ ]:


def run_script(place, places):
    import yeouido
    yeouido.main(place, places)

if __name__ == "__main__":
    places = [
 '강남 MICE 관광특구',
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
 '잠실한강공원']  # Add all 50 places here

    # Create a process pool and start all processes
    with multiprocessing.Pool(processes=5) as pool:
        pool.map(partial(run_script, places=places), places)

