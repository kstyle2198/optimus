import pandas as pd
import numpy as np
from tqdm import tqdm
from random import random, randrange, choice, shuffle
from datetime import datetime, timedelta, date
from itertools import combinations
from pprint import pprint
import calendar
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

import warnings
warnings.filterwarnings('ignore')

import functools
def unpack_df_columns(func):
    @functools.wraps(func)
    def _unpack_df_columns(*args, **kwargs):
        series = args[0]
        return func(*series.values)
    return _unpack_df_columns

@unpack_df_columns
def 최소착수요구일구하기(납기, 공기):
    result = pd.to_datetime(납기) - timedelta(days=int(공기))
    return result.date()

착수일가중치, 공기가중치, 크기가중치 = 1, 1, 1

@unpack_df_columns
def 블록우선순위구하기(날순, 공순, 크순):
    global 착수일가중치, 공기가중치, 크기가중치
    result = np.round((날순*float(착수일가중치) + 공순*float(공기가중치) + 크순*float(크기가중치))/3,1)
    return result

@unpack_df_columns
def 블록사이즈튜플만들기(가로, 세로):
    길이2 = max(가로, 세로)  
    길이1 = min(가로, 세로)
    result = (길이2, 길이1)
    return result

def 블록데이터전처리(블록원데이터):
    df1 = 블록원데이터.copy()
    df1["사이즈"] = df1[["가로", "세로"]].apply(블록사이즈튜플만들기, axis=1)
    df1["최대길이"] = df1["사이즈"].apply(lambda x: max(x))
    df1["납기"] = pd.to_datetime(df1["납기"])
    df1["면적"] = df1.eval("가로*세로")
    df1["최소착수요구일"] = df1[["납기", "표준공기"]].apply(최소착수요구일구하기, axis=1)
    df1["날짜순서"] = df1["최소착수요구일"].rank()
    df1["공기순서"] = df1["표준공기"].rank(ascending=False)
    df1["크기순서"] = df1["면적"].rank(ascending=False)
    df1["우선순위"] = df1[["날짜순서", "공기순서", "크기순서"]].apply(블록우선순위구하기, axis=1)
    
    df1 = df1.drop(['가로', '세로'], axis=1)
    df1 = df1.sort_values(by=["우선순위"])
    return df1

중량가중치, 크기가중치 = 1, 1

@unpack_df_columns
def 정반우선순위구하기(중순, 크순):
    global 중량가중치, 크기가중치
    result = np.round((중순*float(중량가중치) + 크순*float(크기가중치))/3,1)
    return result

@unpack_df_columns
def 정반사이즈튜플만들기(가로, 세로):
    길이1 = max(가로, 세로)  
    길이2 = min(가로, 세로)
    result = (길이1, 길이2)
    return result

def 정반데이터전처리(정반원데이터):
    df = 정반원데이터.copy()
    
    df["사이즈"] = df[["가로", "세로"]].apply(정반사이즈튜플만들기, axis=1)
    df["면적"] = df.eval("가로*세로")
    df["최대길이"] = df["사이즈"].apply(lambda x: max(x))
    df["중량순서"] = df["가능중량"].rank(ascending=False)
    df["크기순서"] = df["면적"].rank(ascending=False)
    df["우선순위"] = df[["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
    
    df = df.drop(['가로', '세로'], axis=1)
    df = df.sort_values(by=["우선순위"])
    return df


def 블록변수정리(블록데이터, target_block):
    블록 = dict()
    블록["블록명"] = target_block
    블록["인덱스"] =  블록데이터[블록데이터["블록명"]==target_block].index.values[0]
    블록["중량"] = 블록데이터[블록데이터["블록명"]==target_block]["중량"].values[0]
    블록["사이즈"] = 블록데이터[블록데이터["블록명"]==target_block]["사이즈"].values[0]
    블록["최대길이"] = 블록데이터[블록데이터["블록명"]==target_block]["최대길이"].values[0]
    블록["면적"] = 블록데이터[블록데이터["블록명"]==target_block]["면적"].values[0]
    블록["납기"] = 블록데이터[블록데이터["블록명"]==target_block]["납기"].values[0]
    블록["표준공기"] = 블록데이터[블록데이터["블록명"]==target_block]["표준공기"].values[0]
    블록["최소착수요구일"] = 블록데이터[블록데이터["블록명"]==target_block]["최소착수요구일"].values[0]
    return 블록

def 정반변수정리(정반데이터, 정반명):
    정반 = dict()
    정반["정반명"] = 정반명
    정반["가능중량"] = 정반데이터[정반데이터["정반명"]==정반명]["가능중량"].values[0]
    정반["사이즈"] = 정반데이터[정반데이터["정반명"]==정반명]["사이즈"].values[0]
    정반["최대길이"] = 정반데이터[정반데이터["정반명"]==정반명]["최대길이"].values[0]
    정반["면적"] = 정반데이터[정반데이터["정반명"]==정반명]["면적"].values[0]
    
    return 정반

def 정반가능중량검토(target_block, 블록데이터, 정반, 정반데이터):
    블록중량 = 블록데이터[블록데이터["블록명"]==target_block]["중량"].values[0]
    정반가능중량 = 정반데이터[정반데이터["정반명"]==정반]["가능중량"].values[0]
    
    if 정반가능중량 > 블록중량: 
        return "적합"
    else:
        return "부적합"
    
def 정반최대길이검토(target_block, 블록데이터, 정반, 정반데이터):
    블록최대길이 = 블록데이터[블록데이터["블록명"]==target_block]["최대길이"].values[0]
    정반최대길이 = 정반데이터[정반데이터["정반명"]==정반]["가능중량"].values[0]
    
    if 정반최대길이 > 블록최대길이: 
        return "적합"
    else:
        return "부적합"
    
def get_end_date_of_month(year, month):
    # Get the number of days in the given month
    num_days = calendar.monthrange(year, month)[1]
    # Return the last day of the month as a datetime object
    return date(year, month, num_days)


def create_면적달력(시작년:int, 시작월:int, 종료년:int, 종료월:int, 정반데이터):
    start_date = datetime(시작년, 시작월, 1)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    면적달력 = pd.DataFrame()
    면적달력.index = 날짜집합
   
    for 정반 in 정반집합:
        면적달력[f"{정반}"] = 정반데이터[정반데이터["정반명"]==정반]["면적"].values[0]

    return 면적달력

def get_최선조기착수일후보_가중(면적달력, 정반리스트, 대상블록, 블록데이터, 시점후퇴배수,  면적가중비율, 조기착수금지일수):
    '''
    최소착수요구일 보다 7일 이상 조기 착수 금지 로직 반영
    '''
    result_dict = dict()
    
    블록면적 = 블록데이터[블록데이터["블록명"]==대상블록]["면적"].values[0]
    표준공기 = 블록데이터[블록데이터["블록명"]==대상블록]["표준공기"].values[0]
    최소착수요구일_datetime = 블록데이터[블록데이터["블록명"]==대상블록]["최소착수요구일"].values[0]
    면적달력날짜리스트_datetime = list(면적달력.index)
    
    최소착수요구일 = 최소착수요구일_datetime.strftime("%Y-%m-%d")
    면적달력날짜리스트 = [date_obj.strftime("%Y-%m-%d") for date_obj in 면적달력날짜리스트_datetime]
    
    최소착수요구일달력인덱스 = 면적달력날짜리스트.index(최소착수요구일)
    
    한계시점인덱스 = max(0, 최소착수요구일달력인덱스 - 조기착수금지일수)  #### 최소요구착수일보다 7일이상 조기 착수하지 않도록!!

    for 대상정반 in 정반리스트:
        대상정반면적리스트 = 면적달력[f"{대상정반}"].iloc[:].values
        
        for i in range(len(대상정반면적리스트)):
            조회시점인덱스 = int(np.round(한계시점인덱스 + 표준공기 * 시점후퇴배수 * 0.1, 0))  ### 시점을 표준공기의 몇배만큼 뒤로 뺄 건가??

            조회구간면적리스트 = 대상정반면적리스트[조회시점인덱스:조회시점인덱스+표준공기]
            조회기간최소면적 = min(조회구간면적리스트)
            
            if 조회기간최소면적 >= 블록면적 * 면적가중비율:
                최선조기착수일 = 면적달력.index[조회시점인덱스]
                result_dict[대상정반] = 최선조기착수일
                break
        
    return result_dict

def get_최선조기착수대상(최선조기착수일후보, 최소요구착수일):
    # 복수의 경우 셔플후 랜덤 선택
    temp_list = list(최선조기착수일후보.items())
    shuffle(temp_list)
    최선조기착수일후보 = dict(temp_list)
    
    최소요구착수일 = pd.Timestamp(최소요구착수일)
    
#     print(temp_list)
    
    최소요구납기충족리스트 = []
    for 후보날짜 in temp_list:
        # 후보날짜 = pd.Timestamp(후보날짜)
        
        if 후보날짜[1] <= 최소요구착수일:
            최소요구납기충족리스트.append(후보날짜)
            
    # print(f">>> 최소요구납기충족리스트: {최소요구납기충족리스트}")
    
    if 최소요구납기충족리스트 != []:
        최소요구납기충족리스트 = dict(최소요구납기충족리스트)

        earliest_item = min(최소요구납기충족리스트.items(), key=lambda x: x[1])
        최선정반 = earliest_item[0]
        최선착수일 = earliest_item[1].strftime('%Y-%m-%d')
        return 최선정반, 최선착수일
    
    else:
        return None, None

def update_면적달력(면적달력, 최선조기착수대상, 배치블록명, 블록데이터):
    
    정반리스트 = 면적달력.columns.tolist()
    블록면적 = 블록데이터[블록데이터["블록명"]==배치블록명]["면적"].values[0]
    표준공기 = 블록데이터[블록데이터["블록명"]==배치블록명]["표준공기"].values[0]
    블록착수일 = 최선조기착수대상[1]
    시점인덱스 = list(면적달력.index.strftime('%Y-%m-%d')).index(블록착수일)
    정반 = 최선조기착수대상[0]
        

    조회기간면적리스트 = 면적달력[f"{정반}"].iloc[시점인덱스:시점인덱스+표준공기].values

    if min(조회기간면적리스트) >= 블록면적:
        for idx, 대상일면적 in enumerate(조회기간면적리스트):
            수정면적 = 대상일면적 - 블록면적
            대상일인덱스 = 시점인덱스 + idx
            면적달력[f"{정반}"].iloc[대상일인덱스:대상일인덱스+1] = 수정면적
            
        return 면적달력

def create_블록명달력(시작년:int, 시작월:int, 종료년:int, 종료월:int, 정반데이터):
    start_date = datetime(시작년, 시작월, 1)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    달력 = pd.DataFrame()
    달력.index = 날짜집합
    
    for 정반 in 정반집합:
        달력[정반] = [[] for _ in range(len(날짜집합))]
        
    return 달력

def update_블록명달력(블록명달력, 최선정반, 블록데이터, block_names, best_st_date):
    
    달력 = 블록명달력
    날짜집합 = 블록명달력.index.tolist()
    결과모음 = [[] for _ in range(len(날짜집합))]

    for block_name, 블록착수일 in zip(block_names, best_st_date):

        시점인덱스 = list(달력.index.strftime('%Y-%m-%d')).index(블록착수일)
        표준공기 = 블록데이터[블록데이터["블록명"]==block_name]["표준공기"].values[0]

        for i in 결과모음[시점인덱스:시점인덱스+표준공기]:
            i.append(block_name)
            
    달력[f"{최선정반}"] = 결과모음
    return 달력

def create_사이즈달력(시작년:int, 시작월:int, 종료년:int, 종료월:int, 정반데이터):
    start_date = datetime(시작년, 시작월, 1)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    달력 = pd.DataFrame()
    달력.index = 날짜집합
    
    for 정반 in 정반집합:
        달력[정반] = [[] for _ in range(len(날짜집합))]
        
    return 달력

    

def update_사이즈달력(사이즈달력, 최선정반, 블록데이터, block_names, block_sizes, best_st_date):
    
    달력 = 사이즈달력
    날짜집합 = 달력.index.tolist()
    결과모음 = [[] for _ in range(len(날짜집합))]
        
    for i in range(len(block_names)):
        블록명 = block_names[i]
        블록사이즈 = block_sizes[i]
        블록착수일 = best_st_date[i]
        
        표준공기 = 블록데이터[블록데이터["블록명"]==블록명]["표준공기"].values[0]
        시점인덱스 = list(달력.index.strftime('%Y-%m-%d')).index(블록착수일)
        
        for i in 결과모음[시점인덱스:시점인덱스+표준공기]:
            i.append(블록사이즈)
            
    달력[f"{최선정반}"] = 결과모음

    return 달력

def 정반세팅(사이즈):  #사이즈 = (10, 10)
    surface_width, surface_height = 사이즈[0], 사이즈[1]  # Adjusted to match the provided image for demonstration
    surface = np.zeros((surface_height, surface_width), dtype=int)
    return surface, surface_width, surface_height

def can_place_with_thresh(surface, surface_width, surface_height, block_height, block_width, start_row, start_col, thresh):
    new_width = surface_width - thresh
    new_height = surface_height - thresh
    
#     block_height, block_width = block_height, block_width
    if start_row + block_height > surface_height or start_col + block_width > surface_width:
        return False

    block_area = surface[start_row:start_row+block_height, start_col:start_col+block_width]
    if np.any(block_area != 0):
        return False
    
    if start_row > 0 and np.any(surface[start_row-thresh: start_row, start_col:start_col+block_width] != 0):
        return False
    if start_col > 0 and np.any(surface[start_row: start_row+block_height, start_col-thresh: start_col] != 0):
        return False
    
    return True



# Function to place a block on the surface, if possible
def place_block(surface, block_height, block_width, start_row, start_col, block_id):
    block_height, block_width = block_height, block_width
    surface[start_row:start_row + block_height, start_col:start_col + block_width] = block_id

# Function to find the best fit for a block on the surface
def find_best_fit_with_thresh(surface, surface_width, surface_height, block_height, block_width, block_id, thresh):
    best_fit_score = float('inf')
    best_position = None
    block_height, block_width = block_height, block_width

    # Iterate over all possible positions on the surface
    for y in range(surface_height - block_height + 1):
        for x in range(surface_width - block_width + 1):
            if can_place_with_thresh(surface, surface_width, surface_height, block_height, block_width, y, x, thresh):
                # Calculate a score; here we use the top-left corner (y, x) as the score
                # A lower score means the block is closer to the top-left
                score = y + x
                if score < best_fit_score:
                    best_fit_score = score
                    best_position = (y, x)

    # If a best position was found, place the block there
    if best_position:
        place_block(surface, block_height, block_width, *best_position, block_id)
        return True

    return False  # No fit found

# Function to fit blocks on the surface in order
def fit_blocks_with_thresh(surface, surface_width, surface_height, blocks, names, thresh):
    result = ""
    block_id = max(map(max, surface))+1  # Start numbering blocks from 1
    for name, block in zip(names, blocks):
        
        block_height, block_width = block
        
        if find_best_fit_with_thresh(surface, surface_width, surface_height, block_height, block_width, block_id, thresh) == False:
            # print(f"1차검토 - Block {name} of height {block_height} width {block_width} could not be placed.")
            result = "부적합"
            
            if block_height != block_width:  # 가로 세로 길이가 같지 않다면...
                
                ## 가로 세로 길이 바꿔서 검토 -------------------------------------
                block_height, block_width = block_width, block_height
                if find_best_fit_with_thresh(surface, surface_width, surface_height, block_height, block_width, block_id, thresh) == False:
                    # print(f"2차검토 - Block {name} of height {block_height} width {block_width} could not be placed.")
                    result = "부적합"
            
        else:
            result = "적합"
        block_id += 1  # Increment block_id for the next block
    return surface, result

def 정반배치레이아웃적합도(정반명, 정반데이터, 조회날짜, 블록명달력, 블록사이즈달력):

    thresh = 1
    
    정반사이즈 = 정반데이터[정반데이터["정반명"]==정반명]["사이즈"].values[0]
    block_names = 블록명달력.at[조회날짜, 정반명]
    block_sizes = 블록사이즈달력.at[조회날짜, 정반명]
        
    surface, surface_width, surface_height = 정반세팅(정반사이즈)
    배치결과 = fit_blocks_with_thresh(surface, surface_width, surface_height, block_sizes, block_names, thresh)
    적합도 = 배치결과[1]
    
    return 적합도

def check_if_value_in_list(my_list, value_to_check):
    if value_to_check in my_list:
        return "부적합"
    else:
        return "적합"
def 생산계획수립(블록데이터, 정반데이터, 면적달력, 블록명달력, 사이즈달력, 조기착수금지일수):
    
    블록리스트 = 블록데이터["블록명"].tolist()
    정반리스트 = 정반데이터["정반명"].tolist()
    
    면적달력 = 면적달력
    블록명달력 = 블록명달력
    사이즈달력 = 사이즈달력
    
    결론_정반_dict = {key: [] for key in 정반리스트}
    결론_블록_dict = {key: [] for key in 정반리스트}
    결론_블록사이즈_dict = {key: [] for key in 정반리스트}
    결론_정반_dict = {key: [] for key in 정반리스트}
    결론_착수일자_dict = {key: [] for key in 정반리스트}
    
    df_블록리스트 = []
    df_블록사이즈리스트 = []
    df_정반리스트 = []
    df_착수일자 = []
    df_상태정보 = []
    
    레이아웃배치실패 = []
    
    for _ in tqdm(range(len(블록리스트))):
                
        if 블록리스트:
            target_block = 블록리스트[0]
        else:
            # print("수정블록리스트에 검토대상 잔여블록이 없습니다.")
            break        
        
        블록 = 블록변수정리(블록데이터, target_block)
        블록명 = 블록["블록명"]
        블록사이즈 = 블록["사이즈"]
        납기 = 블록["납기"]
        표준공기 = 블록["표준공기"]
        최소착수요구일 = 블록["최소착수요구일"]
        
        # print(f"*** 검토대상블록: {블록명}, 사이즈: {블록사이즈}, 납기: {납기}, 표준공기: {표준공기}, 최소착수요구일:{최소착수요구일}")
        
        ## 정반가능중량 조건 적합도 검토 ----------------------------------------------------------
        중량조건적합정반들 = []
        정반리스트_iter = iter(정반리스트)
        
        try:
            while True:
                정반 = next(정반리스트_iter)
                중량적합도 = 정반가능중량검토(블록명, 블록데이터, 정반, 정반데이터)
                if 중량적합도 == "적합":
                    중량조건적합정반들.append(정반)
        except StopIteration:
            pass
                       

        # print(f"*** 중량조건적합정반들: {중량조건적합정반들}")
        
        ## 최장길이적합 조건 검토 ---------------------------------------
        최장길이조건적합정반들 = []
        중량조건적합_iter = iter(중량조건적합정반들)
        
        try:
            while True:
                정반 = next(중량조건적합_iter)
                최장길이적합도 = 정반최대길이검토(target_block, 블록데이터, 정반, 정반데이터)
                if 최장길이적합도 == "적합":
                    최장길이조건적합정반들.append(정반)
        except StopIteration:
            pass
                    
        # print(f"*** 최장길이조건적합정반들: {최장길이조건적합정반들}")
        
        ## 면적달력으로 최선조기착수일 검토 ---------------------------------------------------------------------
        if 최장길이조건적합정반들 != []:
            
            ####------------------------------------------------------------------------------------
            레이아웃배치순환 = [i for i in range(1, 31)]
            레이아웃배치순환_iterator = iter(레이아웃배치순환)
            
            try:     
                while True:
                    i = next(레이아웃배치순환_iterator)
                    # print()
                    # print(f">>> 제{i}차 레이아웃 적합도 검토")


                    최선조기착수일후보 = get_최선조기착수일후보_가중(면적달력, 최장길이조건적합정반들, 블록명, 블록데이터, i-1, 1, 조기착수금지일수) 
                    # print(f"*** 최선조기착수일후보: {최선조기착수일후보}")

                    최선조기착수대상 = get_최선조기착수대상(최선조기착수일후보, 최소착수요구일)  ##랜덤 셀렉트 적용

                    if 최선조기착수대상[0] == None:
                        df_상태정보.append('최선조기착수 후보날짜가 없음')
                        블록리스트.remove(블록명)
                        # print(">>> 최선조기착수 후보날짜가 없습니다.")

                        break


                    else:      #  최선조기착수대상[0] != None:
                        최선정반명 = 최선조기착수대상[0]
                        최선조기착수일 = 최선조기착수대상[1]

                        # print(f"*** 랜덤선택 최선정반명:{최선정반명}, 최선조기착수일:{최선조기착수일}")

                        결론_블록_dict[최선정반명].append(블록명)
                        결론_블록사이즈_dict[최선정반명].append(블록사이즈)
                        결론_정반_dict[최선정반명].append(최선정반명)
                        결론_착수일자_dict[최선정반명].append(최선조기착수일)

                        df_블록리스트.append(블록명)
                        df_블록사이즈리스트.append(블록사이즈)
                        df_정반리스트.append(최선정반명)
                        df_착수일자.append(최선조기착수일)

                        임시_블록명달력 = update_블록명달력(블록명달력, 최선정반명, 블록데이터, 결론_블록_dict[최선정반명], 결론_착수일자_dict[최선정반명])
                        임시_사이즈달력 = update_사이즈달력(사이즈달력, 최선정반명, 블록데이터, 결론_블록_dict[최선정반명], 결론_블록사이즈_dict[최선정반명], 결론_착수일자_dict[최선정반명])

                        ### 나중에 함수로 빼야...
                        레이아웃적합도리스트 = []
                        for i in range(표준공기):
                            date_object = datetime.strptime(최선조기착수일, "%Y-%m-%d")
                            new_date = date_object + timedelta(days=i)
                            레이아웃검토날짜 = new_date.strftime("%Y-%m-%d")   

                            적합도 = 정반배치레이아웃적합도(최선정반명, 정반데이터, 레이아웃검토날짜, 임시_블록명달력, 임시_사이즈달력)

                            레이아웃적합도리스트.append(적합도)

                        적합도리스트체크 = check_if_value_in_list(레이아웃적합도리스트, "부적합")
                        # print(f"*** 레이아웃 적합도리스트 체크: {적합도리스트체크}")

                        ### -----------------------------------------------------------------------------------------------------
                        if 적합도리스트체크 == "부적합":

                            결론_블록_dict[최선정반명] = 결론_블록_dict[최선정반명][:-1]                    
                            결론_블록사이즈_dict[최선정반명] = 결론_블록사이즈_dict[최선정반명][:-1]
                            결론_정반_dict[최선정반명] = 결론_정반_dict[최선정반명][:-1]
                            결론_착수일자_dict[최선정반명] = 결론_착수일자_dict[최선정반명][:-1]

                            df_블록리스트 = df_블록리스트[:-1]
                            df_블록사이즈리스트 = df_블록사이즈리스트[:-1]
                            df_정반리스트 = df_정반리스트[:-1]
                            df_착수일자 = df_착수일자[:-1]

                            임시_블록명달력 = None
                            임시_사이즈달력 = None

                            pass

                        else:
                            블록명달력 = 임시_블록명달력
                            사이즈달력 = 임시_사이즈달력
                            면적달력 = update_면적달력(면적달력, 최선조기착수대상, 블록명, 블록데이터)
                            블록리스트.remove(블록명)

                            df_상태정보.append('정상배치완료')
                            # print(f"*** 정상배치 완료")
                            break
            except StopIteration:
                pass
                # print("End of 정반레이아웃 iteration")
                
#             else:
#                 df_상태정보.append('최소요구납기 충족 불가능')
#                 블록리스트.remove(블록명)
#                 print(">>> 최소요구납기 충족 정반이 없습니다.")
                
        else:
            df_상태정보.append('중량 또는 최장길이 조건 충족 불가능')
            블록리스트.remove(블록명)
            # print(">>>중량 또는 최장길이 충족 정반이 없습니다.")
        
        # print("="*120)
        # print()
    
    return df_블록리스트, df_정반리스트, df_착수일자, df_상태정보, 면적달력, 블록명달력, 사이즈달력

@unpack_df_columns
def 종료일구하기(착수일자, 표준공기):
    try:
        original_date = datetime.strptime(착수일자, "%Y-%m-%d")
        종료날짜 = original_date + timedelta(days=int(표준공기)) 
        종료날짜 = 종료날짜.strftime("%Y-%m-%d")
        return 종료날짜
    except:
        pass

@unpack_df_columns
def create_text(블록명, 납기, 최소착수요구일, 착수일자,  종료일자):
    납기 = 납기.date()
    return str(블록명)+"/"+"납기:"+str(납기)+"/" +"최소착수요구일:"+str(최소착수요구일)+"/"+"착수일자:"+str(착수일자)+"/"+"종료일자:"+str(종료일자)






if __name__ == "__main__":
    start = time.time()


    
    print("Strat - 원 데이터 로드----------------------------------------------------")
    data_num = "_블록130개_정반6개"
    블록원데이터 = pd.read_excel(f"./data/data{data_num}.xlsx", sheet_name="블록데이터")
    정반원데이터 = pd.read_excel(f"./data/data{data_num}.xlsx", sheet_name="정반데이터")
    

    print("데이터 전처리--------------------------------------------------------------")
    블록데이터 = 블록데이터전처리(블록원데이터)
    정반데이터 = 정반데이터전처리(정반원데이터)

    print("달력데이터 생성 ---------------------------------------------------------")
    면적달력 = create_면적달력(2024, 1, 2024, 4, 정반데이터)
    블록명달력 = create_블록명달력(2024, 1, 2024, 4, 정반데이터)
    사이즈달력 = create_사이즈달력(2024, 1, 2024, 4, 정반데이터)

    착수일가중치들 = [1.0, 1.5, 2.0, 2.5, 3.0]
    공기가중치들 = [0.9, 0.75, 0.5, 0.25, 0.1]
    크기가중치들 = [0.1, 0.5, 0.9]
    조기착수금지일수 = 7

    배치결과모음 = []
    가중치모음 = []
    for _ in tqdm(range(10)):  #랜덤 10회 순환 검토
    
        # 블록가중치 랜덤 선택
        공기가중치 = choice(공기가중치들)
        크기가중치 = choice(크기가중치들)
        착수일가중치 = choice(착수일가중치들)
        
        가중치세트 = [착수일가중치, 공기가중치, 크기가중치]
        가중치모음.append(가중치세트)
        
        블록데이터 = 블록데이터전처리(블록원데이터)    
        면적달력 = create_면적달력(2024, 1, 2024, 4, 정반데이터)
        블록명달력 = create_블록명달력(2024, 1, 2024, 4, 정반데이터)
        사이즈달력 = create_사이즈달력(2024, 1, 2024, 4, 정반데이터)
            
        생산계획결과 = 생산계획수립(블록데이터, 정반데이터, 면적달력, 블록명달력, 사이즈달력, 조기착수금지일수)
        
        결론_블록리스트 = 생산계획결과[0]
        결론_정반리스트 = 생산계획결과[1]
        결론_착수일자 = 생산계획결과[2]
        결론_상태정보 = 생산계획결과[3]
        면적달력 = 생산계획결과[4]
        블록명달력 = 생산계획결과[5]
        사이즈달력 = 생산계획결과[6]

        # print(f">>>>>>>> 총 블록 대수: {len(결론_상태정보)}")
        # print(f">>>>>>>> 정상배치 블록 대수: {결론_상태정보.count('정상배치완료')}")
        # print(f">>>>>>>> 배치못한 블록 대수: {len(결론_상태정보) - 결론_상태정보.count('정상배치완료')}")
        # print("-"*70)
        
        fin_df = pd.DataFrame({
        "블록명":결론_블록리스트,
        "정반명":결론_정반리스트,
        "착수일자":결론_착수일자,
        })
        merged_df = pd.merge(블록데이터, fin_df, on="블록명", how="left")
        # print(f"결론_상태정보: {결론_상태정보}")
        merged_df["상태정보"] = 결론_상태정보
        
        merged_df["종료일자"] = merged_df[["착수일자", "표준공기"]].apply(종료일구하기, axis=1)
        merged_df["차트텍스트"] = merged_df[["블록명", "납기", "최소착수요구일", "착수일자",  "종료일자"]].apply(create_text, axis=1)
        
        배치결과모음.append(merged_df)

for i, 가중치세트 in enumerate(가중치모음):
    print(f">>> {int(i)+1}번째: 블록가중치 세트 --> {가중치세트} (착수일, 공기, 크기 순서)----------------------")
    t_df = 배치결과모음[i]
    t_df = t_df.dropna()
    t_df["종료월"] = t_df["종료일자"].apply(lambda x: x[5:7])
    g_df1 = t_df.groupby(["종료월"]).agg({"블록명":'count', "중량":'sum'})
    g_df1.columns = ["블록개수", "총조립중량"]
    print(">>> 월별 총조립량 -------------------")
    print(g_df1)

    end = time.time()
    print(f"실행 완료까지 걸린 시간 : {end-start:.2f}")