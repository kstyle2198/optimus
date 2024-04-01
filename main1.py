import pandas as pd
import numpy as np
from tqdm import tqdm
from random import random, randrange, choice, shuffle
from datetime import datetime, timedelta, date
from itertools import combinations
from pprint import pprint
import calendar
import plotly.express as px
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all" # Cell의 모든 반환값 출력


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


@unpack_df_columns
def 블록우선순위구하기(날짜순서, 공기순서, 크기순서):
    global 착수일가중치, 공기가중치, 크기가중치
    result = np.round((날짜순서*float(착수일가중치) + 공기순서*float(공기가중치) + 크기순서*float(크기가중치))/3,1)
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

def get_end_date_of_month(year, month):
    # Get the number of days in the given month
    num_days = calendar.monthrange(year, month)[1]
    # Return the last day of the month as a datetime object
    return date(year, month, num_days)


def create_면적달력(시작년:int, 시작월:int, 시작일: int, 종료년:int, 종료월:int, 정반데이터: pd.DataFrame) -> pd.DataFrame:
    start_date = datetime(시작년, 시작월, 시작일)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    면적달력 = pd.DataFrame()
    면적달력.index = 날짜집합
   
    for 정반 in 정반집합:
        면적달력[f"{정반}"] = 정반데이터[정반데이터["정반명"]==정반]["면적"].values[0]

    return 면적달력

def update_init_면적달력(면적달력:pd.DataFrame, 블록착수일:str, 정반:str, 블록면적:int, 표준공기:int) -> pd.DataFrame:
    
    정반리스트 = 면적달력.columns.tolist()
    시점인덱스 = list(면적달력.index.strftime('%Y-%m-%d')).index(블록착수일)   
    조회기간면적리스트 = 면적달력[f"{정반}"].iloc[시점인덱스:시점인덱스+표준공기].values

    for idx, 대상일면적 in enumerate(조회기간면적리스트):
        수정면적 = 대상일면적 - 블록면적
        대상일인덱스 = 시점인덱스 + idx
        면적달력[f"{정반}"].iloc[대상일인덱스:대상일인덱스+1] = 수정면적
        
    return 면적달력

def update_면적달력1(면적달력, 정반, 블록착수일, 배치블록명, 블록데이터):
    정반리스트 = 면적달력.columns.tolist()
    블록면적 = 블록데이터[블록데이터["블록명"]==배치블록명]["면적"].values[0]
    표준공기 = 블록데이터[블록데이터["블록명"]==배치블록명]["표준공기"].values[0]
    시점인덱스 = list(면적달력.index.strftime('%Y-%m-%d')).index(블록착수일)

    조회기간면적리스트 = 면적달력[f"{정반}"].iloc[시점인덱스:시점인덱스+표준공기].values
    
    print(f"조회기간 최소면적: {min(조회기간면적리스트)}, 블록면적: {블록면적}")
    if min(조회기간면적리스트) >= 블록면적:
        for idx, 대상일면적 in enumerate(조회기간면적리스트):
            수정면적 = 대상일면적 - 블록면적
            대상일인덱스 = 시점인덱스 + idx
            면적달력[f"{정반}"].iloc[대상일인덱스:대상일인덱스+1] = 수정면적
        return True, 면적달력
    else:
        print("면적부족으로 배치불가능")
        return False, 면적달력
    

def create_블록명달력(시작년:int, 시작월:int, 시작일: int, 종료년:int, 종료월:int, 정반데이터:pd.DataFrame) -> pd.DataFrame:
    start_date = datetime(시작년, 시작월, 시작일)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    달력 = pd.DataFrame()
    달력.index = 날짜집합
    
    for 정반 in 정반집합:
        달력[정반] = [[] for _ in range(len(날짜집합))]
        
    return 달력

def update_블록명달력(블록명달력:pd.DataFrame, 최선정반:str, 블록데이터:pd.DataFrame, block_names: list, best_st_date: list) -> pd.DataFrame:
    
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

def create_사이즈달력(시작년:int, 시작월:int, 시작일: int, 종료년:int, 종료월:int, 정반데이터:pd.DataFrame) -> pd.DataFrame:
    start_date = datetime(시작년, 시작월, 시작일)
    end_date = get_end_date_of_month(종료년, 종료월)
    정반집합 = 정반데이터["정반명"].tolist()
    날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')
    
    달력 = pd.DataFrame()
    달력.index = 날짜집합
    
    for 정반 in 정반집합:
        달력[정반] = [[] for _ in range(len(날짜집합))]
        
    return 달력

def update_사이즈달력(사이즈달력: pd.DataFrame, 최선정반:str, 블록데이터:pd.DataFrame, block_names:list, block_sizes:tuple, best_st_date:list) -> pd.DataFrame:
    
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
    result = "적합"
    block_id = max(map(max, surface))+1  # Start numbering blocks from 1
    for name, block in zip(names, blocks):
        
        block_height, block_width = block
        if find_best_fit_with_thresh(surface, surface_width, surface_height, block_height, block_width, block_id, thresh) == False:
            # print(f"-----1차검토 부적합 - Block {name} of height {block_height} width {block_width} could not be placed.")
            # result = "부적합"
            
            if block_height != block_width:  # 가로 세로 길이가 같지 않다면...
                ## 가로 세로 길이 바꿔서 검토 -------------------------------------
                block_height, block_width = block_width, block_height
                
                if find_best_fit_with_thresh(surface, surface_width, surface_height, block_height, block_width, block_id, thresh) == False:
                    # print(f"-----2차검토 부적합 - Block {name} of height {block_height} width {block_width} could not be placed.")
                    result = "부적합"
                else:
                    # print(f"2차검토 적합 - Block {name} of height {block_height} width {block_width} could be placed.")
                    result = "적합"
            else:
                pass
        else:
            # print(f"1차검토 적합 - Block {name} of height {block_height} width {block_width} could be placed.")
            result = "적합"
            
        block_id += 1  # Increment block_id for the next block
    return surface, result

def 정반배치레이아웃적합도(정반명, 정반데이터, 조회날짜, 블록명달력, 블록사이즈달력):
    
    block_names = 블록명달력.at[조회날짜, 정반명]
    block_sizes = 블록사이즈달력.at[조회날짜, 정반명]
    정반사이즈 = 정반데이터[정반데이터["정반명"]==정반명]["사이즈"].values[0]

    surface, surface_width, surface_height = 정반세팅(정반사이즈)
    배치결과 = fit_blocks_with_thresh(surface, surface_width, surface_height, block_sizes, block_names, thresh)
    적합도 = 배치결과[1]
    
    return 적합도

def draw1(surface, block_names):
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = mcolors.ListedColormap(['white'] + ['C{}'.format(i) for i in range(len(block_names))])
    norm = mcolors.BoundaryNorm(np.arange(0.5, len(block_names) + 2), cmap.N)

    ax.imshow(surface, cmap=cmap, norm=norm)

    for y in range(surface.shape[0]):
        for x in range(surface.shape[1]):
            if surface[y, x] != 0:
                ax.text(x, y, str(block_names[surface[y, x]-1]), ha='center', va='center', color='black', fontsize=6)
                
    ax.set_xticks(np.arange(-.5, surface.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, surface.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def 정반배치레이아웃(정반명, 정반데이터, 조회날짜, 블록명달력, 블록사이즈달력):
    
    정반사이즈 = 정반데이터[정반데이터["정반명"]==정반명]["사이즈"].values[0]
    block_names = 블록명달력.at[조회날짜, 정반명]
    block_sizes = 블록사이즈달력.at[조회날짜, 정반명]
        
    surface, surface_width, surface_height = 정반세팅(정반사이즈)
    배치결과 = fit_blocks_with_thresh(surface, surface_width, surface_height, block_sizes, block_names, thresh)
    배치레이아웃 = 배치결과[0]
    draw1(배치레이아웃, block_names)

def 중량조건체크(target_block):
    가능정반리스트 = []
    블록중량 = 블록데이터[블록데이터["블록명"]==target_block]["중량"].values[0]
    for 정반 in 정반데이터["정반명"]:
        정반가능중량 = 정반데이터[정반데이터["정반명"]==정반]["가능중량"].values[0]
        if 정반가능중량 > 블록중량:
            가능정반리스트.append(정반)       
    return 가능정반리스트   

def 최장길이체크(target_block:str, 중량가능정반들:list):
    가능정반리스트 = []
    블록최장길이 = 블록데이터[블록데이터["블록명"]==target_block]["최대길이"].values[0]
    for 정반 in 중량가능정반들:
        정반최장길이 = 정반데이터[정반데이터["정반명"]==정반]["가능중량"].values[0]
        if 정반최장길이 > 블록최장길이:
            가능정반리스트.append(정반)
    return 가능정반리스트

def 최선착수일후보체크(블록데이터, target_block:str, 길이가능정반들:list, 조기착수금지일수:int):
    result_dict = dict()
    # 블록납기 및 표준공기, 블록면적 확인
    블록데이터["납기"] = 블록데이터["납기"].astype('str')
    블록납기 = 블록데이터[블록데이터["블록명"]==target_block]["납기"].values[0]
    표준공기 = 블록데이터[블록데이터["블록명"]==target_block]["표준공기"].values[0]
    블록면적 = 블록데이터[블록데이터["블록명"]==target_block]["면적"].values[0]
    # 블록 최소착수요구일 확인
    최소착수요구일 = 블록데이터[블록데이터["블록명"]==target_block]["최소착수요구일"].values[0]
    최소착수요구일 = 최소착수요구일.strftime("%Y-%m-%d")
    # 면적달력날짜리스트 확인
    면적달력날짜리스트_datetime = list(면적달력.index)
    면적달력날짜리스트 = [date_obj.strftime("%Y-%m-%d") for date_obj in 면적달력날짜리스트_datetime]
    # 최소착수요구일인덱스 및 한계시점인덱스 확인
    블록납기일인덱스 = 면적달력날짜리스트.index(블록납기)
    최소착수요구일인덱스 = 면적달력날짜리스트.index(최소착수요구일)
    한계시점인덱스 = max(0, 최소착수요구일인덱스 - 조기착수금지일수)
    # print(f"최소착수요구일:{최소착수요구일}, 최소착수요구일인덱스:{최소착수요구일인덱스}, 한계시점인덱스:{한계시점인덱스}")
    
    for 대상정반 in 길이가능정반들:
        정반면적리스트 = 면적달력[f"{대상정반}"].iloc[한계시점인덱스:블록납기일인덱스].values
        # print(f"대상정반:{대상정반}- \n{정반면적리스트}")

        for idx in range(len(정반면적리스트)):
            조회구간면적리스트 = 정반면적리스트[idx:idx+표준공기]
            조회구간최소면적 = min(조회구간면적리스트)
            # print(f"{대상정반} - 조회구간면적리스트: {조회구간면적리스트}, 조회구간최소면적: {조회구간최소면적}, 블록면적: {블록면적}")

            if 조회구간최소면적 >= 블록면적:
                조회시점인덱스 = 한계시점인덱스 + idx
                최선조기착수일 = 면적달력.index[조회시점인덱스]
                result_dict[대상정반] = 최선조기착수일
                break
            else:
                pass

    return result_dict


def 최선조기착수대상선택(최선착수일후보):
    values = 최선착수일후보.values()
    unique_values = set(values)

    if len(unique_values) > 1:
        # 최선 날짜가 복수개 있으면 조기 착수일 선택
        최선조기착수정반 = min(최선착수일후보, key=lambda k: 최선착수일후보[k])
        최선조기착수날짜 = 최선착수일후보[최선조기착수정반]
        return (최선조기착수정반, 최선조기착수날짜)
    else:
        # 최선 날짜가 한개면, 랜덤 선택
        random_key = choice(list(최선착수일후보.keys()))
        return (random_key, 최선착수일후보[random_key])
    



if __name__ == "__main__":
    data_num = "_실행계획"

    시작년, 시작월, 시작일, 종료년, 종료월 = 2023, 10, 1, 2024, 4

    thresh = 1  # 블록 간격(m)
    조기착수금지일수 = 7

    블록원데이터 = pd.read_excel(f"D:/공정최적화/data/data1{data_num}.xlsx", sheet_name="블록데이터")
    기배치블록 = pd.read_excel(f"D:/공정최적화/data/data1{data_num}.xlsx", sheet_name="기배치블록")
    정반원데이터 = pd.read_excel(f"D:/공정최적화/data/data1{data_num}.xlsx", sheet_name="정반데이터")
    블록원데이터.shape

    검토개수 = 123
    블록원데이터 = 블록원데이터.iloc[:검토개수,:]
    블록원데이터.shape

    착수일가중치, 공기가중치, 크기가중치 = 5, 0.5, 0.1
    블록데이터 = 블록데이터전처리(블록원데이터)

    중량가중치, 크기가중치 = 0.5, 0.5
    정반데이터 = 정반데이터전처리(정반원데이터)
    블록데이터.shape, 정반데이터.shape

    면적달력 = create_면적달력(시작년, 시작월, 시작일, 종료년, 종료월, 정반데이터)
    블록명달력 = create_블록명달력(시작년, 시작월, 시작일, 종료년, 종료월, 정반데이터)
    사이즈달력 = create_사이즈달력(시작년, 시작월, 시작일, 종료년, 종료월, 정반데이터)

    기배치블록["블록면적"] = 기배치블록.eval("가로*세로")
    기배치블록["사이즈"] = 기배치블록[["가로", "세로"]].apply(블록사이즈튜플만들기, axis=1)
    기배치블록["착수일"] = 기배치블록["착수일"].dt.strftime('%Y-%m-%d')
    기배치블록["종료일"] = 기배치블록["종료일"].dt.strftime('%Y-%m-%d')
    for 블록착수일, 배치블록명, 블록면적, 표준공기, 배치정반 in zip(기배치블록["착수일"], 기배치블록["블록명"], 기배치블록["블록면적"], 기배치블록["공기"], 기배치블록["배치정반"],):
        면적달력 = update_init_면적달력(면적달력, 블록착수일, 배치정반, 블록면적, 표준공기)

    # 블록데이터 및 정반데이터 정리
    # 달력초기화 및 기진행블록 달력에 반영
    block_list = 블록데이터["블록명"].tolist()
    정반리스트 = 정반데이터["정반명"].tolist()

    블록명_dict = {key: [] for key in 정반리스트}
    착수일자_dict = {key: [] for key in 정반리스트}
    사이즈_dict = {key: [] for key in 정반리스트}
    배치상태 = []
    착수일자 = []

    for _ in tqdm(range(len(block_list))):
        
        target_block = block_list[0]
        print()
        print(f"***** 검토대상블록: {target_block} ---------------------------------------------")

        # 중량조건체크함수  --> 적합 정반리스트
        중량가능정반들 = 중량조건체크(target_block)
        # print(f"중량가능정반들: {중량가능정반들}")

        # 최장길이체크함수  --> 적합 정반리스트
        길이가능정반들 = 최장길이체크(target_block, 중량가능정반들)
        # print(f"길이가능정반들: {길이가능정반들}")

        # 날짜 및 면적 체크후 정반별 최선조기착수일 후보 검토
        최선착수일후보 = 최선착수일후보체크(블록데이터, target_block, 길이가능정반들, 7)      
        # print(f"최선착수일후보: {최선착수일후보}")
        
        # 최선 조기착수대상 선택 (착수일자가 빠른 것 우선)
        최선조기착수대상 = 최선조기착수대상선택(최선착수일후보)
        # print(f"최선조기착수대상: {최선조기착수대상}")
        최소착수요구일 = 블록데이터[블록데이터["블록명"]==target_block]["최소착수요구일"].values[0]
        최선정반, 최선착수일 = 최선조기착수대상[0], 최선조기착수대상[1]

        # 최선착수일과 최소착수요구일 간격 구하기
        최선착수일 = 최선착수일.date()
        gap = 최소착수요구일 - 최선착수일
        gap = int(gap.days + 1)

        # origin 변수 저장
        origin_블록명 = 블록명_dict[최선정반].copy()
        origin_착수일자 = 착수일자_dict[최선정반].copy()
        origin_사이즈 = 사이즈_dict[최선정반].copy()

        origin_면적달력 = 면적달력.copy()
        origin_블록명달력 = 블록명달력.copy()
        origin_사이즈달력 = 사이즈달력.copy()

        # 달력업데이트 변수 정의
        블록명_dict[최선정반].append(target_block)

        최선착수일 = 최선착수일.strftime("%Y-%m-%d")
        착수일자_dict[최선정반].append(최선착수일)
        
        사이즈 = 블록데이터[블록데이터["블록명"]==target_block]["사이즈"].values[0]
        사이즈_dict[최선정반].append(사이즈)

        # 기타 참고변수
        납기 = 블록데이터[블록데이터["블록명"]==target_block]["납기"].values[0]
        표준공기 = 블록데이터[블록데이터["블록명"]==target_block]["표준공기"].values[0]
        print(f"*** 최선정반: {최선정반}, 납기: {납기}, , 표준공기: {표준공기}, 최소요구착수일: {최소착수요구일}, 최선착수일: {최선착수일}, gap: {gap}, 사이즈: {사이즈}")


        
        # 레이아웃 체크 함수 ------------------------------------------------------------------------------------------------------------------------------------------------   
        status = "정상배치"

        # Gap의 00배수만큼 후퇴하면서 검토 (최소착수요구일 초과하더라도..)
        for g in range(gap*5):
            print(f">>>> 레이아웃검토: {g}")
        
            레이아웃적합도모음= []
            
            # 최선착수일 순환 계산 가산 계수
            g = 0 if g == 0 else 1
            
            최선착수일 = datetime.strptime(최선착수일, "%Y-%m-%d").date() + timedelta(days= g)
            최선착수일 = 최선착수일.strftime("%Y-%m-%d")
            착수일자_dict[최선정반][-1] = 최선착수일#.strftime("%Y-%m-%d")
            
            # print(f"{g}: {블록명_dict[최선정반]}")
            # print(f"{g}: {착수일자_dict[최선정반]}")

            블록명달력 = origin_블록명달력
            사이즈달력 = origin_사이즈달력
            
            # 레이아웃 적합도 판단을 위해 임시 배치한 달력을 생성
            임시_블록명달력 = update_블록명달력(블록명달력, 최선정반, 블록데이터, 블록명_dict[최선정반], 착수일자_dict[최선정반])
            임시_사이즈달력 = update_사이즈달력(사이즈달력, 최선정반, 블록데이터, 블록명_dict[최선정반], 사이즈_dict[최선정반], 착수일자_dict[최선정반])

            for k in range(표준공기):
                레이아웃배치검토시작일 = datetime.strptime(최선착수일, "%Y-%m-%d")  # + timedelta(days=g)
                new_date = 레이아웃배치검토시작일 + timedelta(days=k)
                레이아웃검토날짜 = new_date.strftime("%Y-%m-%d") 
                # print(f"--- g: {g}, k: {k}, 레이아웃배치검토시작일: {레이아웃배치검토시작일.strftime('%Y-%m-%d')}, 레이아웃검토날짜: {레이아웃검토날짜}")

                적합도 = 정반배치레이아웃적합도(최선정반, 정반데이터, 레이아웃검토날짜, 임시_블록명달력, 임시_사이즈달력)

                레이아웃적합도모음.append(적합도)

            print(f"레이아웃적합도모음: {레이아웃적합도모음}")
            
            if "부적합" not in  레이아웃적합도모음:

                레이아웃검토후최선착수일 = 레이아웃배치검토시작일.strftime("%Y-%m-%d")
                # print(f"레이아웃검토후최선착수일:  {레이아웃검토후최선착수일}")

                착수일자_dict[최선정반] = 착수일자_dict[최선정반][:-1]
                착수일자_dict[최선정반].append(레이아웃검토후최선착수일)
                
                # 달력 업데이트
                면적달력결과 = update_면적달력1(면적달력, 최선정반, 레이아웃검토후최선착수일, target_block, 블록데이터)
                # print(f"면적달력 업데이트 결과: {면적달력결과[0]}")

                if 면적달력결과[0] == True:
                    면적달력 = 면적달력결과[1]
                    블록명달력 = update_블록명달력(블록명달력, 최선정반, 블록데이터, 블록명_dict[최선정반], 착수일자_dict[최선정반])
                    사이즈달력 = update_사이즈달력(사이즈달력, 최선정반, 블록데이터, 블록명_dict[최선정반], 사이즈_dict[최선정반], 착수일자_dict[최선정반])
                    status = "정상배치"
                    착수일 = 레이아웃검토후최선착수일
                    print(">>>>> 정상배치 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print()
                    break
                    
                else:
                    
                    면적달력 = origin_면적달력            
                    블록명달력 = origin_블록명달력
                    사이즈달력 = origin_사이즈달력
            
            else:
                status = "배치불가"
                착수일 = None
                # print("부적합")
                # print()
                
                면적달력 = origin_면적달력            
                블록명달력 = origin_블록명달력
                사이즈달력 = origin_사이즈달력
            # print("배치불가")
        
        
        배치상태.append(status)
        착수일자.append(착수일)
        block_list.remove(target_block)
        print("="*80)
        time.sleep(0.1)



    배치후데이터 = 블록데이터.copy()
    배치후데이터["배치결과"] = 배치상태
    배치후데이터["착수일자"] = 착수일자
    배치후데이터['최소착수요구일'] = pd.to_datetime(배치후데이터['최소착수요구일'])
    배치후데이터['착수일자'] = pd.to_datetime(배치후데이터['착수일자'])
    배치후데이터["델타"] = 배치후데이터['착수일자'] - 배치후데이터['최소착수요구일']
    배치후데이터.to_csv("final.csv", encoding="utf-8-sig")
    면적달력.to_csv("면적달력.csv", encoding="utf-8-sig")
    블록명달력.to_csv("블록명달력.csv", encoding="utf-8-sig")
    사이즈달력.to_csv("사이즈달력.csv", encoding="utf-8-sig")

    print(f"{len(배치상태)}, {배치상태.count('정상배치')}, {배치상태.count('배치불가')}")


