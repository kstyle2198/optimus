import streamlit as st
import pandas as pd
import numpy as np
# from tqdm import tqdm
from datetime import datetime, timedelta
import openpyxl

st.set_page_config(
    page_title="정반공정최적화",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

## Data Load ############

블록원데이터 = pd.read_excel("./data/data1.xlsx", sheet_name="블록데이터", engine='openpyxl')
정반원데이터 = pd.read_excel("./data/data1.xlsx", sheet_name="정반데이터", engine='openpyxl')


## State Variable #############

if '블록원데이터' not in st.session_state:
    st.session_state['블록원데이터'] = pd.DataFrame()

if '정반원데이터' not in st.session_state:
    st.session_state['정반원데이터'] = pd.DataFrame()

# if '정반집합' not in st.session_state:
#     st.session_state['정반집합'] = ""

if '블록데이터' not in st.session_state:
    st.session_state['블록데이터'] = pd.DataFrame()

if '정반데이터' not in st.session_state:
    st.session_state['정반데이터'] = pd.DataFrame()

if '배치달력' not in st.session_state:
    st.session_state['배치달력'] = pd.DataFrame()

if '공기달력' not in st.session_state:
    st.session_state['공기달력'] = pd.DataFrame()

if '공백순서달력' not in st.session_state:
    st.session_state['공백순서달력'] = pd.DataFrame()

if '최종배정결과' not in st.session_state:
    st.session_state['최종배정결과'] = pd.DataFrame()

if '병합최종결과' not in st.session_state:
    st.session_state['병합최종결과'] = pd.DataFrame()

if '수정블록리스트' not in st.session_state:
    st.session_state['수정블록리스트'] = []

if '수정정반리스트' not in st.session_state:
    st.session_state['수정정반리스트'] = []

## Function #################################
import functools
def unpack_df_columns(func):
    @functools.wraps(func)
    def _unpack_df_columns(*args, **kwargs):
        series = args[0]
        return func(*series.values)
    return _unpack_df_columns

@unpack_df_columns
def 최소요구착수일구하기(납기, 공기):
    result = pd.to_datetime(납기) - timedelta(days=int(공기))
    return result.date()

@unpack_df_columns
def 블록우선순위구하기(날순, 공순, 크순):
    global 착수일가중치, 공기가중치, 크기가중치
    result = np.round((날순*착수일가중치 + 공순*공기가중치 + 크순*크기가중치)/3,1)
    return result

def 블록데이터전처리(블록원데이터):
    df1 = 블록원데이터.copy()
    df1["납기"] = pd.to_datetime(df1["납기"])
    df1["사이즈"] = df1.eval("가로*세로")
    df1["최소요구착수일"] = df1[["납기", "표준공기"]].apply(최소요구착수일구하기, axis=1)
    df1["날짜순서"] = df1["최소요구착수일"].rank()
    df1["공기순서"] = df1["표준공기"].rank(ascending=False)
    df1["크기순서"] = df1["사이즈"].rank(ascending=False)
    df1["우선순위"] = df1[["날짜순서", "공기순서", "크기순서"]].apply(블록우선순위구하기, axis=1)
    df1 = df1.sort_values(by=["우선순위"])
    return df1

@unpack_df_columns
def 정반우선순위구하기(중순, 크순):
    global 중량가중치, 크기가중치
    result = np.round((중순*중량가중치 + 크순*크기가중치)/3,1)
    return result

def 정반데이터전처리(정반원데이터):
    df2 = 정반원데이터.copy()
    df2["사이즈"] = df2.eval("가로 * 세로")
    df2["중량순서"] = df2["가능중량"].rank(ascending=False)
    df2["크기순서"] = df2["사이즈"].rank(ascending=False)
    df2["우선순위"] = df2[["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
    df2 = df2.sort_values(by=["우선순위"])
    return df2

def create_init_calendar(날짜집합, 정반집합):
    배치달력 = pd.DataFrame()
    배치달력.index = 날짜집합
   
    for 정반 in 정반집합:
        배치달력[f"{정반}"] = 0

    return 배치달력

def update_배치달력(배치달력, 정반명, 착수날짜, 필요공기, 정반집합):
    
    신규칼럼리스트 = 정반집합.copy()
    try:
        for 현칼럼 in 배치달력.columns:
            신규칼럼리스트.remove(현칼럼)

        for 신규칼럼 in 신규칼럼리스트:
            배치달력[f"{신규칼럼}"] = 0

        시점인덱스 = list(배치달력.index.strftime('%Y-%m-%d')).index(착수날짜)
        배치달력[f"{정반명}"].iloc[시점인덱스:시점인덱스+필요공기] = 1
        return 배치달력
    except:
        return 배치달력
    
def create_공기달력(배치달력, 날짜집합, 정반집합):
    total_list = []

    for 정반 in 정반집합:
        검토대상 = 배치달력[f"{정반}"].tolist()

        new_list = []
        new_num = 0
        for idx, i in enumerate(검토대상):
            if i == 0:
                new_num = new_num  + 1
                new_list.append(new_num)
            else:
                new_list.append(0)
                new_num = 0
        total_list.append(new_list)
        
    new_total = []
    for original_list in total_list:

        result_list = []
        group = []
        for num in original_list:
            if num == 0 and group:
                result_list.extend(reversed(group))
                group = []
            group.append(num)

        result_list.extend(reversed(group))

        new_total.append(result_list)

    공기달력 = pd.DataFrame()
    공기달력.index = 날짜집합

    for idx, 정반 in enumerate(정반집합):
        공기달력[f"{정반}"] =  new_total[idx]

    for 정반 in 정반집합:
        if 공기달력[f"{정반}"][0]== 0:
            공기달력[f"{정반}"] = 공기달력[f"{정반}"].shift(1)
        else:
            pass
    공기달력.fillna(0, inplace=True)
    return 공기달력

def create_공백순서달력(배치_달력, 정반집합, 날짜집합):
    total = []

    for 정반 in 배치_달력.columns.tolist():
        
        input_list = 배치_달력[f"{정반}"].tolist()
        
        counter = 1
        result_list = []

        for idx, x in enumerate(input_list):

            if idx == 0:
                if x == 1:
                    result_list.append(0)
                else:
                    result_list.append(counter)
                    counter += 1

            else:   
                if input_list[idx-1] == 1 and x == 0:
                    result_list.append(counter)
                    counter += 1
                else:
                    result_list.append(0)

        total.append(result_list)

    공백순서달력 = pd.DataFrame()
    공백순서달력.index = 날짜집합

    for idx, 정반 in enumerate(배치_달력.columns.tolist()):
        공백순서달력[f"{정반}"] =  total[idx]

    return 공백순서달력

def 착수가능일찾기(공기달력, 공백순서달력, 정반, 표준공기):
        
    first_zeros = []
    
    for idx, i in enumerate(공백순서달력[f"{정반}"].tolist()):
        if i != 0:
            first_zeros.append(공백순서달력.index.strftime('%Y-%m-%d').values[idx])

    for idx, 착수가능일 in enumerate(first_zeros):
        
        
        착수가능일인덱스 = list(공기달력.index.strftime('%Y-%m-%d')).index(first_zeros[idx])
        착수가능일의확보가능공기 = 공기달력[f"{정반}"].iloc[착수가능일인덱스]
        
        if 착수가능일의확보가능공기 > 표준공기:
            
            return 착수가능일
        
        else:
            pass

def 생산계획수립():

    if st.button("🗑️ States변수 초기화"):
        st.session_state['블록원데이터'] = pd.DataFrame()
        st.session_state['정반원데이터'] = pd.DataFrame()
        st.session_state['정반집합'] = ""
        st.session_state['블록데이터'] = pd.DataFrame()
        st.session_state['정반데이터'] = pd.DataFrame()
        st.session_state['배치달력'] = pd.DataFrame()
        st.session_state['공기달력'] = pd.DataFrame()
        st.session_state['공백순서달력'] = pd.DataFrame()
        st.info("States 변수 초기화 완료")

    ## Initial Settings
    with st.expander("📜 원데이터 로딩"):
        if st.button("Raw Data Loading"):
            st.session_state['블록원데이터'] = 블록원데이터
            st.session_state['정반원데이터'] = 정반원데이터
        col01, col02 = st.columns([6, 4])
        with col01:
            st.markdown(f"📏 Dataframe Shape: **{st.session_state['블록원데이터'].shape}**")
            st.dataframe(st.session_state['블록원데이터'], use_container_width=True)
        with col02:
            st.markdown(f"📏 Dataframe Shape: **{st.session_state['정반원데이터'].shape}**")
            st.dataframe(st.session_state['정반원데이터'], use_container_width=True)

    with st.expander("📝 전처리 데이터 - 우선순위 계산후 정렬"):
        if st.button("Preprocessed Data Loading", key="dfkjl"):
            st.session_state['블록데이터'] = 블록데이터전처리(st.session_state['블록원데이터'])
            st.session_state['정반데이터'] = 정반데이터전처리(st.session_state['정반원데이터'])
        col1, col2 = st.columns([6, 4])
        with col1:
            st.markdown(f"🧱 블록데이터 - Shape : **{st.session_state['블록데이터'].shape}**")
            st.dataframe(st.session_state['블록데이터'], use_container_width=True)
        with col2:
            st.markdown(f"🧇 정반데이터 - Shape : **{st.session_state['정반데이터'].shape}**")
            st.dataframe(st.session_state['정반데이터'], use_container_width=True)

    # 캘린더 로딩
    with st.expander("📆 캘린더 생성"):
        global start_date, end_date

        날짜집합  = pd.date_range(start=start_date, end=end_date, freq='D')

        if st.button("Calendar Loading"):
            최초정반집합 = st.session_state['정반데이터']["정반명"].unique().tolist()
            st.session_state['배치달력'] = create_init_calendar(날짜집합, 최초정반집합)

            # 기 배치 생산계획 하드코딩 반영
            st.session_state['배치달력'].iloc[:2,:] = 1
            st.session_state['배치달력'].iloc[3:5,:1] = 1
            st.session_state['배치달력'].iloc[:4,1:2] = 1
            st.session_state['배치달력'].iloc[:5,2:3] = 1

            st.session_state['정반집합'] = st.session_state['배치달력']. columns.tolist()

            st.session_state['공기달력'] = create_공기달력(st.session_state['배치달력'], 날짜집합, st.session_state['정반집합'])
            st.session_state['공백순서달력'] = create_공백순서달력(st.session_state['배치달력'], st.session_state['정반집합'], 날짜집합)

        col11, col12, col13 = st.columns(3)
        with col11:
            st.markdown("📅 배치달력 - 1이면 기배치, 0이면 배치가능")
            st.dataframe(st.session_state['배치달력'].style.highlight_min(axis=0), use_container_width=True)
        with col12:
            st.markdown("📅 공기달력-날짜별 연속확보 가능 공기")
            st.dataframe(st.session_state['공기달력'].style.highlight_min(axis=0), use_container_width=True)
        with col13:
            st.markdown("📅 공백순서달력- 공백구간 간의 순서")
            st.dataframe(st.session_state['공백순서달력'].style.highlight_min(axis=0), use_container_width=True)

    # 결과모음리스트
    배정된블록 = []
    배정된정반 = []
    착수일 = []

    with st.expander("생산계획수립"):

        if st.button("📈 생산계획수립"):

            st.session_state['수정블록리스트'] = st.session_state['블록데이터']["블록명"].tolist()
            st.session_state['수정정반리스트'] = st.session_state['정반데이터']["정반명"].tolist()

            for target_block in st.session_state['수정블록리스트']:

                blk_index = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block].index.values[0]
                target_weight = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["중량"].values[0]
                target_size = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["사이즈"].values[0]
                least_start_date = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["최소요구착수일"].values[0]
                target_표준공기 = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["표준공기"].values[0]
                st.warning(f"타겟블록정보 - idx{blk_index}, 블록명 {target_block}, 무게 {target_weight}, 사이즈 {target_size}, 최소요구착수일 {least_start_date}, 표준공기 {target_표준공기}")
                st.markdown(f"(전단) 수정블록리스트 : {st.session_state['수정블록리스트']}, 수정정반리스트 : {st.session_state['수정정반리스트']}")
            
                ## Target Block의 가능 정반 리스트를 먼저 찾아라 #####################################################
                가능정반_1순위 = []
                가능정반_2순위 = []
                
                for 정반 in st.session_state['수정정반리스트']:
                    공백순서1인덱스 = list(st.session_state['공백순서달력'][f"{정반}"]).index(1)
                    공백순서1인덱스의날짜 = st.session_state['배치달력'].index[공백순서1인덱스]
                    공백순서1인덱스의날짜의확보가능공기 = st.session_state['공기달력'][f"{정반}"][공백순서1인덱스]
                    공백순서1인덱스날짜의공백순서 = st.session_state['공백순서달력'][f"{정반}"][공백순서1인덱스]
                    weight_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==정반]["가능중량"].values[0]
                    size_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==정반]["사이즈"].values[0]

                    st.markdown(f"검토정반: {정반}, size_capa: {size_capa}")

                    try:
                        공백순서2인덱스 = list(st.session_state['공백순서달력'][f"{정반}"]).index(2)
                        공백순서2인덱스의날짜 = st.session_state['배치달력'].index[공백순서2인덱스]

                        공백순서2인덱스의날짜의확보가능공기 = st.session_state['공기달력'][f"{정반}"][공백순서2인덱스]
                        공백순서2인덱스날짜의공백순서 = st.session_state['공백순서달력'][f"{정반}"][공백순서2인덱스]
                    
                    except:
                        pass

                    if 공백순서1인덱스의날짜 <= least_start_date and 공백순서1인덱스의날짜의확보가능공기 >= target_표준공기 and weight_capa >= target_weight and size_capa >= target_size and 공백순서1인덱스날짜의공백순서 == 1:
                        st.markdown(f"타겟블록{target_block}는 정반 {정반}가 1순위 가능정반에 해당")
                        가능정반_1순위.append(정반)
                    elif 공백순서2인덱스의날짜 <= least_start_date and 공백순서2인덱스의날짜의확보가능공기 >= target_표준공기 and weight_capa >= target_weight and size_capa >= target_size and 공백순서2인덱스날짜의공백순서 == 2:
                        st.markdown(f"타겟블록{target_block}는 정반 {정반}가 2순위 가능정반에 해당")
                        가능정반_2순위.append(정반)
                    else:
                        pass

                ### 1순위와 2순위를 따로 저장하면 안된다.. 어느 정반의 2순위가 다른 정반의 1순위보다 조기 착수가 가능할 수 있다.
                ### 그래서 1~2순위 정반 + 날짜를 다 모은후, 가능 여부를 판단해야 한다.
                    
                st.markdown(f"1순위 가능정반 : {가능정반_1순위}")
                st.markdown(f"2순위 가능정반 : {가능정반_2순위}")

                if 가능정반_1순위 != []:

                    for 가능정반 in 가능정반_1순위:

                        weight_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["가능중량"].values[0]
                        size_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["사이즈"].values[0]

                        # 잔여면적 체크후 정반쪼개기 검토
                        잔여면적비율 = (size_capa - target_size) / size_capa

                        st.markdown(f"적격한 검토정반: {가능정반}, size_capa: {size_capa}, target_size:{target_size}, 잔여면적비율 :{잔여면적비율}")

                        if 잔여면적비율 >= 정반쪼개는면적_Thresh:
                            새정반이름 = 가능정반+"_추가"
                            새정반면적 = size_capa * 잔여면적비율
                            새오큐파이드 = 0
                            기존정반새면적 = size_capa - 새정반면적
                            가능정반인덱스 = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반].index.values[0]
                            
                            st.session_state['정반데이터'].loc[가능정반인덱스,"사이즈"] = 기존정반새면적
                            
                            st.session_state['정반데이터'].loc[len(st.session_state['정반데이터'])] = {"정반명":새정반이름, "가능중량": weight_capa, "사이즈":새정반면적, "occupied":새오큐파이드}
                            
                            st.session_state['정반데이터']["중량순서"] = st.session_state['정반데이터']["가능중량"].rank(ascending=False)
                            st.session_state['정반데이터']["크기순서"] = st.session_state['정반데이터']["사이즈"].rank(ascending=False)
                            st.session_state['정반데이터']["우선순위"] = st.session_state['정반데이터'][["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
                            st.session_state['정반데이터'] = st.session_state['정반데이터'].sort_values(by=["우선순위"])
                            
                            st.info(f"잔여면적비율 {np.round(잔여면적비율,1)*100}%로 30% 이상이므로 정반 쪼개기 - 새이름 {새정반이름} / 새면적 {새정반면적} / 기존정반수정면적-{기존정반새면적}") 
                        




                            ## 새정반 추가사항 달력에 반영
                            st.markdown(f"새정반이름 : {새정반이름}, 수정정반리스트: {st.session_state['수정정반리스트']}")
                            st.session_state['수정정반리스트'].append(새정반이름)    
                            st.info(f"수정정반집합- {st.session_state['수정정반리스트']}")





                        else:
                            st.info(f"1순위 정반의 잔여 면적이 Thresh 비율보다 작아 쪼갤 수 없습니다.")


                        착수가능일 = 착수가능일찾기(st.session_state['공기달력'], st.session_state['공백순서달력'], 가능정반, target_표준공기)

                            
                        배정결과 = {"블록명": target_block, "정반명": 가능정반, "착수일": 착수가능일}    

                        st.success(f"최종배정결과 - {배정결과}")

                        배정된블록.append(target_block)
                        배정된정반.append(가능정반)
                        착수일.append(착수가능일)


                        st.markdown(f"(배치정반빼기전) 수정블록리스트 : {st.session_state['수정블록리스트']}, 수정정반리스트 : {st.session_state['수정정반리스트']}")

                        # st.session_state['수정정반리스트'].remove(가능정반) #[st.session_state['수정정반리스트'] != 가능정반]
                        st.session_state['수정블록리스트'].remove(target_block) #[st.session_state['수정블록리스트'] != target_block]

                        st.markdown(f"(배치정반제거후) 수정블록리스트 : {st.session_state['수정블록리스트']}, 수정정반리스트 : {st.session_state['수정정반리스트']}")

                        st.session_state['배치달력'] =  update_배치달력(st.session_state['배치달력'], 가능정반, 착수가능일, target_표준공기, st.session_state['수정정반리스트']) 
                        st.session_state['배치달력'].iloc[:2,:] = 1    ## 갱신시 하드코딩 2일차까지 강제로 1 채우기
                        st.session_state['공기달력'] = create_공기달력(st.session_state['배치달력'], 날짜집합, st.session_state['수정정반리스트'])
                        st.session_state['공백순서달력'] = create_공백순서달력(st.session_state['배치달력'], st.session_state['수정정반리스트'], 날짜집합)

                        st.session_state['배치달력']
                        st.session_state['공기달력']
                        st.session_state['공백순서달력']

                        break
                    

                elif 가능정반_2순위 != []:

                    for 가능정반 in 가능정반_2순위:

                        weight_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["가능중량"].values[0]
                        size_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["사이즈"].values[0]

                        # 잔여면적 체크후 정반쪼개기 검토
                        잔여면적비율 = (size_capa - target_size) / size_capa

                        st.markdown(f"검토정반: {가능정반}, size_capa: {size_capa}, target_size:{target_size}, 잔여면적비율 :{잔여면적비율}")

                        if 잔여면적비율 >= 정반쪼개는면적_Thresh:
                            새정반이름 = 가능정반+"_추가"
                            새정반면적 = size_capa * 잔여면적비율
                            새오큐파이드 = 0
                            기존정반새면적 = size_capa - 새정반면적
                            가능정반인덱스 = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반].index.values[0]
                            
                            st.session_state['정반데이터'].loc[가능정반인덱스,"사이즈"] = 기존정반새면적
                            
                            st.session_state['정반데이터'].loc[len(st.session_state['정반데이터'])] = {"정반명":새정반이름, "가능중량": weight_capa, "사이즈":새정반면적, "occupied":새오큐파이드}
                            
                            st.session_state['정반데이터']["중량순서"] = st.session_state['정반데이터']["가능중량"].rank(ascending=False)
                            st.session_state['정반데이터']["크기순서"] = st.session_state['정반데이터']["사이즈"].rank(ascending=False)
                            st.session_state['정반데이터']["우선순위"] = st.session_state['정반데이터'][["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
                            st.session_state['정반데이터'] = st.session_state['정반데이터'].sort_values(by=["우선순위"])
                            
                            st.info(f"잔여면적비율 {np.round(잔여면적비율,1)*100}%로 30% 이상이므로 정반 쪼개기 - 새이름 {새정반이름} / 새면적 {새정반면적} / 기존정반수정면적-{기존정반새면적}") 
                        
                            ## 새정반 추가사항 달력에 반영

                            st.markdown(f"새정반이름 : {새정반이름}, 수정정반리스트: {st.session_state['수정정반리스트']}")
                            st.session_state['수정정반리스트'].append(새정반이름)    
                            st.info(f"수정정반리스트- {st.session_state['수정정반리스트']}")
                        else:
                            st.info(f"2순위 정반의 잔여 면적이 Thresh 비율보다 작아 쪼갤 수 없습니다.")


                        착수가능일 = 착수가능일찾기(st.session_state['공기달력'], st.session_state['공백순서달력'], 가능정반, target_표준공기)
                        
                            
                        배정결과 = {"블록명": target_block, "정반명": 가능정반, "착수일": 착수가능일}
                        st.success(f"최종배정결과 - {배정결과}")

                        배정된블록.append(target_block)
                        배정된정반.append(가능정반)
                        착수일.append(착수가능일)

                        # st.session_state['수정정반리스트'].remove(가능정반) #[st.session_state['수정정반리스트'] != 가능정반]
                        st.session_state['수정블록리스트'].remove(target_block) #[st.session_state['수정블록리스트'] != target_block]
                        st.markdown(f"(2순위 후단) 수정블록리스트 : {st.session_state['수정블록리스트']}, 수정정반리스트 : {st.session_state['수정정반리스트']}")


                        st.session_state['배치달력'] =  update_배치달력(st.session_state['배치달력'], 가능정반, 착수가능일, target_표준공기, st.session_state['수정정반리스트']) 
                        st.session_state['배치달력'].iloc[:2,:] = 1   ## 갱신시 하드코딩 2일차까지 강제로 1 채우기

                        st.session_state['배치달력']


                        st.session_state['공기달력'] = create_공기달력(st.session_state['배치달력'], 날짜집합, st.session_state['수정정반리스트'])
                        st.session_state['공백순서달력'] = create_공백순서달력(st.session_state['배치달력'], st.session_state['수정정반리스트'], 날짜집합)

                        
                        break


                else:
                    pass


        st.session_state['최종배정결과'] = pd.DataFrame({"블록명":배정된블록, "정반명":배정된정반, "착수일":착수일})
        st.markdown("##### 🌻 최종 블록 - 정반 배정 결과")
        st.dataframe(st.session_state['최종배정결과'], use_container_width=True)













    #             for 가능정반 in st.session_state['정반데이터'][st.session_state['정반데이터']["occupied"]==0]["정반명"]:
                    
    #                 가능정반인덱스 = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반].index.values[0]
    #                 weight_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["가능중량"].values[0]
    #                 size_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["사이즈"].values[0]
    #                 착수가능일 = 착수가능일찾기(st.session_state['공기달력'], st.session_state['공백순서달력'], 가능정반, target_표준공기)
    #                 착수가능일공백순서 = st.session_state['공백순서달력'].loc[착수가능일, 가능정반]
    #                 st.warning(f"검토정반 - idx{가능정반인덱스}, 정반명{가능정반}, 가능중량{weight_capa}, 사이즈{size_capa}, 착수가능일 {착수가능일}, 공백순서 {착수가능일공백순서}")
                    
    #                 if weight_capa >= target_weight and size_capa >= target_size and 착수가능일공백순서 == 1:  ##########################
                        
    #                     st.success(f"{target_block}를 {가능정반}에 배치가능 / 착수가능일- {착수가능일} / 공백순서- {착수가능일공백순서}")





    #                     st.session_state['정반데이터'].loc[가능정반인덱스, "occupied"] = 1


    #                     st.session_state['블록데이터'].loc[blk_index, "정반배치"] = 1
                                
    #                     ## 날짜까지 이상없으면 잔여면적 검토후 정반 쪼개기
    #                     잔여면적비율 = (size_capa - target_size) / size_capa
                        
    #                     if 잔여면적비율 >= 정반쪼개는면적_Thresh:
    #                         새정반이름 = 가능정반+"_추가"
    #                         새정반면적 = size_capa * 잔여면적비율
    #                         새오큐파이드 = 0
    #                         기존정반새면적 = size_capa - 새정반면적
                            
    #                         st.session_state['정반데이터'].loc[가능정반인덱스,"사이즈"] = 기존정반새면적
                            
    #                         st.session_state['정반데이터'].loc[len(st.session_state['정반데이터'])] = {"정반명":새정반이름, "가능중량": weight_capa, "사이즈":새정반면적, "occupied":새오큐파이드}
                            
    #                         st.session_state['정반데이터']["중량순서"] = st.session_state['정반데이터']["가능중량"].rank(ascending=False)
    #                         st.session_state['정반데이터']["크기순서"] = st.session_state['정반데이터']["사이즈"].rank(ascending=False)
    #                         st.session_state['정반데이터']["우선순위"] = st.session_state['정반데이터'][["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
    #                         st.session_state['정반데이터'] = st.session_state['정반데이터'].sort_values(by=["우선순위"])
                            
    #                         st.info(f"잔여면적비율 {np.round(잔여면적비율,1)*100}%로 30% 이상이므로 정반 쪼개기 - 새이름 {새정반이름} / 새면적 {새정반면적} / 기존정반수정면적-{기존정반새면적}") 
                        
    #                         ## 새정반 추가사항 달력에 반영
    #                         st.session_state['정반집합'].append(새정반이름)    
    #                         # 새정반집합 = st.session_state['정반집합']
    #                         st.info(f"새정반집합- {st.session_state['정반집합']}")

    #                     st.session_state['배치달력'] =  update_배치달력(st.session_state['배치달력'], 가능정반, 착수가능일, target_표준공기, st.session_state['정반집합']) 
    #                     st.session_state['배치달력'].iloc[:2,:] = 1
    #                     st.session_state['공기달력'] = create_공기달력(st.session_state['배치달력'], 날짜집합, st.session_state['정반집합'])
    #                     st.session_state['공백순서달력'] = create_공백순서달력(st.session_state['배치달력'], st.session_state['정반집합'], 날짜집합)
                            
    #                     배정결과 = {"블록명": target_block, "정반명": 가능정반, "착수일": 착수가능일}    
    #                     st.success(f"최종배정결과 - {배정결과}")
    #                     st.markdown("---")

    #                     배정된블록.append(target_block)
    #                     배정된정반.append(가능정반)
    #                     착수일.append(착수가능일)
                        
    #                     st.session_state['수정블록리스트'] = st.session_state['수정블록리스트'][st.session_state['수정블록리스트'] != target_block]
    #                     break    
                        
    #                 else:
    #                     st.markdown("정반배정불가 - 1차")
    #                     pass

            
    #         st.markdown(f"2차검토 블록리스트 : {st.session_state['수정블록리스트']}")

    #         for target_block in st.session_state['수정블록리스트']:
    #             print(f"2차: {target_block}")

    #             blk_index = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block].index.values[0]
    #             target_weight = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["중량"].values[0]
    #             target_size = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["사이즈"].values[0]
    #             target_표준공기 = st.session_state['블록데이터'][st.session_state['블록데이터']["블록명"]==target_block]["표준공기"].values[0]
    #             st.warning(f"타겟블록정보 - idx{blk_index}, 블록명 {target_block}, 무게 {target_weight}, 사이즈 {target_size}")
                
    #             # 가능정반리스트 = st.session_state['정반데이터'][st.session_state['정반데이터']["occupied"]==0]["정반명"].tolist()
    #             # st.warning(f"가능정반리스트 - {가능정반리스트}")
            
    #             for 가능정반 in st.session_state['정반데이터'][st.session_state['정반데이터']["occupied"]==0]["정반명"]:


                    
    #                 가능정반인덱스 = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반].index.values[0]
    #                 weight_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["가능중량"].values[0]
    #                 size_capa = st.session_state['정반데이터'][st.session_state['정반데이터']["정반명"]==가능정반]["사이즈"].values[0]
    #                 착수가능일 = 착수가능일찾기(st.session_state['공기달력'], st.session_state['공백순서달력'], 가능정반, target_표준공기)
    #                 착수가능일공백순서 = st.session_state['공백순서달력'].loc[착수가능일, 가능정반]
    #                 st.warning(f"검토정반 - idx{가능정반인덱스}, 정반명{가능정반}, 가능중량{weight_capa}, 사이즈{size_capa}, 착수가능일 {착수가능일}, 공백순서 {착수가능일공백순서}")
                    
    #                 if weight_capa >= target_weight and size_capa >= target_size: # and 착수가능일공백순서 == 2:  ########################
                        
    #                     st.success(f"{target_block}를 {가능정반}에 배치가능 / 착수가능일- {착수가능일} / 공백순서- {착수가능일공백순서}")
    #                     st.session_state['정반데이터'].loc[가능정반인덱스, "occupied"] = 1
    #                     st.session_state['블록데이터'].loc[blk_index, "정반배치"] = 1
                                
    #                     ## 날짜까지 이상없으면 잔여면적 검토후 정반 쪼개기
    #                     잔여면적비율 = (size_capa - target_size) / size_capa
                        
    #                     if 잔여면적비율 >= 정반쪼개는면적_Thresh:
    #                         새정반이름 = 가능정반+"_추가"
    #                         새정반면적 = size_capa * 잔여면적비율
    #                         새오큐파이드 = 0
    #                         기존정반새면적 = size_capa - 새정반면적
                            
    #                         st.session_state['정반데이터'].loc[가능정반인덱스,"사이즈"] = 기존정반새면적
                            
    #                         st.session_state['정반데이터'].loc[len(st.session_state['정반데이터'])] = {"정반명":새정반이름, "가능중량": weight_capa, "사이즈":새정반면적, "occupied":새오큐파이드}
                            
    #                         st.session_state['정반데이터']["중량순서"] = st.session_state['정반데이터']["가능중량"].rank(ascending=False)
    #                         st.session_state['정반데이터']["크기순서"] = st.session_state['정반데이터']["사이즈"].rank(ascending=False)
    #                         st.session_state['정반데이터']["우선순위"] = st.session_state['정반데이터'][["중량순서", "크기순서"]].apply(정반우선순위구하기, axis=1)
    #                         st.session_state['정반데이터'] = st.session_state['정반데이터'].sort_values(by=["우선순위"])
                            
    #                         st.info(f"잔여면적비율 {np.round(잔여면적비율,1)*100}%로 30% 이상이므로 정반 쪼개기 - 새이름 {새정반이름} / 새면적 {새정반면적} / 기존정반수정면적-{기존정반새면적}") 
                        
    #                         ## 새정반 추가사항 달력에 반영
    #                         st.session_state['정반집합'].append(새정반이름)    
    #                         # 새정반집합 = st.session_state['정반집합']
    #                         st.info(f"새정반집합- {st.session_state['정반집합']}")

    #                     st.session_state['배치달력'] =  update_배치달력(st.session_state['배치달력'], 가능정반, 착수가능일, target_표준공기, st.session_state['정반집합']) 
    #                     st.session_state['배치달력'].iloc[:2,:] = 1
    #                     st.session_state['공기달력'] = create_공기달력(st.session_state['배치달력'], 날짜집합, st.session_state['정반집합'])
    #                     st.session_state['공백순서달력'] = create_공백순서달력(st.session_state['배치달력'], st.session_state['정반집합'], 날짜집합)
                            
    #                     배정결과 = {"블록명": target_block, "정반명": 가능정반, "착수일": 착수가능일}    
    #                     st.success(f"최종배정결과 - {배정결과}")
    #                     st.markdown("---")

    #                     배정된블록.append(target_block)
    #                     배정된정반.append(가능정반)
    #                     착수일.append(착수가능일)
                        
    #                     st.session_state['수정블록리스트'] = st.session_state['수정블록리스트'][st.session_state['수정블록리스트'] != target_block]
    #                     break    
                        
    #                 else:
    #                     st.markdown("정반배정불가-2차")
    #                     pass
            
    #         st.markdown(f"최종 잔여블록 리스트 : {st.session_state['수정블록리스트']}")

    

    # st.session_state['최종배정결과'] = pd.DataFrame({"블록명":배정된블록, "정반명":배정된정반, "착수일":착수일})
    # st.markdown("##### 🌻 최종 블록 - 정반 배정 결과")
    # st.dataframe(st.session_state['최종배정결과'], use_container_width=True)

##############################################3

if __name__ == "__main__":
    st.title("🚢 :blue[미포조선] :red[정반배치] :green[최적화 분석(Prototype)]")
    st.markdown("---")

    with st.expander("✏️ :blue[**검토개요**]"):
        st.markdown('''
                    - 본 서비스는 **미포조선 중조립 및 대조립 블록의 정반 배치 최적화** 프로토타입 분석 결과를 제공함 (데이터셋을 작게하여)
                    - 블록리스트상 블록들에 우선순위 부여, 정반리스트상 정반들에 우선순위를 부여하고 각각의 Weight값 설정 및 곱셉후 우선순위에 따른 배치를 함
                    - 우선순위 동일 블록, 동일 정반간에는 랜덤 배치
                    - 정반 배치후 정반의 잔여 면적이 기존 면적의 30% 이상이면 정반 분리하여 새정반으로 추가 
                    - 목적함수는 기준시점의 총 조립중량으로 생각중...
                    - 최적조합 산출 방식은 유전알고리즘 등 참고하여 검토 중 (Weight 및 Thresh로 다수 조합생성후 비교 고려...)
                    ''')

    tab1, tab2, tab3 = st.tabs(["🐳 **계획수립**", "🐬 **계획분석**", "공란"])
    with tab1:

        # global 변수들
        착수일가중치, 공기가중치, 크기가중치 = 0.7, 0.5, 0.5
        중량가중치, 크기가중치 = 0.5, 0.7
        정반쪼개는면적_Thresh = 0.3
        start_date = datetime(2024, 2, 1)
        end_date = datetime(2024, 2, 28)

        생산계획수립()

    with tab2:

        try:
            st.session_state['병합최종결과'] = pd.merge(st.session_state['블록데이터'], st.session_state['최종배정결과'], on="블록명", how="left")
            new_orders = ["블록명", "중량", "가로", "세로", "사이즈", "정반배치", "정반명", "최소요구착수일", "착수일", "표준공기", "납기", "날짜순서", "공기순서", "크기순서", "우선순위"]
            st.session_state['병합최종결과'] = st.session_state['병합최종결과'][new_orders]
        except:
            st.empty()

        with st.expander("🌞 **배치후 블록 및 정반 데이터**", expanded=True):

            col31, col32 = st.columns(2)
            with col31:
                st.dataframe(st.session_state['병합최종결과'], use_container_width=True)
            with col32:
                st.dataframe(st.session_state['정반데이터'], use_container_width=True)

        with st.expander("📑 **계획후 달력 조회**"):

            col21, col22, col23 = st.columns(3)
            with col21:
                st.markdown("📅 배치달력 - 1이면 기배치, 0이면 배치가능")
                st.dataframe(st.session_state['배치달력'].style.highlight_min(axis=0), use_container_width=True)
            with col22:
                st.markdown("📅 공기달력-날짜별 연속확보 가능 공기")
                st.dataframe(st.session_state['공기달력'].style.highlight_min(axis=0), use_container_width=True)
            with col23:
                st.markdown("📅 공백순서달력- 공백구간 간의 순서")
                st.dataframe(st.session_state['공백순서달력'].style.highlight_min(axis=0), use_container_width=True)

        st.markdown('''
                    **검토자 중간의견**
                    - 상기 데이터면, 간트 시각화 및 시점별 목적함수값 산출 가능할 듯...
                    - 유전알고리즘의 변수조합을.. 정반-블록-날짜로 할지.. 아니면.. 우선순위 Weight 값들로 할지는 생각해봐야 할 듯..
                    ''')

    with tab3:
        st.empty()
