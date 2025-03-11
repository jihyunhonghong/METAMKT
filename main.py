from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from evaluate_prediction import *
from batch_service import create_version
from typing import List


# FastAPI 앱 생성
app = FastAPI()


# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 평가 예측 API 엔드포인트
@app.get("/predict/detail")
def evaluate_prediction(adword_id: str = Query(..., title="광고ID" ,description="광고ID를 입력해주세요.", example="120211716936450003"),
date: str = Query(..., title="예측일", description="예측일을 입력해주세요. EX)YYYY-MM-DD", example="2024-12-30")
,time: str = Query(..., title="예측시간", description="예측시간을 입력해주세요. EX)HH:mm", example="17:30")):
    print("evaluate_prediction endpoint called")
    results = execute_stored_procedure(
        adword_id=adword_id,
        date=date,
        time=time,
    )
    if not results:
        return {}
    return results

# 캠페인 정보 API 엔드포인트


# /predict 라우트 추가
@app.get("/predict/list")
def predict_list(advertiser: List[str] = Query(default=[], title="광고주 배열", description="광고주 ID를 쉼표(,)로 구분하여 입력해주세요.", example=["2", "3", "145"])
,adword_type: List[str] = Query(default=[], title="매체 배열", description="매체 ID를 쉼표(,)로 구분하여 입력해주세요.", example=["2", "3", "4"])
,description: List[str] = Query(default=[], title="이벤트 배열", description="이벤트를 쉼표(,)로 구분하여 입력해주세요.", example=["대구점_모발이식", "미니리프팅", "3개월(수원)"])
,date: str = Query(..., title="예측일", description="예측일을 입력해주세요. EX)YYYY-MM-DD", example="2024-12-31")
,time: str = Query(..., title="예측시간", description="예측시간을 입력해주세요. EX)HH:mm", example="17:30")
,model_time: str = Query(..., title="모델 값", description="1일모델:1440,2일모델:2880, ..., 7일모델:10080", example="1440")
,list_type: str = Query(..., title="구분 값", description="캠페인 : CAMPAIGN, 광고 : ADWORD", example="CAMPAIGN")
,search: str = Query(default="", title="검색어", description="검색어를 입력해주세요.")
,page_size: int  = Query(default=10, title="페이지 사이즈", description="페이지 사이지를 입력해주세요.", example=10)
,page_number: int = Query(default=1, title="페이지 번호", description="페이지 번호를 입력해주세요.", example=1)
,sort_by: str = Query(default="PRE_SPEND", title="정렬조건", description="컬럼명 기재 EX)PRED_IMPRESSIONS", example="PRED_IMPRESSIONS")
,sort_direction: str = Query(default="DESC", title="정렬기준", description="정렬 기준을 입력해주세요.", example="ASC")):
    print("predict endpoint called")
    results = execute_sp_campaign_list_procedure(
        advertiser=advertiser,
        adword_type=adword_type,
        description=description,
        date=date,
        time=time,
        model_time=model_time,
        list_type=list_type,
        search=search,
        page_size=page_size,
        page_number=page_number,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )
    if not results:
        return {}
    return results


# /create_version 라우트 추가
@app.get("/predict/model/chart")
def get_model_chart(adword_id: str = Query(..., title="광고ID" ,description="광고ID를 입력해주세요.", example="120211716936450003"),
date: str = Query(..., title="예측일", description="예측일을 입력해주세요. EX)YYYY-MM-DD", example="2025-01-06")
,time: str = Query(..., title="예측시간", description="예측시간을 입력해주세요. EX)HH:mm", example="12:30")
,model_time: str = Query(..., title="모델 값", description="1일모델:1440,2일모델:2880, ..., 7일모델:10080", example="1440")):
    print("predict endpoint called")
    results = execute_sp_prediction_model_time_procedure(
        adword_id=adword_id,
        date=date,
        time=time,
        model_time=model_time
    )
    if not results:
        return {}
    return results


# /create_version 라우트 추가
@app.get("/predict/time")
def get_model_time(date: str = Query(..., title="기준시간", description="기준시간을 입력해주세요. EX)YYYY-MM-DD", example="2025-01-06")):
    print("predict endpoint called")
    
    results = execute_sp_select_ver_procedure(date=date)

    if not results:
        return {}
    return results

###batch
@app.post("/create_version", include_in_schema=False)
def batch_create_version():
    create_version()
