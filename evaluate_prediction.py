from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime, timedelta
import pyodbc
import math

# SQLAlchemy 엔진 생성
user = "ai_user"
password = "mkt2024_!@#"
host = "27.96.134.44"
port = 1433
database = "master"

encoded_password = quote(password)
connection_url = f"mssql+pyodbc://{user}:{encoded_password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_url)

# 저장 프로시저 실행 함수
def execute_stored_procedure(adword_id, date, time):
    try:
        with engine.connect() as conn:
            # 광고 기본 정보를 가져오는 쿼리
            campaign_query = text("EXEC SP_GET_CAMPAIGN_INFO :adword_id, :date, :time")
            campaign_result = conn.execute(campaign_query, {"adword_id": adword_id, "date": date, "time": time}).fetchone()

            if not campaign_result:
                return None

            campaign_columns = [
                "ADWORD_TYPE", "CAMPAIGN_NAME", "ADWORD_SET_NAME", "ADWORD_NAME",
                "DATE", "TIME", "IMPRESSIONS", "CLICKS", "SPEND"
            ]
            campaign_data = dict(zip(campaign_columns, campaign_result))

            # 7일 예측 데이터를 가져오는 쿼리
            prediction_query = text("EXEC SP_PREDICTION_EVALUATION :adword_id, :date, :time")
            prediction_results = conn.execute(prediction_query, {"adword_id": adword_id, "date": date, "time": time}).fetchall()

            if not prediction_results:
                campaign_data["FORECAST_LIST"] = []
                return campaign_data

            prediction_columns = [
                "DATETIME", "PRED_IMPRESSIONS", "IMPRESSIONS", "IMPRESSIONS_SCORE",
                "PRED_CLICKS", "CLICKS", "CLICKS_SCORE",
                "PRED_SPEND", "SPEND", "SPEND_SCORE"
            ]

            prediction_data = [
                dict(zip(prediction_columns, row)) for row in prediction_results
            ]            

            campaign_data["FORECAST_LIST"] = prediction_data

            return campaign_data

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass          

def execute_sp_campaign_list_procedure(advertiser, adword_type, description,date,time,model_time,list_type,search,page_size,page_number,sort_by,sort_direction):
    print("predict endpoint called")

    try:
        with engine.connect() as conn:

            # 리스트를 쉼표로 구분된 문자열로 변환
            advertiser_str = ",".join(advertiser)
            adword_type_str = ",".join(adword_type)
            description_str = ",".join(description)

            offset = (page_number-1)*page_size

            date_obj = datetime.strptime(date, '%Y-%m-%d')
            add_date = date_obj + timedelta(minutes=int(model_time))
            add_date_str = add_date.strftime('%Y-%m-%d')

            search_type = "C"
            if list_type == "ADWORD":
                search_type = "A"

            if list_type == "ADWORD":
                query = text("EXEC SP_ADWORD_LIST :model_time, :date, :time, :offset, :page_size, :advertiser_str, :adword_type_str, :description_str, :search_value, :sort_by, :sort_direction")
                params = {
                    "model_time": int(model_time),
                    "date": add_date_str,
                    "time": time,
                    "offset": offset,
                    "page_size": page_size,
                    "advertiser_str": advertiser_str,
                    "adword_type_str": adword_type_str,
                    "description_str": description_str,
                    "search_value": search,
                    "sort_by": sort_by,
                    "sort_direction": sort_direction,
                }

                result = conn.execute(query, params).fetchall()

                if not result:
                    return  {
                        "LIST": [],
                        "TOTAL_DATA": 0,
                        "TOTAL_PAGE": 1,
                        "PAGE_SIZE": page_size,
                        "PAGE_NUMBER": page_number,
                        "DATE" : date,
                        "TIME" : time,
                        "MODEL_TIME" : model_time                        
                    }   

                columns = [
                    "ID", "ADWORD_TYPE", "NAME",
                    "STATUS", "DATETIME", "PRED_IMPRESSIONS", "IMPRESSIONS", "PRED_CLICKS",
                    "CLICKS", "PRED_SPEND", "SPEND", "PRE_SPEND"
                ]

                cnt_query = text("EXEC SP_ADWORD_LIST_COUNT :model_time, :date, :time, :advertiser_str, :adword_type_str, :description_str, :search_value")
                cnt_params = {
                    "model_time": int(model_time),
                    "date": add_date_str,
                    "time": time,
                    "advertiser_str": advertiser_str,
                    "adword_type_str": adword_type_str,
                    "description_str": description_str,
                    "search_value": search
                }                

                list_cnt = 0
                cnt_result = conn.execute(cnt_query, cnt_params).fetchone()

                if cnt_result:
                    list_cnt = int(cnt_result[0])

                data = {
                    "LIST": [
                        {**dict(zip(columns, row))} for row in result
                    ],
                    "TOTAL_DATA": list_cnt,
                    "TOTAL_PAGE": math.ceil(list_cnt / page_size),
                    "PAGE_SIZE": page_size,
                    "PAGE_NUMBER": page_number,
                    "DATE" : date,
                    "TIME" : time,
                    "MODEL_TIME" : model_time                        
                }   
                  
                return data

            else:
                query = text("EXEC SP_CAMPAIGN_LIST :model_time, :date, :time, :offset, :page_size, :advertiser_str, :adword_type_str, :description_str, :search_value, :sort_by, :sort_direction")
                params = {
                    "model_time": int(model_time),
                    "date": add_date_str,
                    "time": time,
                    "offset": offset,
                    "page_size": page_size,
                    "advertiser_str": advertiser_str,
                    "adword_type_str": adword_type_str,
                    "description_str": description_str,
                    "search_value": search,
                    "sort_by": sort_by,
                    "sort_direction": sort_direction,
                }

                result = conn.execute(query, params).fetchall()

                if not result:
                    return  {
                        "LIST": [],
                        "TOTAL_DATA": 0,
                        "TOTAL_PAGE": 1,
                        "PAGE_SIZE": page_size,
                        "PAGE_NUMBER": page_number,
                        "DATE" : date,
                        "TIME" : time,
                        "MODEL_TIME" : model_time                        
                    }   

                columns = [
                    "ADWORD_ID", "ADWORD_TYPE", "CAMPAIGN_ID", "CAMPAIGN_NAME", "CAMPAIGN_BUDGET",
                    "ADWORD_SET_ID", "ADWORD_SET_NAME", "ADWORD_SET_BUDGET", "ADWORD_NAME","PRED_DATE",
                    "PRED_TIME", "A_PRED_IMPRESSIONS", "A_IMPRESSIONS", "A_PRED_CLICKS","A_CLICKS",
                    "A_PRED_SPEND", "A_SPEND", "CAMPAIGN_STATUS", "ADWORD_SET_STATUS", "ADWORD_STATUS",
                    "DATETIME", "PRED_IMPRESSIONS", "IMPRESSIONS", "PRED_CLICKS","CLICKS",
                    "PRED_SPEND", "SPEND", "S_PRED_IMPRESSIONS", "S_IMPRESSIONS", "S_PRED_CLICKS",
                    "S_CLICKS", "S_PRED_SPEND", "S_SPEND","PRE_SPEND"
                ]

                campaign_dict = {}

                for row in result:
                    # Extract values from row
                    campaign_id = row[2]
                    adword_set_id = row[5]
                    adword_id = row[0]

                    # Top-level campaign data
                    if campaign_id not in campaign_dict:
                        campaign_dict[campaign_id] = {
                            "ID": row[2],
                            "ADWORD_TYPE": row[1],
                            "STATUS": row[17],
                            "NAME": row[3],
                            "DATETIME": row[20],
                            "BUDGET": row[4],
                            "PRED_IMPRESSIONS": row[21],
                            "IMPRESSIONS": row[22],
                            "PRED_CLICKS": row[23],
                            "CLICKS": row[24],
                            "PRED_SPEND": row[25],
                            "SPEND": row[26],
                            "PRE_SPEND": row[33],
                            "ADWORD_SET_LIST": []
                        }

                    campaign = campaign_dict[campaign_id]
                    adword_set_list = campaign["ADWORD_SET_LIST"]

                    adword_set = next(
                        (aset for aset in adword_set_list if aset.get("ID") == adword_set_id),
                        None
                    )                    

                    if not adword_set:
                        adword_set = {
                            "ID": row[5],
                            "ADWORD_TYPE": row[1],
                            "STATUS": row[18],
                            "NAME": row[6],
                            "DATETIME": row[20],
                            "BUDGET": row[7],
                            "PRED_IMPRESSIONS": row[27],
                            "IMPRESSIONS": row[28],
                            "PRED_CLICKS": row[29],
                            "CLICKS": row[30],
                            "PRED_SPEND": row[31],
                            "SPEND": row[32],
                            "ADWORD_LIST": []
                        }
                        adword_set_list.append(adword_set)    

                    adword_list = adword_set["ADWORD_LIST"]

                    adword_list.append({
                        "ID": row[0],
                        "ADWORD_TYPE": row[1],
                        "STATUS": row[19],
                        "NAME": row[8],
                        "DATETIME": row[20],
                        "PRED_IMPRESSIONS": row[11],
                        "IMPRESSIONS": row[12],
                        "PRED_CLICKS": row[13],
                        "CLICKS": row[14],
                        "PRED_SPEND": row[15],
                        "SPEND": row[16]
                    })

                cnt_query = text("EXEC SP_CAMPAIGN_LIST_COUNT :model_time, :date, :time, :advertiser_str, :adword_type_str, :description_str, :search_value")
                cnt_params = {
                    "model_time": int(model_time),
                    "date": add_date_str,
                    "time": time,
                    "advertiser_str": advertiser_str,
                    "adword_type_str": adword_type_str,
                    "description_str": description_str,
                    "search_value": search
                }                

                list_cnt = 0
                cnt_result = conn.execute(cnt_query, cnt_params).fetchone()

                if cnt_result:
                    list_cnt = int(cnt_result[0])

                data = {
                    "LIST": [
                        {
                            **campaign
                        }
                        for campaign in campaign_dict.values()
                    ],
                    "TOTAL_DATA": list_cnt,
                    "TOTAL_PAGE": math.ceil(list_cnt / page_size),
                    "PAGE_SIZE": page_size,
                    "PAGE_NUMBER": page_number,
                    "DATE" : date,
                    "TIME" : time,
                    "MODEL_TIME" : model_time                    
                }                                 
                return data

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}    

    finally:
        try:
            conn.close()
        except Exception:
            pass        

# 저장 프로시저 실행 함수
def execute_sp_prediction_model_time_procedure(adword_id, date, time, model_time):
    try:
        with engine.connect() as conn:
            # 광고 기본 정보를 가져오는 쿼리
            model_query = text("EXEC SP_PREDICTION_MODEL_TIME_INFO :adword_id, :date, :time")
            model_result = conn.execute(model_query, {"adword_id": adword_id, "date": date, "time": time}).fetchone()

            if not model_result:
                return None

            model_columns = [
                "ADWORD_TYPE", "CAMPAIGN_NAME", "ADWORD_SET_NAME", "ADWORD_NAME"
            ]
            model_data = dict(zip(model_columns, model_result))

            date_obj = datetime.strptime(date, '%Y-%m-%d')
            add_date = date_obj + timedelta(minutes=int(model_time))
            add_date_str = add_date.strftime('%Y-%m-%d')

            # 7일 예측 데이터를 가져오는 쿼리
            prediction_query = text("EXEC SP_PREDICTION_MODEL_TIME :adword_id, :model_time, :date, :time")
            prediction_results = conn.execute(prediction_query, {"adword_id": adword_id, "model_time": int(model_time), "date": add_date_str, "time": time}).fetchall()

            if not prediction_results:
                model_data["FORECAST_LIST"] = []
                return model_data

            prediction_columns = [
                "DATETIME", "PRED_IMPRESSIONS", "PRED_CLICKS", "PRED_SPEND",
                "IMPRESSIONS", "CLICKS", "SPEND"
            ]

            prediction_data = [
                dict(zip(prediction_columns, row)) for row in prediction_results
            ]            

            model_data["FORECAST_LIST"] = prediction_data

            return model_data

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass                      

# 저장 프로시저 실행 함수
def execute_sp_select_ver_procedure(date):
    try:
        with engine.connect() as conn:
            # 광고 기본 정보를 가져오는 쿼리
            model_query = text("EXEC SP_SELECT_VER :date")
            model_result = conn.execute(model_query, {"date": date}).fetchall()

            if not model_result:
                return None

            model_columns = [
                "TIME"
            ]

            model_data = [
                dict(zip(model_columns, row)) for row in model_result
            ]

            return model_data

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass                                  