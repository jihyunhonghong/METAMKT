import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote
import numpy as np
import pickle
from datetime import datetime, timedelta
import os

# SQLAlchemy 엔진 생성
user = "ai_user"
password = "mkt2024_!@#"
host = "27.96.134.44"
port = 1433
database = "master"

# 비밀번호 URL 인코딩
encoded_password = quote(password)

connection_url = f"mssql+pyodbc://{user}:{encoded_password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_url)

def log_to_db(engine, step, status, message):
    """
    Logs a message to the TB_LOG table in the database.
   
    Args:
        engine: SQLAlchemy engine connected to the database.
        step: The step or function name.
        status: The status of the operation (e.g., 'SUCCESS', 'FAILED').
        message: A descriptive message for the log.
    """
    try:
        with engine.connect() as conn:
            log_query = text('''
            INSERT INTO TB_LOG (STEP, STATUS, MESSAGE, LOG_TIME)
            VALUES (:step, :status, :message, GETDATE())
            ''')
            conn.execute(log_query, {"step": step, "status": status, "message": message})
            conn.commit()  # 명시적으로 트랜잭션 커밋
            print(f"[TB_LOG] Log entry added successfully: Step={step}, Status={status}, Message={message}")
    except Exception as e:
        # 로그 실패 시 콘솔에 출력
        print(f"[TB_LOG] Error logging to TB_LOG: {str(e)}\nStep={step}, Status={status}, Message={message}")

def preprocess_data(df):
    df = df.sort_values(by=['ADWORD_ID']).reset_index(drop=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['TIME'] = df['TIME'].astype(str).str.slice(0, 5)
    df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M').dt.time
    df['TIME_MINUTES'] = df['TIME'].apply(lambda x: x.hour * 60 + x.minute)
    df['YEAR'] = df['DATE'].dt.year
    df['DAY'] = df['DATE'].dt.day
    df['MONTH'] = df['DATE'].dt.month
    df['WEEK'] = df['DATE'].dt.dayofweek
    df['ADVERTISER'] = df['ADVERTISER'].astype('category')
    df['ADWORD_ID'] = df['ADWORD_ID'].astype('category')
    return df

def load_models(model_files):
    models = []
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            models.append(pickle.load(f))
    return models

def create_version():
    try:
        with engine.connect() as conn:
            log_to_db(engine, "create_version", "START", "Version creation process started")


            while True:
                # step1: VERSION_YN이 'N'인 레코드 중 가장 오래된 한 개만 조회
                proc_query = '''
                SELECT TOP 1 *
                FROM TB_ADWORD_ACCUM_PROC WITH (NOLOCK)
                WHERE VERSION_YN = 'N'
                ORDER BY DATE DESC, TIME DESC
                '''
                proc_df = pd.read_sql_query(proc_query, conn)


                if len(proc_df) == 0:
                    print("No new versions to process.")
                    log_to_db(engine, "step1", "SUCCESS", "No new versions to process.")
                    break  # 더 이상 처리할 버전이 없으므로 종료


                row = proc_df.iloc[0]
                log_to_db(engine, "step1", "SUCCESS", f"Selected record: VER = {row['TIME']}")


                # step2: 버전 생성 및 TB_PRED_VER 테이블에 인서트
                date = pd.to_datetime(row['DATE'], format='%Y-%m-%d')
                time_str = row['TIME']
                try:
                    time_ = pd.to_datetime(time_str, format='%H:%M:%S').time()
                except ValueError:
                    time_ = pd.to_datetime(time_str, format='%H:%M').time()


                date_str = date.strftime('%y%m%d')
                time_str_format = time_.strftime('%H:%M')
                ver_value = f"{date_str}-{time_str_format.replace(':', '')}"


                # 중복 삽입 방지
                check_query = text('''
                SELECT COUNT(1)
                FROM TB_PRED_VER
                WHERE VER = :ver
                ''')
                res = conn.execute(check_query, {"ver": ver_value}).fetchone()
                if res[0] > 0:
                    print(f"Version {ver_value} already exists. Skipping...")
                    log_to_db(engine, "step2", "WARNING", f"Version {ver_value} already exists. Skipping...")
                    continue


                ver_insert_query = text('''
                INSERT INTO TB_PRED_VER (VER, DATE, TIME, STATUS, INSERT_DATE, UPDATE_DATE, EXCEL_FILE_LOC)
                VALUES (:ver, :date_str, :time_str_format, 'R', GETDATE(), GETDATE(), :excel_loc)
                ''')
                conn.execute(ver_insert_query, {
                    "ver": ver_value,
                    "date_str": date_str,
                    "time_str_format": time_str_format,
                    "excel_loc": f"/HOME/UPLOAD/EXCEL/{date_str}/{ver_value}.csv"
                })
                conn.commit()
                print(f"step2 - TB_PRED_VER, Inserted version: {ver_value}")
                log_to_db(engine, "step2", "SUCCESS", f"Inserted version: {ver_value}")


                # step3: 작업현황 테이블 상태값 'Y'로 업데이트
                proc_update_query = text('''
                UPDATE TB_ADWORD_ACCUM_PROC
                SET VERSION_YN = 'Y'
                WHERE DATE = :proc_date
                  AND TIME = :proc_time
                ''')
                conn.execute(proc_update_query, {
                    "proc_date": row['DATE'],
                    "proc_time": row['TIME']
                })
                conn.commit()
                print(f"step3 - Updated record: Date {row['DATE']}, Time {row['TIME']}")
                log_to_db(engine, "step3", "SUCCESS", f"Updated record: Date {row['DATE']}, Time {row['TIME']}")


                # step4: Accumulated Data 가져오기
                accum_query = f'''
                SELECT A.*
                FROM TB_AI_ADWORD_ACCUM A WITH (NOLOCK)
                JOIN TB_PRED_VER V WITH (NOLOCK)
                  ON A.DATE = V.DATE AND A.TIME = V.TIME
                WHERE V.STATUS = 'R'
                  AND V.VER = '{ver_value}'
                '''
                df = pd.read_sql_query(accum_query, conn)
                print(f"step4 - Accumulated Data: {len(df)} rows")
                log_to_db(engine, "step4", "SUCCESS", f"Accumulated Data: {len(df)} rows")


                # step5: 데이터 전처리
                df = preprocess_data(df)
                print(f"step5 - Preprocessed Data: {len(df)} rows")
                log_to_db(engine, "step5", "SUCCESS", f"Preprocessed Data: {len(df)} rows")


                # step6: Feature 준비
                if 'ADWORD_TYPE' not in df.columns:
                    print("ADWORD_TYPE column not found in data.")
                    log_to_db(engine, "step6", "ERROR", "ADWORD_TYPE column not found in data.")
                    continue

                adword_type = df['ADWORD_TYPE'].iloc[0]

                X_imp = df[['ADWORD_ID', 'ACT_DAYS', 'TIME_MINUTES', 'MONTH', 'WEEK',
                            'ADVERTISER', 'CLICKS', 'SPEND', 'DB_COUNT', 'REVENUE', 'SALES']]

                X_cli = df[['ADWORD_ID', 'ACT_DAYS', 'TIME_MINUTES', 'MONTH', 'WEEK',
                            'ADVERTISER', 'IMPRESSIONS', 'SPEND', 'DB_COUNT', 'REVENUE', 'SALES']]

                X_spe = df[['ADWORD_ID', 'ACT_DAYS', 'TIME_MINUTES', 'MONTH', 'WEEK',
                            'ADVERTISER', 'IMPRESSIONS', 'CLICKS', 'DB_COUNT', 'REVENUE', 'SALES']]

                print(f"step6 - Features prepared")
                log_to_db(engine, "step6", "SUCCESS", "Features prepared")

                MODEL_TIMES = [1440, 2880, 4320, 5760, 7200, 8640, 10080]

                # 모델 로드 및 예측 수행
                imp_model_files = [f"250102_IMP_{mt}M_VER1.pkl" for mt in MODEL_TIMES]
                cli_model_files = [f"250102_CLK_{mt}M_VER1.pkl" for mt in MODEL_TIMES]
                spe_model_files = [f"250102_SPD_{mt}M_VER1.pkl" for mt in MODEL_TIMES]

                imp_models = load_models(imp_model_files)
                cli_models = load_models(cli_model_files)
                spe_models = load_models(spe_model_files)

                np.set_printoptions(precision=2, suppress=True)

                result_data = []
                for idx, model_time in enumerate(MODEL_TIMES):
                    y_imp_pred = np.round(imp_models[idx].predict(X_imp), 2)
                    y_cli_pred = np.round(cli_models[idx].predict(X_cli), 2)
                    y_spe_pred = np.round(spe_models[idx].predict(X_spe), 2)

                    print(f"step7 - Model Time {model_time}")
                    log_to_db(engine, f"step7_{model_time}", "SUCCESS", f"Model Time {model_time}")

                    for adword_idx, row_ in df.iterrows():
                        pred_datetime = datetime.strptime(
                            f"{row_['DATE'].strftime('%Y-%m-%d')} {row_['TIME'].strftime('%H:%M')}",
                            "%Y-%m-%d %H:%M"
                        )
                        pred_datetime += timedelta(minutes=model_time)
                        pred_date = pred_datetime.strftime("%Y-%m-%d")
                        pred_time = pred_datetime.strftime("%H:%M")

                        result_data.append({
                            'VER': f"{row_['DATE'].strftime('%y%m%d')}-{row_['TIME'].strftime('%H%M')}",
                            'ADWORD_ID': row_['ADWORD_ID'],
                            'MODEL_TIME': model_time,
                            'PRED_IMPRESSIONS': y_imp_pred[adword_idx],
                            'PRED_CLICKS': y_cli_pred[adword_idx],
                            'PRED_SPEND': y_spe_pred[adword_idx],
                            'PRED_DATE': pred_date,
                            'PRED_TIME': pred_time
                        })

                result_df = pd.DataFrame(result_data)
                print(f"step8 - Final Prediction DataFrame {len(result_df)} rows")
                log_to_db(engine, "step8", "SUCCESS", f"Final Prediction DataFrame {len(result_df)} rows")

                # step9: 결과 데이터베이스에 삽입
                insert_count = 0
                for _, row_ in result_df.iterrows():
                    check_query = text('''
                    SELECT COUNT(1)
                    FROM TB_PRED_RESULT
                    WHERE VER = :ver
                      AND ADWORD_ID = :adword_id
                      AND MODEL_TIME = :model_time
                    ''')
                    res = conn.execute(check_query, {
                        "ver": row_['VER'],
                        "adword_id": row_['ADWORD_ID'],
                        "model_time": row_['MODEL_TIME']
                    }).fetchone()


                    if res[0] == 0:
                        result_insert_query = text('''
                        INSERT INTO TB_PRED_RESULT (VER, ADWORD_ID, MODEL_TIME, PRED_IMPRESSIONS, PRED_CLICKS, PRED_SPEND, PRED_DATE, PRED_TIME, INSERT_DATE)
                        VALUES (:ver, :adword_id, :model_time, :pred_impressions, :pred_clicks, :pred_spend, :pred_date, :pred_time, GETDATE())
                        ''')
                        conn.execute(result_insert_query, {
                            "ver": row_['VER'],
                            "adword_id": row_['ADWORD_ID'],
                            "model_time": row_['MODEL_TIME'],
                            "pred_impressions": row_['PRED_IMPRESSIONS'],
                            "pred_clicks": row_['PRED_CLICKS'],
                            "pred_spend": row_['PRED_SPEND'],
                            "pred_date": row_['PRED_DATE'],
                            "pred_time": row_['PRED_TIME']
                        })
                        insert_count += 1

                conn.commit()
                print(f"step9 - Inserted records: {insert_count}")
                log_to_db(engine, "step9", "SUCCESS", f"Inserted records: {insert_count}")

                # step10: TB_PRED_VER 상태 업데이트
                ver_update_query = text('''
                UPDATE TB_PRED_VER
                SET STATUS = 'E'
                WHERE VER = :ver_value
                ''')
                conn.execute(ver_update_query, {"ver_value": ver_value})
                conn.commit()
                print(f"step10 - TB_PRED_VER STATUS updated for {ver_value}")
                log_to_db(engine, "step10", "SUCCESS", f"TB_PRED_VER STATUS updated for {ver_value}")

    except Exception as e:
        print("Database error occurred:", e)
        log_to_db(engine, "error", "FAILED", str(e))

    finally:
        try:
            conn.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing connection: {e}")