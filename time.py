from fastapi import FastAPI, HTTPException
import pandas as pd
import pymssql

app = FastAPI()

data = {
    'DATE': ['2024-09-08', '2025-10-10'], 
    'TIME': ['11:45:00', '14:00:00']       
}
df = pd.DataFrame(data)

@app.post("/predict")
def database():
    conn = None
    cursor = None
    
    try:
        conn = pymssql.connect(
            server='27.96.134.44:1433', 
            database='master', 
            user='ai_user', 
            password='mkt2024_!@',
            charset='EUC-KR'
        )
        cursor = conn.cursor()
        
        df['DATE'] = df['DATE'].apply(lambda x: x[2:4] + x[5:7] + x[8:10])
        df['TIME'] = df['TIME'].apply(lambda x: x.replace(':', '')[:4])
        
        for index, row in df.iterrows():
            date = row['DATE']
            time = row['TIME']
        
            ver_value = f"{date}-{time}"
        
            query = f'''
            INSERT INTO TB_PRED_VER VALUES('{ver_value}', 'E', GETDATE(), GETDATE(), '/HOME/UPLOAD/EXCEL/{date}/{ver_value}.csv')
            '''
            cursor.execute(query)
            
        conn.commit()
    
    except pymssql.DatabaseError as e:
        print("Database error occurred:", e)
    except pymssql.InterfaceError as e:
        print("Interface error occurred:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

