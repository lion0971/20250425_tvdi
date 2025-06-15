from flask import Flask,render_template,request
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




# 載入 .env 檔案
load_dotenv()
conn_string = os.getenv('RENDER_DATABASE')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html.jinja2")

@app.route("/classes",defaults={'course_types':'一般課程'})
@app.route("/classes/<course_types>")
def classes(course_types):
    print(course_types)
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        sql = """
        SELECT DISTINCT "課程類別" FROM "進修課程";
        """
        cur.execute(sql)
        temps = cur.fetchall()
        kinds = [kind[0] for kind in temps]
        kinds.reverse()

        sql_course = """
        SELECT
            "課程名稱",
            "群組",
            "進修人數",
            "進修時數",
            "進修費用",
            "上課時間",
            "課程開始日期"
        FROM
            "進修課程"
        WHERE
            "課程類別" = %s;
        """
        cur.execute(sql_course, (course_types,))
        course_data = cur.fetchall()
        page = request.args.get('page', 1, type=int)
        per_page = 6
        total = len(course_data)
        
        # 修正分頁總數計算，避免在項目剛好是 per_page 的倍數時產生多餘頁面
        total_pages = (total + per_page - 1) // per_page
        if total_pages == 0:
            total_pages = 1

        start = (page - 1) * per_page
        end = start + per_page
        items = course_data[start:end]  # 取得該頁資料

    # 將 current_kind 傳遞給模板，用於標記當前選中的課程類型
    return render_template("classes.html.jinja2",
                           kinds=kinds,
                           course_data=items,
                           page=page,
                           total_pages=total_pages,
                           current_kind=course_types)

@app.route("/news")
def news():
    try:
        conn = psycopg2.connect(conn_string)
        with conn.cursor() as cur:
            sql = """SELECT * FROM public.最新訊息
                     ORDER BY 上版日期 desc"""
            cur.execute(sql)
        # 取得所有資料
            rows = cur.fetchall()
            
        
    except OperationalError as e:
        print("連線失敗")
        print(e)
        return render_template("error.html.jinja2",error_message="資料庫錯誤"),500
    except:
        return render_template("error.html.jinja2",error_message="不知名錯誤"),500
    conn.close()
    return render_template("news.html.jinja2",rows=rows)

@app.route("/traffic")
def traffic():
    return render_template("traffic.html.jinja2")

@app.route("/contact")
def contact():
    return render_template("contact.html.jinja2")

@app.route('/matplotlib1')
def matplotlib1():
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        sql = '''
        SELECT 
            "每戶可支配所得", 
            "西元年", 
            "區別", 
            "結婚率"
        FROM "結婚率資料";
        '''
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    X = df[['每戶可支配所得', '西元年', '區別']]
    y = df['結婚率']

    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('onehot', onehotencoder, ['區別'])],
        remainder='passthrough'
    )

    model_pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = model_pipeline_rf.predict(X_test)

    # 圖表 1
    fig1 = px.scatter(x=y_test, y=y_pred_rf, labels={'x': '實際結婚率', 'y': '預測結婚率'}, title='實際值 vs 預測值')
    fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                              mode='lines', line=dict(dash='dash', color='red'), name='理想預測線'))


    from plotly.io import to_html
    fig1_html = to_html(fig1, full_html=False)

    return render_template('matplotlib1.html.jinja2',
                           fig1_html=fig1_html)

@app.route('/matplotlib2')
def matplotlib2():
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        sql = '''
        SELECT 
            "每戶可支配所得", 
            "西元年", 
            "區別", 
            "結婚率"
        FROM "結婚率資料";
        '''
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    X = df[['每戶可支配所得', '西元年', '區別']]
    y = df['結婚率']

    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('onehot', onehotencoder, ['區別'])],
        remainder='passthrough'
    )

    model_pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = model_pipeline_rf.predict(X_test)

 
    # 圖表 2
    residuals = y_test - y_pred_rf
    fig2 = px.scatter(x=y_pred_rf, y=residuals, labels={'x': '預測結婚率', 'y': '殘差'}, title='殘差圖')
    fig2.add_hline(y=0, line_dash='dash', line_color='red')


    from plotly.io import to_html

    fig2_html = to_html(fig2, full_html=False)


    return render_template('matplotlib2.html.jinja2',
                           fig2_html=fig2_html)

@app.route('/matplotlib3')
def matplotlib3():
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        sql = '''
        SELECT 
            "每戶可支配所得", 
            "西元年", 
            "區別", 
            "結婚率"
        FROM "結婚率資料";
        '''
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    X = df[['每戶可支配所得', '西元年', '區別']]
    y = df['結婚率']

    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('onehot', onehotencoder, ['區別'])],
        remainder='passthrough'
    )

    model_pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = model_pipeline_rf.predict(X_test)

    # 圖表 3
    rf_model = model_pipeline_rf.named_steps['regressor']
    fitted_encoder = model_pipeline_rf.named_steps['preprocessor'].named_transformers_['onehot']
    onehot_features = fitted_encoder.get_feature_names_out(['區別'])
    remaining_features = ['每戶可支配所得', '西元年']
    all_features = list(onehot_features) + remaining_features

    feature_importances = pd.Series(rf_model.feature_importances_, index=all_features).sort_values()
    fig3 = px.bar(feature_importances, orientation='h', labels={'value': '重要性分數', 'index': '特徵'}, title='特徵重要性')

    from plotly.io import to_html
    fig3_html = to_html(fig3, full_html=False)

    return render_template('matplotlib3.html.jinja2',
                           fig3_html=fig3_html)



@app.route('/table')
def table():

    try:
        # 連接 PostgreSQL 資料庫
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()

        # 查詢資料
        cursor.execute("""SELECT "西元年", "區別", "每戶可支配所得", "結婚率" FROM "結婚率資料";""")
        data = cursor.fetchall()
        conn.close()

        # 轉成 list of dicts 以利 Jinja2 模板使用
        data_dicts = [{"西元年": row[0], "區別": row[1], "每戶可支配所得": row[2], "結婚率": row[3]} for row in data]

        return render_template('table.html.jinja2', items=data_dicts)

    except OperationalError as e:
        return f"資料庫連線錯誤：{e}"
