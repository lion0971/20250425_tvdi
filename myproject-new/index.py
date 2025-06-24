from flask import Flask, render_template, request, redirect, url_for
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
from plotly.io import to_html

load_dotenv()
conn_string = os.getenv('RENDER_DATABASE')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html.jinja2")

@app.route("/classes", defaults={'course_types': '一般課程'})
@app.route("/classes/<course_types>")
def classes(course_types):
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT \"課程類別\" FROM \"進修課程\";")
        kinds = [kind[0] for kind in cur.fetchall()]
        kinds.reverse()

        cur.execute("""
        SELECT "課程名稱", "群組", "進修人數", "進修時數", "進修費用", "上課時間", "課程開始日期"
        FROM "進修課程"
        WHERE "課程類別" = %s;
        """, (course_types,))
        course_data = cur.fetchall()

    page = request.args.get('page', 1, type=int)
    per_page = 6
    total = len(course_data)
    total_pages = max((total + per_page - 1) // per_page, 1)
    start = (page - 1) * per_page
    end = start + per_page
    items = course_data[start:end]

    return render_template("classes.html.jinja2", kinds=kinds, course_data=items, page=page, total_pages=total_pages, current_kind=course_types)

@app.route("/news")
def news():
    try:
        conn = psycopg2.connect(conn_string)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM public.最新訊息 ORDER BY 上版日期 desc")
            rows = cur.fetchall()
    except OperationalError as e:
        return render_template("error.html.jinja2", error_message="資料庫錯誤"), 500
    except:
        return render_template("error.html.jinja2", error_message="不知名錯誤"), 500
    finally:
        conn.close()

    return render_template("news.html.jinja2", rows=rows)

@app.route("/traffic")
def traffic():
    return render_template("traffic.html.jinja2")

@app.route("/contact")
def contact():
    return render_template("contact.html.jinja2")

@app.route("/matplotlib")
def matplotlib():
    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        cur.execute("""
        SELECT "每戶可支配所得", "西元年", "區別", "結婚率"
        FROM "結婚率資料";
        """)
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    X = df[['每戶可支配所得', '西元年', '區別']]
    y = df['結婚率']

    model_pipeline_rf = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['區別'])
        ], remainder='passthrough')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = model_pipeline_rf.predict(X_test)

    fig1 = px.scatter(x=y_test, y=y_pred_rf, labels={'x': '實際結婚率', 'y': '預測結婚率'}, title='實際值 vs 預測值')
    fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(dash='dash', color='red'), name='理想預測線'))
    fig1_html = to_html(fig1, full_html=False)

    residuals = y_test - y_pred_rf
    fig2 = px.scatter(x=y_pred_rf, y=residuals, labels={'x': '預測結婚率', 'y': '殘差'}, title='殘差圖')
    fig2.add_hline(y=0, line_dash='dash', line_color='red')
    fig2_html = to_html(fig2, full_html=False)

    rf_model = model_pipeline_rf.named_steps['regressor']
    onehot = model_pipeline_rf.named_steps['preprocessor'].named_transformers_['onehot']
    feature_names = list(onehot.get_feature_names_out(['區別'])) + ['每戶可支配所得', '西元年']
    importance = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    fig3 = px.bar(importance, orientation='h', labels={'value': '重要性分數', 'index': '特徵'}, title='特徵重要性')
    fig3_html = to_html(fig3, full_html=False)

    metrics = {
        'r2_rf': r2_score(y_test, y_pred_rf),
        'mse_rf': mean_squared_error(y_test, y_pred_rf),
        'rmse_rf': mean_squared_error(y_test, y_pred_rf) ** 0.5,
        'mae_rf': mean_absolute_error(y_test, y_pred_rf)
    }

    return render_template('matplotlib.html.jinja2', **metrics, fig1_html=fig1_html, fig2_html=fig2_html, fig3_html=fig3_html)

@app.route("/table")
def table():
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT \"西元年\", \"區別\", \"每戶可支配所得\", \"結婚率\" FROM \"結婚率資料\";")
        data = cursor.fetchall()
    except OperationalError as e:
        return f"資料庫連線錯誤：{e}"
    finally:
        conn.close()

    data_dicts = [
        {"西元年": row[0], "區別": row[1], "每戶可支配所得": row[2], "結婚率": row[3]}
        for row in data
    ]
    return render_template('table.html.jinja2', items=data_dicts)

@app.route('/predict_marriage', methods=['GET', 'POST'])
def predict_marriage():
    prediction = None
    error_msg = None

    conn = psycopg2.connect(conn_string)
    with conn.cursor() as cur:
        cur.execute('''SELECT "每戶可支配所得", "西元年", "區別", "結婚率" FROM "結婚率資料"''')
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    conn.close()

    X = df[['每戶可支配所得', '西元年', '區別']]
    y = df['結婚率']

    model_pipeline_rf = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['區別'])
        ], remainder='passthrough')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model_pipeline_rf.fit(X, y)

    if request.method == 'POST':
        year = request.form.get('year')
        income = request.form.get('income')
        region = request.form.get('region')

        try:
            if not year or not income or not region:
                raise ValueError("所有欄位皆為必填")

            year = int(year)
            income = float(income)
            if year < 2001 or year > 2023:
                raise ValueError("請輸入 2001 到 2023 年之間的年份")
            if income < 900000 or income > 1800000:
                raise ValueError("每戶可支配所得必須介於 900,000 到 1,800,000 之間")

            input_df = pd.DataFrame([{
                '每戶可支配所得': income,
                '西元年': year,
                '區別': region
            }])

            prediction = round(model_pipeline_rf.predict(input_df)[0], 2)

            return redirect(url_for('predict_marriage', prediction=prediction, year=year, income=income, region=region))

        except Exception as e:
            return redirect(url_for('predict_marriage', error_msg=str(e), year=year, income=income, region=region))

    prediction = request.args.get('prediction') if 'error_msg' not in request.args else None
    error_msg = request.args.get('error_msg')
    year = request.args.get('year', '')
    income = request.args.get('income', '')
    region_selected = request.args.get('region', '')

    regions = sorted(df['區別'].unique())

    return render_template('predict_marriage.html.jinja2', prediction=prediction, error_msg=error_msg, year=year, income=income, regions=regions, region_selected=region_selected)

if __name__ == '__main__':
    app.run(debug=True)
