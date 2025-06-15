
from flask import Flask,render_template

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError

# 載入 .env 檔案
load_dotenv()
conn_string = os.getenv('RENDER_DATABASE')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html.jinja2')

@app.route('/classes')
def classes():
    name= 'Mary'
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    return render_template('classes.html.jinja2',name = name,weekdays = weekdays)

@app.route('/news')
def news():
    try:
        conn = psycopg2.connect(conn_string)
        with conn.cursor() as cur:
            sql = "SELECT * FROM 最新訊息"
            cur.execute(sql)
            # 取得所有資料
            rows = cur.fetchall()
            print(rows)
            # raise Exception('出現錯誤')
    except OperationalError as e:
        print("連線失敗")
        print(e)
        return render_template('error.html.jinja2',error_message = "資料庫錯誤")
    except :
        return render_template('error.html.jinja2',error_message = "不知名錯誤")
        
    conn.close()
    return render_template('news.html.jinja2',rows = rows)

@app.route('/traffic')
def traffic():
    return render_template('traffic.html.jinja2')

@app.route('/contact')
def contact():
    return render_template('contact.html.jinja2')

@app.route('/product')
def product():
    return '<h1>product, World!</h1><p>這是我的第三頁</p>'

@app.route('/matplotlib')
def traffic():
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

    # 圖表 2
    residuals = y_test - y_pred_rf
    fig2 = px.scatter(x=y_pred_rf, y=residuals, labels={'x': '預測結婚率', 'y': '殘差'}, title='殘差圖')
    fig2.add_hline(y=0, line_dash='dash', line_color='red')

    # 圖表 3
    rf_model = model_pipeline_rf.named_steps['regressor']
    fitted_encoder = model_pipeline_rf.named_steps['preprocessor'].named_transformers_['onehot']
    onehot_features = fitted_encoder.get_feature_names_out(['區別'])
    remaining_features = ['每戶可支配所得', '西元年']
    all_features = list(onehot_features) + remaining_features

    feature_importances = pd.Series(rf_model.feature_importances_, index=all_features).sort_values()
    fig3 = px.bar(feature_importances, orientation='h', labels={'value': '重要性分數', 'index': '特徵'}, title='特徵重要性')

    from plotly.io import to_html
    fig1_html = to_html(fig1, full_html=False)
    fig2_html = to_html(fig2, full_html=False)
    fig3_html = to_html(fig3, full_html=False)

    return render_template('matplotlib.html.jinja2',
                           fig1_html=fig1_html,
                           fig2_html=fig2_html,
                           fig3_html=fig3_html)
