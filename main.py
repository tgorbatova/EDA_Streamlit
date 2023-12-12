import streamlit as st
from functools import reduce
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

bank = Image.open(r'bank-customers-and-staff-serving-clients-vector-33337933.jpg')



st.set_page_config(layout="wide")
# заголовок приложения
st.title('Uplift-моделирование отклика клиента на предложение банка')
left_co, cent_co,last_co = st.columns([1,3,1])
with cent_co:
    st.image(bank)


clients = pd.read_csv(r'data\D_clients.csv')
clients.rename(columns={'ID':'ID_CLIENT'}, inplace=True)
job = pd.read_csv(r'data\D_job.csv')
last_credit = pd.read_csv(r'data\D_last_credit.csv')

close_loan = pd.read_csv(r'data\D_close_loan.csv')
loan = pd.read_csv(r'data\D_loan.csv')

loan = loan.merge(close_loan, on='ID_LOAN', how='left')

data = clients.merge(loan.groupby('ID_CLIENT')['ID_LOAN'].agg(list), on='ID_CLIENT', how='left')
data = data.merge(loan.groupby('ID_CLIENT')['CLOSED_FL'].agg(list), on='ID_CLIENT', how='left')

salary = pd.read_csv(r'data\D_salary.csv')
target = pd.read_csv(r'data\D_target.csv')

dfs = [data, job, last_credit, salary, target]
data = reduce(lambda left,right: pd.merge(left,right,on='ID_CLIENT'), dfs)

data[['ID_LOAN', 'CLOSED_FL']] = data[['ID_LOAN', 'CLOSED_FL']].astype('str')

tooltips_df = {
    'CLIENT_ID': 'Идентификатор клиента',
    'AGE': 'Возраст клиента',
    'GENDER': 'Пол клиента (1 — мужчина, 0 — женщина)',
    'EDUCATION':'Уровень образования',
    'MARITAL_STATUS':'Семейное положение',
    'CHILD_TOTAL':'Количество детей клиента',
    'DEPENDANTS':'Количество иждивенцев клиента',
    'SOCSTATUS_WORK_FL':'Cоциальный статус клиента относительно работы (1 — работает, 0 — не работает)',
    'SOCSTATUS_PENS_FL':'Социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер)',
    'REG_ADDRESS_PROVINCE':'Область регистрации клиента',
    'FACT_ADDRESS_PROVINCE':'Область фактического пребывания клиента',
    'POSTAL_ADDRESS_PROVINCE':'Почтовый адрес области',
    'FL_PRESENCE_FL':'Наличие в собственности квартиры (1 — есть, 0 — нет)',
    'OWN_AUTO':'Количество автомобилей в собственности',
    'AGREEMENT_RK':'уникальный идентификатор объекта в выборке',
    'ID_CLIENT':'идентификатор клиента',
    'TARGET':'целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было)',
    'GEN_INDUSTRY':'отрасль работы клиента',
    'GEN_TITLE':'должность',
    'JOB_DIR':'направление деятельности внутри компании',
    'WORK_TIME':'время работы на текущем месте (в месяцах)',
    'FAMILY_INCOME':'семейный доход (несколько категорий)',
    'PERSONAL_INCOME':'личный доход клиента (в рублях)',
    'CREDIT':'сумма последнего кредита клиента (в рублях)',
    'TERM':'срок кредита',
    'FST_PAYMENT':'первоначальный взнос (в рублях)',
    'ID_LOAN':'идентификатор кредита',
    'ID_LOAN':'идентификатор кредита',
    'CLOSED_FL':'текущий статус кредита (1 — закрыт, 0 — не закрыт)'
}

# add file-uploader widget in sidebar
uploaded_data = st.sidebar.file_uploader("Choose a CSV file")


with st.sidebar:
    selected = option_menu(menu_title = "МЕНЮ",
                          options = ['Описание и анализ данных', 'Моделирование', 'Предсказание']
    )

if selected == 'Описание и анализ данных':
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)


    # add tabs
    tab1, tab2, tab3, tab4 = st.tabs(["General Data Info", "Numeric Features", "Categorical Features", 'Relationships'])
    with tab1:
        st.header('Data Example')

        # пример полного датасета
        st.table(data.head())

        # column types
        st.header("Columns Type")
 
        # get feature names
        columns = list(data.columns)
        
        # create dataframe
        column_info_table = pd.DataFrame({
            "column": columns,
            "data_type": data.dtypes.tolist()
        })
            
          
        # display pandas dataframe as a table
        st.table(column_info_table.T)


        # extract meta-data from the uploaded dataset
        st.header("Meta-data")
    
        row_count = data.shape[0]
    
        column_count = data.shape[1]
        
        # Use the duplicated() function to identify duplicate rows
        duplicates = data[data.duplicated()]
        duplicate_row_count =  duplicates.shape[0]
    
        missing_value_row_count = data[data.isna().any(axis=1)].shape[0]
    
        table_markdown = f"""
        | Description | Value | 
        |---|---|
        | Number of Rows | {row_count} |
        | Number of Columns | {column_count} |
        | Number of Duplicated Rows | {duplicate_row_count} |
        | Number of Rows with Missing Values | {missing_value_row_count} |
        """
    
        st.markdown(table_markdown)

        # pd.set_option('display.float_format', lambda x: '%.1f' % x)
        # desc = data[['AGE', 'WORK_TIME', 'CREDIT', 'TERM', 'FST_PAYMENT', 'PERSONAL_INCOME']].describe()
        # st.header('Data Description')
        # st.dataframe(desc)

    with tab2:
        # find numeric features  in the dataframe
        numeric_cols = ['AGE', 'WORK_TIME', 'CREDIT', 'TERM', 'FST_PAYMENT', 'PERSONAL_INCOME']
    
        # add selection-box widget
        selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)

        # create columns
        # cols = st.columns(2)

        col_info = {}
        col_info["Number of Unique Values"] = len(data[selected_num_col].unique())
        col_info["Number of Rows with Missing Values"] = data[selected_num_col].isnull().sum()
        col_info["Number of Rows with 0"] = data[selected_num_col].eq(0).sum()
        col_info["Number of Rows with Negative Values"] = data[selected_num_col].lt(0).sum()
        col_info["Average Value"] = data[selected_num_col].mean()
        col_info["Standard Deviation Value"] = data[selected_num_col].std()
        col_info["Minimum Value"] = data[selected_num_col].min()
        col_info["Maximum Value"] = data[selected_num_col].max()
        col_info["Median Value"] = data[selected_num_col].median()
        
        info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
        
        st.header('Data Decription')
        st.text(tooltips_df[selected_num_col])

        # display dataframe as a markdown table
        st.table(info_df)

        st.header(f"{selected_num_col} - Statistics")
     



        fig1 = px.histogram(data[selected_num_col], x=selected_num_col, opacity=0.5)
        fig1.update_layout(
            title=f'Распределение {selected_num_col}',
            yaxis_title_text='Количество клиентов',
            xaxis_title_text=f'{selected_num_col}',
            plot_bgcolor="white",
            bargap=0.3)

        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(data[selected_num_col],  x=selected_num_col)
        fig2.update_layout(
            title=f'Распределение {selected_num_col}',
            yaxis_title_text='Количество клиентов',
            xaxis_title_text=f'{selected_num_col}',
            plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

        fig = px.histogram(data, x=selected_num_col, color='TARGET')
        fig.update_layout(
            title=f'Распределение {selected_num_col} совместно с таргетом',
            yaxis_title_text='Количество клиентов',
            plot_bgcolor="white",
            bargap=0.3) 
        st.plotly_chart(fig, use_container_width=True)

        if selected_num_col == 'WORK_TIME':
            st.text('Наблюдается очень явный выброс, похожий на ошибку. Посмотрим без него:')

            data = data[~(data['ID_CLIENT'] == 106813354)]

            # cols = st.columns(2)
            fig2 = px.box(data['WORK_TIME'],  x="WORK_TIME")
            fig2.update_layout(
                title=f'Распределение {selected_num_col}',
                yaxis_title_text='Количество клиентов',
                xaxis_title_text=f'{selected_num_col}',
                plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)


            st.text('Все еще существуют выбросы, поэтому ограничим рабочий стаж 80 годами и снова посмотрим на распределение:')

            data = data.loc[(data['WORK_TIME'] <= 80)]

            fig2 = px.box(data['WORK_TIME'],  x="WORK_TIME")
            fig2.update_layout(
                title=f'Распределение {selected_num_col}',
                yaxis_title_text='Количество клиентов',
                xaxis_title_text=f'{selected_num_col}',
                plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

            fig = px.histogram(data, x=selected_num_col, color='TARGET')
            fig.update_layout(
                title=f'Распределение {selected_num_col} совместно с таргетом',
                yaxis_title_text='Количество клиентов',
                plot_bgcolor="white",
                bargap=0.3) 
            st.plotly_chart(fig, use_container_width=True)




    with tab3:
        cat_cols = ['GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'FL_PRESENCE_FL', 'OWN_AUTO', 'TARGET', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'FAMILY_INCOME']

         # add selection-box widget
        selected_cat_col = st.selectbox("Which categorical column do you want to explore?", cat_cols)

        # create columns
        cols = st.columns(2)

        cols[0].header('Data Description')
        cols[0].text(tooltips_df[selected_cat_col])


        # add categorical column stats
        cat_col_info = {}
        cat_col_info["Number of Unique Values"] = len(data[selected_cat_col].unique())
        cat_col_info["Number of Rows with Missing Values"] = data[selected_cat_col].isnull().sum()
        cat_col_info["Number of Empty Rows"] = data[selected_cat_col].eq("").sum()
        cat_col_info["Mode Value"] = data[selected_cat_col].mode()[0]

        cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Description', 'Value'])
        cols[0].table(cat_info_df)

        
        if selected_cat_col in ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR']:
            cols[0].header('10 Most Common')
            cols[0].table(pd.DataFrame(data[selected_cat_col].value_counts().index[:10], columns=[selected_cat_col]))

        cols[1].header('Unique Values')
        cols[1].text(' ')
        cols[1].text(' ')
        u = pd.DataFrame(data[selected_cat_col].astype(str).unique(), columns=['Value'])
        cols[1].table(u)

        fig1 = px.histogram(data[selected_cat_col], y=selected_cat_col, opacity=0.5)
        fig1.update_layout(
            title=f'Распределение {selected_cat_col}',
            yaxis_title_text='Количество клиентов',
            plot_bgcolor="white",
            bargap=0.3) 
        st.plotly_chart(fig1, use_container_width=True)

        fig = px.histogram(data, y=selected_cat_col, color='TARGET')
        fig.update_layout(
            title=f'Распределение {selected_cat_col} совместно с таргетом',
            yaxis_title_text='Количество клиентов',
            plot_bgcolor="white",
            bargap=0.3) 
        st.plotly_chart(fig, use_container_width=True)


    with tab4:
        # матрица корреляций 
        pio.templates.default = "plotly_white"
        corr = data.corr(numeric_only=True).round(1)
        mask = np.zeros_like(corr, dtype=bool)
        fig = go.Figure(data=go.Heatmap(
            z=corr.mask(mask),
            x=corr.columns,
            y=corr.columns,
            colorscale=px.colors.diverging.RdBu,
            reversescale=True,
            zmin=-1,
            zmax=1))
        
        fig.update_layout(width=500, height=1000)

        st.header('Correlation matrix')
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('''
                    Видим внутреннюю корреляцию между признаками, связанными с последним кредитом клиента (CREDIT, TERM, FST_PAYMENT) и PERSONAL_INCOME.
                    Корреляций с таргетом выше 0.1 нет.
                    При этом надо понимать, что корреляция Пирсона - не лучший инструмент для анализа зависимостей с бинарной переменной (в нашем случае трагет как раз бинарный).
                    ''')

