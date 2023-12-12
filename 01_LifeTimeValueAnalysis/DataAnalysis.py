#! Import Library
import pandas as pd
import warnings
import os
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gamma
from scipy.stats import beta 

from datetime import datetime
from datetime import timedelta

from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import rand
from hyperopt import SparkTrials
from hyperopt import STATUS_OK
from hyperopt import space_eval
from hyperopt import Trials

# 
period_holdout = 90
period_predict = 365
verbose_flag = True
doodler_flag = False
savefile_flag = False

def score_model(actuals, predicted, metric = 'mse') :
    actuals = np.array(actuals) if type(actuals) == 'list' else actuals
    predicted = np.array(predicted) if type(predicted) == 'list' else predicted
   
    # MSE
    if metric.lower() == 'mse' or metric.lower() == 'rmse' :
        val = np.sum(np.square(actuals - predicted)) / actuals.shape[0]
    
    # RMSE
    elif metric == 'rmse' :
        val = np.sqrt(np.sum(np.square(actuals - predicted)) / actuals.shape[0])
        
    # MAE
    elif metric == 'mae' :
        val = np.sum(np.abs(actuals - predicted)) / actuals.shape[0]
        
    return val    

def evaluate_BG_NBD_model(inputs, param) :
    data        = inputs
    L2_penalty  = param
    
    # BG/NBD 모형 피팅
    model = BetaGeoFitter(penalizer_coef = L2_penalty)
    model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
    
    # 모형 평가
    frequency_actual = data['frequency_holdout']
    frequency_predicted = model.predict(data['duration_holdout'],
                                        data['frequency_cal'],
                                        data['recency_cal'],
                                        data['T_cal']
                                        )   
    mse = score_model(frequency_actual, frequency_predicted, metric = 'mse' )
    return {'loss' : mse, 'status' : STATUS_OK}

def find_L2penalty_BG_NBD_model(df):
    search_space = hp.uniform('l2', 0.0, 1.0)
    algo = tpe.suggest
    trials = Trials()
    inputs = df
    
    def tmp_evaluate_BG_NBD_model(param) :
        return evaluate_BG_NBD_model(inputs, param)
    
    argmin = fmin(
        fn = tmp_evaluate_BG_NBD_model,     # 목적 함수
        space = search_space,               # 파라미터 공간
        algo = algo,                        # 최적화 알고리즘 : Tree of Parzen Estimators (TPE)
        max_evals = 100,                    # iteration
        trials = trials
        )
    
    l2_bgnbd = space_eval(search_space, argmin)
    return l2_bgnbd
    
def evaluate_Gamma_Gamma_model(inputs, param) :
    data        = inputs
    L2_penalty  = param

    # GammaGamma 모형 피팅
    model = GammaGammaFitter(penalizer_coef = L2_penalty)
    model.fit(data['frequency_cal'], data['monetary_value_cal'])

    # 모형 평가
    monetary_actual = data['monetary_value_holdout']
    monetary_predicted = model.conditional_expected_average_profit(
        data['frequency_holdout'],
        data['monetary_value_holdout']
        )
    mse = score_model(monetary_actual, monetary_predicted)
    return {'loss' : mse, 'status' : STATUS_OK}


def find_L2penalty_Gamma_Gamma_model(df) :
    search_space = hp.uniform('l2', 0.0, 1.0)
    algo = tpe.suggest
    trials = Trials()
    inputs = df
    
    def tmp_evaluate_Gamma_Gamma_model(param) :
        return evaluate_Gamma_Gamma_model(inputs, param)
        
    # Gamma Gamma
    argmin = fmin(
        fn = tmp_evaluate_Gamma_Gamma_model,
        space = search_space,
        algo = algo,
        max_evals = 100,
        trials = trials
        )
    
    l2_gg = space_eval(search_space, argmin)
    return l2_gg

def calibrate_BG_NBD_model(df):
    L2_penalty = find_L2penalty_BG_NBD_model(df)
    
    model = BetaGeoFitter(penalizer_coef = L2_penalty)    
    
    # calibration 데이터의 R, F, T로 모형 피팅
    model.fit(df['frequency_cal'],
              df['recency_cal'],
              df['T_cal']
              )
    
    # holdout 데이터로 모델 평가 : F의 실제값과 예측값의 MSE
    frequency_actual = df['frequency_holdout']
    frequency_predicted = model.predict(df['duration_holdout'],
                                        df['frequency_cal'],
                                        df['recency_cal'],
                                        df['T_cal']
                                        )
    if doodler_flag == True :
        plot_calibration_accuracy(frequency_actual, frequency_predicted)

    if verbose_flag == True:
        print ('[Result][BG-NBD] Mean Square Error : %s' %score_model(frequency_actual, frequency_predicted, 'mse'))
        print ('[Result][BG-NBD] Coefficient-r : %s' %model.summary['coef']['r'])
        print ('[Result][BG-NBD] Coefficient-α : %s' %model.summary['coef']['alpha'])
        print ('[Result][BG-NBD] Coefficient-a : %s' %model.summary['coef']['a'])
        print ('[Result][BG-NBD] Coefficient-b : %s' %model.summary['coef']['b'])
    return model

def calibrate_Gamma_Gamma_model(df):
    #! Gamma-Gamma 모델 피팅
    L2_penalty = find_L2penalty_Gamma_Gamma_model(df)

    model = GammaGammaFitter(penalizer_coef = L2_penalty)
    model.fit(df['frequency_cal'], df['monetary_value_cal'])

    # conditional_expected_average_profit : 고객별 평균 구매 금액 예측
    monetary_actual = df['monetary_value_holdout']
    monetary_predicted = model.conditional_expected_average_profit(df['frequency_holdout'],
                                                                   df['monetary_value_holdout']
                                                                   )
    if doodler_flag == True :
        plot_calibration_accuracy(monetary_actual, monetary_predicted)

    if verbose_flag == True :
        print ('[Result][Gamma-Gamma] Mean Square Error : %s' %score_model(monetary_actual, monetary_predicted, 'mse'))
        print ('[Result][Gamma-Gamma] Coefficient-p : %s' %model.summary['coef']['p'])
        print ('[Result][Gamma-Gamma] Coefficient-q : %s' %model.summary['coef']['q'])
        print ('[Result][Gamma-Gamma] Coefficient-r : %s' %model.summary['coef']['v'])
    return model

def conduct_EDA(df) :
    # 성별에 따른 거래량과 매출 총합 계산
    if False :
        # 고객 성별에 따른 거래량 및 매출 총합 계산
        gender_sales = df.groupby('CustomerGender').agg({'CustomerID': 'count', 'Amount': 'sum'}).reset_index()

        if False :            
            # 첫 번째 그래프: 고객 성별 구성 비율
            plt.figure(figsize=(8, 8))
            plt.pie(gender_sales['CustomerID'], labels=gender_sales['CustomerGender'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
            plt.title('Customer Gender Composition Ratio', fontsize=18)
            plt.show()
        
        if False:
            # 두 번째 그래프: 고객 성별 매출 총합
            plt.figure(figsize=(8, 8))
            plt.pie(gender_sales['Amount'], labels=gender_sales['CustomerGender'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
            plt.title('Total Sales by Customer Gender', fontsize=18)
            plt.show()
            
        return None

    elif False :
        df['AgeGroup'] = pd.cut(df['CustomerAge'],
                                    bins = [10, 19, 29, 39, 49, 59],
                                    labels = ['10-19', '20-29', '30-39', '40-49', '50-59']
                                    )
        
        # 성별에 따른 거래량 계산
        gender_sales = df.groupby(['AgeGroup', 'CustomerGender']).size().unstack()

        # 그래프 그리기
        fig, ax1 = plt.subplots(figsize=(10, 6))

        gender_sales.plot(kind='bar', ax=ax1)
        ax1.set_title('Sales by Customer Age and Gende')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Total Amount')

        total_sales_by_age = df.groupby('AgeGroup')['Amount'].sum()
        ax2 = ax1.twinx()
        total_sales_by_age.plot(kind='line', marker='o', ax=ax2, linestyle='--', color='green', label='Total Amount')
        ax2.set_ylabel('Total Amount (Line)')

        plt.grid(True)
        plt.show()

        return None

    elif False :
        # InvoiceDate를 월별로 분류
        # df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: x.month)
        
        # 성별에 따른 거래량 계산
        gender_sales = df.groupby(['InvoiceMonth', 'CustomerGender']).agg({'Amount': 'sum'}).unstack()

        # 그래프 그리기
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 막대 그래프
        gender_sales['Amount'].plot(kind='bar', ax=ax1, color=['blue', 'pink'], label=['Male', 'Female'],
                                    position = -1)
        ax1.set_title('Sales by Invoice Month and Gender')
        ax1.set_xlabel('Invoice Month')
        ax1.set_ylabel('Total Amount')

        # 선 그래프
        ax2 = ax1.twinx()
        df.groupby('InvoiceMonth')['Amount'].sum().plot(kind='line', marker='o', ax=ax2, linestyle='--', color='green', label='Total Amount')
        ax2.set_ylabel('Total Amount (Line)')
        ax2.legend(loc='upper right')

        # X 축 레이블 설정
        ax1.set_xticks(sorted(df['InvoiceMonth'].unique()))  # 월 순서대로 정렬
        ax1.set_xticklabels([str(i) for i in sorted(df['InvoiceMonth'].unique())])

        plt.grid(True)
        plt.show()
        return None
    else :
        return None




def run_process():
    PATH = os.getcwd().split('\\')
    MAIN_PATH = r''
    for ii in range(len(PATH)) :
        MAIN_PATH += str(PATH[ii] + r'/') 
    
    df_customer = pd.read_csv(MAIN_PATH + r'data/Customer Info.csv')
    df_retailed = pd.read_csv(MAIN_PATH + r'data/Retailed Info.csv')


    ######## 데이터 전처리
    # df_retailed.CustomerID
    df_retailed['CustomerID'] = df_retailed['CustomerID'].dropna(axis = 0)
    
    # df_retailed.Quantity 
    df_retailed = df_retailed[df_retailed['Quantity'] > 0.]

    # df_retailed.InvoiceDate 
    df_retailed['InvoiceDate'] = pd.to_datetime(df_retailed['InvoiceDate']).dt.date
    
    # df_retailed.Amount
    df_retailed['Amount'] = df_retailed['Quantity'] * df_retailed['UnitPrice']

    # df_customer.CustomerID
    df_customer['CustomerID'] = df_customer['CustomerID'].dropna(axis = 0)

    # df_customer.CustomerGender
    df_customer = df_customer[(df_customer['CustomerGender'] == 'Female') | (df_customer['CustomerGender'] == 'Male')]

    # df_customer.CustomerAge
    df_customer = df_customer[(df_customer['CustomerAge'] >= 10) & (df_customer['CustomerAge'] < 80)]

    # merging two dataframe on customerID
    cols_of_feature = ['InvoiceDate', 'Amount', 'CustomerID', 'CustomerAge', 'CustomerGender']
    df = pd.merge(df_retailed, df_customer, on = ['CustomerID'])[cols_of_feature]


    # plot EDA graph
    df_EDA = df.copy()
    conduct_EDA(df_EDA)


    df_LTV = df.copy()
    cols_of_interest = ['CustomerID', 'InvoiceDate', 'Amount']
    df_LTV = df_LTV[cols_of_interest]

    # 집계일 정의
    date_aggregation = df_LTV['InvoiceDate'].max()    
    enddate_calibration = date_aggregation - timedelta(days = period_holdout)

    # 모수 추정 데이터셋
    df_calibration = calibration_and_holdout_data(df_LTV,
                                                customer_id_col = 'CustomerID',
                                                datetime_col = 'InvoiceDate',
                                                calibration_period_end = enddate_calibration,
                                                observation_period_end = date_aggregation,
                                                monetary_value_col = 'Amount'
                                                )
    df_calibration = df_calibration[df_calibration.frequency_cal > 0]

    BG_NBD_model = calibrate_BG_NBD_model(df_calibration)

    Gamma_Gamma_model = calibrate_Gamma_Gamma_model(df_calibration)

    # 모델 검증 데이터셋
    df_validation = summary_data_from_transaction_data(df_LTV,
                                            customer_id_col = 'CustomerID',
                                            datetime_col = 'InvoiceDate',
                                            monetary_value_col = 'Amount',
                                            observation_period_end = date_aggregation
                                            )
    
    df_validation = df_validation[df_validation.frequency > 0] 

    df_validation['life_time_value'] = Gamma_Gamma_model.customer_lifetime_value(BG_NBD_model,
                                                           df_validation['frequency'],
                                                           df_validation['recency'],
                                                           df_validation['T'],
                                                           df_validation['monetary_value'],
                                                           time = 12,
                                                           discount_rate = 0.01)

    df_validation['predicted_puchase_freqeuncy'] = BG_NBD_model.conditional_expected_number_of_purchases_up_to_time(period_predict,
                                                                                                         df_validation['frequency'],
                                                                                                         df_validation['recency'],
                                                                                                         df_validation['T']
                                                                                                         )
    
    df_validation['predicted_averaged_revenue'] = Gamma_Gamma_model.conditional_expected_average_profit(df_validation['frequency'],
                                                                                            df_validation['monetary_value']
                                                                                            )
    
    if verbose_flag == True :
        df_print = df_validation.sort_values(by='life_time_value', ascending=False)
        pd.set_option('display.max_rows', None)
        print (df_print)
        pd.reset_option('display.max_rows')
    
    if savefile_flag == True :
        df_validation.to_csv(MAIN_PATH + r'data/Result info.csv')
        

    return BG_NBD_model, Gamma_Gamma_model, df_validation
    



def plot_calibration_accuracy(data_actual, data_predicted):
    df = pd.DataFrame({'actual': data_actual, 'predicted': data_predicted}).reset_index()

    chunk_size = 170
    axis_x = range(chunk_size)
    width_bar = 0.6
    num_subplots = -(-len(df) // chunk_size)

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots), sharex=True)
    for idx in range(num_subplots) :
        df_subset = df.iloc[idx * chunk_size : (idx + 1) * chunk_size].reset_index()
        
        # 각 서브플롯에 대해 2개의 막대 그래프 그리기
        axs[idx].bar(df_subset.index - width_bar / 2,
                     df_subset['actual'],
                     width = width_bar,
                     label = 'actual',
                     color = 'red'
                     )
        axs[idx].bar(df_subset.index + width_bar / 2,
                     df_subset['predicted'],
                     width = width_bar,
                     label = 'predicted',
                     color = 'blue'
                     )

    plt.tight_layout()
    plt.show()
    return None


if __name__ == '__main__' :

    BG_NBD_model, Gamma_Gamma_model, df_validation = run_process()
