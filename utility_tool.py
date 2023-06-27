import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt    
import math
import datetime
from scipy.stats import norm
from scipy.optimize import brentq



# column_list = ['Timestamp', 'BidPrice0', 'BidPrice1', 'BidPrice2', 'BidPrice3',
#        'BidPrice4', 'AskPrice0', 'AskPrice1', 'AskPrice2', 'AskPrice3',
#        'AskPrice4']

def preposcessing(path_stock,path_CB):
    stock_df = pd.read_csv(path_stock)
    cd_df = pd.read_csv(path_CB)
    df = pd.merge(stock_df,cd_df,on='Timestamp',how='outer')
    df.rename(columns={'BidPrice0_x':'BidPrice0_S', 'BidPrice1_x':'BidPrice1_S', 'BidPrice2_x':'BidPrice2_S', 'BidPrice3_x':'BidPrice3_S',
        'BidPrice4_x':'BidPrice4_S', 'AskPrice0_x':'AskPrice0_S', 'AskPrice1_x':'AskPrice1_S', 'AskPrice2_x':'AskPrice2_S',
        'AskPrice3_x':'AskPrice3_S', 'AskPrice4_x':'AskPrice4_S', 'BidPrice0_y':'BidPrice0_CB', 'BidPrice1_y':'BidPrice1_CB',
        'BidPrice2_y':'BidPrice2_CB', 'BidPrice3_y':'BidPrice3_CB', 'BidPrice4_y':'BidPrice4_CB', 'AskPrice0_y':'AskPrice0_CB',
        'AskPrice1_y':'AskPrice1_CB', 'AskPrice2_y':'AskPrice2_CB', 'AskPrice3_y':'AskPrice3_CB', 'AskPrice4_y':'AskPrice4_CB'},inplace=True)

    #將nan 以字串'N'填補
    df.fillna('N',inplace=True)
    
    start_value = len(stock_df)   #6317
    end_value = len(df)-1 #6406
    #將所有資料已系統時間排序
    df_sorted = df.sort_values('Timestamp',ascending=True)[5:]
    
    #將原index作為標記
    df_sorted.reset_index(inplace=True)
    df_sorted.rename(columns={'index':'mark'},inplace=True)
   
    #利用原index取可轉債在資料中的位置list
    marker = df_sorted[(df_sorted['mark']>=start_value) & (df_sorted['mark']<=end_value)].index.to_list()

    #利用marker將股價五檔資料賦與到可轉債五檔資料的時間上
    start_value = len(stock_df)   #6317
    end_value = len(df)-1 #6406
    for i, pos in enumerate(marker):
        value = start_value + i
        df_sorted.loc[pos, 'new_marker'] = value
    df_sorted['new_marker'] = df_sorted['new_marker'].fillna(method='backfill') #向透填補缺失值賦予分類
    data_df = df_sorted.drop(columns = 'mark')

    #get stock data covert to specific cb category
    temp = data_df[data_df['BidPrice0_S']!='N'].iloc[:,:11].copy()
    temp['new_marker'] = data_df['new_marker']
    
    #change dtype to float
    for i in temp.columns:
        temp[i] = temp[i].astype('float')
   
    #groupby data to specific category use mean
    temp = temp.groupby('new_marker').mean().drop(columns='Timestamp')
    
    #CB data
    cb_data = data_df.iloc[marker,:]
    cb_data.drop(columns=['BidPrice0_S', 'BidPrice1_S', 'BidPrice2_S', 'BidPrice3_S','BidPrice4_S', 'AskPrice0_S', 'AskPrice1_S', 'AskPrice2_S','AskPrice3_S', 'AskPrice4_S'],inplace=True)
    
    #conbine group and cd
    data_trans = pd.merge(cb_data,temp,on='new_marker',how='left')
    data_trans.dropna(axis=0,inplace=True)
    # data_trans = data_trans[15:]
    return data_trans


# 計算可轉債選擇權價格的函式
def calculate_option_price(volatility, ul_stock_price):
    # 計算時間間隔
    issued_date_obj = datetime.datetime.strptime(issued_date, '%Y%m%d')
    putable_date_obj = datetime.datetime.strptime(putable_date, '%Y%m%d')
    time_to_maturity = (putable_date_obj - issued_date_obj).days / 365.0
    
    # 計算d1和d2
    d1 = (math.log(ul_stock_price / conversion_price) + (corporate_bond_interest_rate + (volatility**2) / 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # 使用Black-Scholes公式計算選擇權價格
    call_price = ul_stock_price * norm.cdf(d1) - conversion_price * math.exp(-corporate_bond_interest_rate * time_to_maturity) * norm.cdf(d2)
    put_price = conversion_price * math.exp(-corporate_bond_interest_rate * time_to_maturity) * norm.cdf(-d2) - ul_stock_price * norm.cdf(-d1)

    return max(call_price, put_price)


day_list = ['02','03','04','05','08','09','10','11','12','15','16','17',
            '18','19','22','23','24','25','26','29','30','31']

# 可轉債資訊

putable_date = '20280329'  # 到期日期
conversion_price = 295.0  # 轉換價格
corporate_bond_interest_rate = 0.0325  # 債券收益率


# 每日平均隱含波動度
daily_avg_ask = []
daily_avg_bid = []

for w in day_list:
    print(f'loop:{w}\n')
    issued_date = f'202305{w}'  # 發行日期
    path_1 = f'./Cathay_CB_27271_Project/Cathay_CB_27271_Project/20230501_20230531_2727/202305{w}_2727.csv'
    path_2 = f'./Cathay_CB_27271_Project/Cathay_CB_27271_Project/20230501_20230531_27271/202305{w}_27271.csv'

    data_trans = preposcessing(path_1,path_2)
    
    IV_ask = []
    IV_bid = []
    for i, j in zip(data_trans['AskPrice2_S'].values,data_trans['AskPrice2_CB'].values):
        # 使用二分法計算賣出價格的隱含波動度
        implied_volatility_ask = brentq(lambda x: calculate_option_price(x, i) - j, -2.0,1.0)
        IV_ask.append(implied_volatility_ask)
        print("Implied Volatility (Ask):", implied_volatility_ask)

    for i, j in zip(data_trans['BidPrice2_S'].values,data_trans['BidPrice2_CB'].values):
        # 使用二分法計算賣出價格的隱含波動度
        print('\n')
        implied_volatility_bid = brentq(lambda x: calculate_option_price(x, i) - j, -2.0,1.0)
        IV_bid.append(implied_volatility_bid)
        print("Implied Volatility (Bid):", implied_volatility_bid)
        
    data_trans['IV_ask'] = IV_ask
    data_trans['IV_bid'] = IV_bid
    ask_avg = data_trans['IV_ask'].mean(axis=0)
    bid_avg = data_trans['IV_bid'].mean(axis=0)
    
    daily_avg_ask.append(ask_avg)
    daily_avg_bid.append(bid_avg)

    # data plot
    plt.figure(figsize=(16,6))
    data_trans['IV_ask'].plot(label = f'IV_ask_05/{w}')
    data_trans['IV_bid'].plot(label = f'IV_bid_05/{w}')
    
    plt.axhline(ask_avg, color='b', linestyle='--', label=f'ask_avg_05/{w}')
    plt.axhline(bid_avg, color='r', linestyle='--', label=f'bid_avg_05/{w}')
    plt.text(len(data_trans), ask_avg+0.015, f'Average_Ask: {ask_avg:.2f}', 
         color='black', ha='right', va='center',fontweight='bold', fontsize=12)
    plt.text(len(data_trans), bid_avg-0.015, f'Average_Bid: {bid_avg:.2f}', 
         color='black', ha='right', va='center',fontweight='bold', fontsize=12)
    plt.title(f'05/{w}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./IV_plot/05_{w}')
    
    #save intraday iv data
    data_trans[['Timestamp','IV_ask','IV_bid']].to_csv(f'./IV_data/05-{w}.csv')



#average calculate
average_ask_May = sum(daily_avg_ask)/len(daily_avg_ask)
average_bid_May = sum(daily_avg_bid)/len(daily_avg_bid)

# daily data plot
plt.figure(figsize=(16,6))
plt.plot(day_list,daily_avg_ask,label='daily_avg_ask')
plt.plot(day_list,daily_avg_bid,label='daily_avg_bid')

plt.axhline(average_ask_May, color='b', linestyle='--', label='average_ask_May')
plt.axhline(average_bid_May, color='r', linestyle='--', label='average_bid_May')
plt.text(len(day_list), average_ask_May+0.015, f'Average_Ask: {average_ask_May:.2f}', 
         color='black', ha='right', va='center',fontweight='bold', fontsize=12)
plt.text(len(day_list), average_bid_May-0.015, f'Average_Bid: {average_bid_May:.2f}', 
         color='black', ha='right', va='center',fontweight='bold', fontsize=12)
plt.title('May')
plt.grid(True)
plt.legend()
plt.savefig(f'./IV_plot/May_daily')

daily = pd.DataFrame()
daily['day'] = day_list
daily['daily_avg_ask'] = daily_avg_ask
daily['daily_avg_bid'] = daily_avg_bid

daily.to_csv('./IV_data/May.csv')   