#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlencode

# импорт библиотек


# In[2]:


# используем api 
base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?' 
public_key_customers = 'https://disk.yandex.ru/d/4davXCKRbz9qsA'  
public_key_orders = 'https://disk.yandex.ru/d/Ez_qSHCv7bjujw'
public_key_items = 'https://disk.yandex.ru/d/kLu5fohbKEbIVw'
    
    
# получаем url 
final_url_customers = base_url + urlencode(dict(public_key=public_key_customers)) 
final_url_orders = base_url + urlencode(dict(public_key=public_key_orders)) 
final_url_items = base_url + urlencode(dict(public_key=public_key_items)) 

response_customers = requests.get(final_url_customers) 
response_orders = requests.get(final_url_orders) 
response_items = requests.get(final_url_items)

download_url_customers = response_customers.json()['href'] 
download_url_orders = response_orders.json()['href'] 
download_url_items = response_items.json()['href'] 
        
# загружаем файлы  
download_response_customers = requests.get(download_url_customers) 
download_response_orders = requests.get(download_url_orders) 
download_response_items = requests.get(download_url_items) 

customers = pd.read_csv(download_url_customers) 
orders = pd.read_csv(download_url_orders,parse_dates=['order_delivered_carrier_date','order_delivered_customer_date',    'order_estimated_delivery_date','order_purchase_timestamp','order_approved_at']) 
order_items = pd.read_csv(download_url_items,parse_dates=['shipping_limit_date']) 


# # Небольшой разведывательный анализ

# In[3]:


customers.describe()


# In[4]:


customers.head()


# In[5]:


customers.dtypes


# In[6]:


order_items.describe()


# In[7]:


order_items.head()


# In[8]:


order_items.dtypes


# In[9]:


orders.describe()


# In[10]:


orders.head()


# In[11]:


orders.dtypes


# Пометки по первичному анализу: Колонки ключи -  order_id, product_id. Колонки с датами нужно будет преобразовывать в datetime

# # 1. Сколько у нас пользователей, которые совершили покупку только один раз?

# In[12]:


orders_uniq_id = orders.merge(customers, on='customer_id')

# соединение датасетов


# In[13]:


orders_uniq_id.query('order_status == "delivered"').groupby('customer_unique_id',as_index=False).agg({'order_id' : 'count'}).query('order_id == 1').shape

# подсчет количества пользователей, которые совершили покупку только один раз. Покупкой считаю доставленный заказ


# #    2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)?

# In[14]:


orders.order_purchase_timestamp = pd.to_datetime(orders.order_purchase_timestamp)
orders['month'] = orders['order_purchase_timestamp'].dt.to_period("M") 

# формат колонки к datetime, создание колонки с годом и месяцем


# In[15]:


avg_month = orders.query('order_status != "delivered"').groupby(['month','order_status'],as_index=False).agg({'order_id':'count'}).groupby('order_status',as_index=False).agg({'order_id' : 'mean'}).rename(columns={'order_id':'avg_month'}).sort_values('avg_month', ascending=False)

# среднемесячное число недоставленных заказов по каждой причине


# In[16]:


avg_month.head()


# In[17]:


sns.set(
    font_scale=1,
    style="whitegrid",
    rc={'figure.figsize':(20,7)}
        )

sns.barplot(x='order_status', y='avg_month', data=avg_month)

# график распределения avg_month 


# #    3. По каждому товару определить, в какой день недели товар чаще всего покупается.

# In[18]:


orders_items = orders.merge(order_items, on='order_id')

# соединение датасетов


# In[19]:


orders_items.order_purchase_timestamp = pd.to_datetime(orders.order_purchase_timestamp)
orders_items['day_of_week'] = orders_items['order_purchase_timestamp'].dt.day_name()

# формат колонки к datetime, создание колонки с днем недели


# In[20]:


orders_items = orders_items.dropna()

# удаление пропущенных значений


# In[21]:


orders_day = orders_items.groupby(['product_id','day_of_week'],as_index=False).agg({'order_id' : 'count'}).drop_duplicates(subset=["product_id"], keep='first').rename(columns={'order_id': 'num_of_orders'}).sort_values('num_of_orders', ascending=False)

# подсчет количества покупок каждого товара в каждый день недели


# In[22]:


orders_day_pivot = orders_day.pivot(index='product_id', columns='day_of_week', values='num_of_orders').fillna(0)

# переворачиваем датасет


# In[23]:


best_day_for_product = orders_day_pivot.idxmax(axis=1).to_frame().rename(columns={0: 'day_of_week'}).reset_index()

# по каждому товару выводим, в какой день недели товар чаще всего покупается


# In[24]:


best_day_for_product.head()


# In[25]:


sns.countplot(x=best_day_for_product["day_of_week"])

# распределение количества самых популярных дней для всех товаров


# # 4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)?

# In[26]:


orders_and_customers = orders.merge(customers, on='customer_id')

#соединение датасетов


# In[27]:


orders_and_customers = orders_and_customers.loc[(orders_and_customers.order_status != 'unavailable') & (orders_and_customers.order_status != 'canceled')]

# оставляем только успешные заказы


# In[28]:


orders_and_customers.order_purchase_timestamp = pd.to_datetime(orders_and_customers.order_purchase_timestamp)
orders_and_customers['year_and_month'] =  orders_and_customers.order_purchase_timestamp.dt.to_period("M") 

# формат колонки к datetime, создание колонки с годом и месяцем


# In[29]:


print('Min date =',orders_and_customers.order_purchase_timestamp.min())
print('Max date =',orders_and_customers.order_purchase_timestamp.max())

# выведем первый и последний месяцы, которые будут неполными


# In[30]:


orders_and_customers = orders_and_customers.query('year_and_month >= "2016-10" & year_and_month < "2018-09"')

# исключаем неполные месяцы


# In[31]:


orders_and_customers = orders_and_customers.groupby(['customer_unique_id','year_and_month'],as_index=False).agg({'order_id' : 'count'}).sort_values('order_id',ascending=False).rename(columns={'order_id' : 'orders_count'})

# количество покупок для каждого пользователя в каждом месяце


# In[32]:


orders_and_customers.head()


# In[33]:


orders_and_customers['avg_week_orders'] = orders_and_customers.orders_count/((orders_and_customers.year_and_month.dt.daysinmonth)/7)

# добавим колонку с средним количеством заказов для каждого пользователя в каждую неделю


# In[34]:


orders_and_customers.head()


# In[35]:


week_avg_orders = orders_and_customers.groupby('customer_unique_id',as_index=False).agg({'avg_week_orders' : 'mean'})

# сгруппируем по пользователям и выведем для каждого среднее количество заказов в неделю


# In[36]:


week_avg_orders.head()


# # 5. Используя pandas, проведи когортный анализ пользователей. В период с января по декабрь выяви когорту с самым высоким retention на 3й месяц.

# In[37]:


orders_customers = orders.merge(customers, on='customer_id')

#соединение датасетов


# In[38]:


orders_customers.head()


# In[39]:


orders_customers.order_purchase_timestamp = pd.to_datetime(orders_customers.order_purchase_timestamp)
orders_customers['year_and_month'] =  orders_customers.order_purchase_timestamp.dt.to_period("M") 

# формат колонки к datetime, создание колонки с годом и месяцем


# In[40]:


orders_customers = orders_customers.loc[(orders_customers.order_status != 'unavailable') & (orders_customers.order_status != 'canceled')]

# оставляем только успешные заказы


# In[41]:


orders_customers.head()


# In[42]:


first_order = orders_customers.groupby('customer_unique_id',as_index=False).agg({'year_and_month' : 'min'}).rename(columns={'year_and_month' : 'first_order_date'})

# находим дату первого заказа для каждого пользователя


# In[43]:


first_order.head()


# In[44]:


orders_customers = orders_customers[['order_id','customer_unique_id','year_and_month']]
cohorts = orders_customers.merge(first_order,how='inner',on='customer_unique_id')

# оставляет нужные кололнки, соединяем с датасетом даты первого заказа


# In[45]:


cohorts.head()


# In[46]:


cohorts = cohorts.groupby(['first_order_date','year_and_month'],as_index=False).agg({'customer_unique_id': 'nunique'}).rename(columns={'customer_unique_id': 'users'})


# In[47]:


cohorts.head()


# In[48]:


def cohort_month(df):
    df['month'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby('first_order_date').apply(cohort_month)

# функция, которая нумерует месяцы по диапазону окна в когорте


# In[49]:


cohorts = cohorts.pivot_table(columns='month', index = 'first_order_date',values='users')

# развернем датасет


# In[50]:


x = cohorts[1]
retention = cohorts.divide(x, axis=0).round(4)
retention[retention[3] == retention[3].max()].index

# находим когорту с самым высоким retention на 3й месяц


# In[51]:


retention


# In[52]:


plt.rcParams['font.size'] = '11'
plt.figure(figsize=(20,14))
plt.title('users')
ax = sns.heatmap(data=retention, annot=True, vmin=0.0,vmax=0.008 ,cmap='Blues')
ax.set_yticklabels(retention.index)
fig=ax.get_figure()
fig.savefig("Retention Counts.png")
plt.show()

# визуализируем


# # 6. Построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию.

# In[53]:


df_rfm = orders.merge(customers, on='customer_id').merge(order_items, on='order_id')

# объеденим все датасеты в один


# In[54]:


df_rfm = df_rfm.loc[(df_rfm.order_status != 'unavailable') & (df_rfm.order_status != 'canceled')]

# оставляем только успешные заказы 


# In[55]:


df_rfm.order_purchase_timestamp = pd.to_datetime(df_rfm.order_purchase_timestamp)

# формат колонки к datetime


# In[56]:


df_rfm.head()


# In[57]:


current_date = df_rfm.order_purchase_timestamp.max()

# за текущую дату возьмем дату последней покупки


# In[58]:


print('Текущая дата:',current_date)


# In[59]:


df_rfm = df_rfm.groupby('customer_unique_id').agg({'order_purchase_timestamp' : 'max','order_id' : 'count','price' : 'sum'})

# найдем дату последнего заказа, а также посчитаем количество заказов, обшую сумму заказов для каждого пользователя


# In[60]:


df_rfm['recency'] = current_date - df_rfm.order_purchase_timestamp

# добавим колонку с временем от последней покупки пользователя до текущей даты


# In[61]:


df_rfm = df_rfm.rename(columns={'order_id' : 'frequency','price' : 'monetary'})
df_rfm.head()
# переименуем колонки


# # Изучим полученные данные и и определим границы метрик RFM

# In[62]:


df_rfm.describe()


# # Recency
Границы Recency определим по значениям 25, 50, 75 перцентилей. Чем меньше дней, тем выше рейтинг.
# In[63]:


r_q25 = df_rfm.recency.quantile(0.25)
r_q50 = df_rfm.recency.quantile(0.5)
r_q75 = df_rfm.recency.quantile(0.75)

def recency_rate(recency):
    if recency < r_q25:
        return 'Н'
    elif r_q25 <= recency < r_q50:
        return 'СД'
    elif r_q50 <= recency < r_q75:
        return 'Д'
    elif recency >= r_q75:
        return 'ОД'

# функция для определения рейтинга Recency    


# In[64]:


df_rfm['recenty_rate'] = df_rfm.recency.apply(lambda x: recency_rate(x))
df_rfm.head()

# применим функцию, рассчитаем рейтинг


# # Frequency

# In[65]:


df_rfm.frequency.value_counts()


#  Т.к большинство клиентов сделали одну покупку, то логично будет разбить frequency на 4 категории:
# - 1 покупка
# - 2 покупки
# - 3 покупки
# - более 3 покупок

# In[66]:


def frequency_rate(frequency):
    if frequency == 1:
        return '1'
    elif frequency == 2:
        return '2'
    elif frequency == 3:
        return '3'
    elif frequency > 3:
        return '4'

# функция для определения рейтинга Frequency 


# In[67]:


df_rfm['frequency_rate'] = df_rfm.frequency.apply(lambda x: frequency_rate(x))
df_rfm.head()

# применим функцию, рассчитаем рейтинг


# # Monetary

# In[68]:


sns.distplot(df_rfm.monetary)

# график распределения


# Распределение значений monetary отклоняются от нормального распределения.

# In[69]:


df_rfm.loc[df_rfm.monetary > df_rfm.monetary.quantile(0.75)].monetary.describe()


# In[70]:


df_rfm.loc[df_rfm.monetary > 371.925000].monetary.describe()


# In[71]:


df_rfm.loc[df_rfm.monetary > 829.495000].monetary.describe()


# В связи с таким неравномерным распределение monetary, для каждой границы я буду брать значения до 75 перцентиля, затем отсекать данные (> 75 перцентиля) и снова возьму значения до 75 перцентиля.

# In[72]:


m_q75_1 = df_rfm.monetary.quantile(0.75)
m_q75_2 = df_rfm.loc[df_rfm.monetary > m_q75_1].monetary.quantile(0.75)
m_q75_3 = df_rfm.loc[df_rfm.monetary > m_q75_2].monetary.quantile(0.75)
def monetary_rate(monetary):
    if monetary < m_q75_1:
        return 'М'
    elif monetary >= m_q75_1 and monetary < m_q75_2:
        return 'НС'
    elif monetary >= m_q75_2 and monetary < m_q75_3:
        return 'С'
    elif monetary >= m_q75_3:
        return 'Б'
    
# функция для определения рейтинга monetary


# In[73]:


df_rfm['monetary_rate'] = df_rfm.monetary.apply(lambda x: monetary_rate(x))
df_rfm.head()
# применим функцию, рассчитаем рейтинг


# In[74]:


df_rfm = df_rfm[['recency','frequency','monetary','recenty_rate','frequency_rate','monetary_rate']]


# In[75]:


df_rfm['rfm'] = df_rfm.recenty_rate + ';' + df_rfm.frequency_rate + ';' +  df_rfm.monetary_rate

# объеденим рейтинги


# In[76]:


df_rfm.head()


# In[77]:


df_rfm.rfm.nunique()

# количество уникальных RFM-микросегментов


# In[78]:


df_rfm.rfm.value_counts()


# # Составляем RFM-сегменты,основываясь в первую очередь на метрике monetary, далее на recency и frequency
Recency   (ОД-очень давний; Д-давний; СД-средней давности; Н-недавний.)
Frequency (1-одна покупка; 2-две покупки; 3-три покупки; 4-более трех покупок.)
Monetary  (М-маленькая сумма; НС-небольная сумма; С-средняя сумма; Б-большая сумма.)


«Бюджетный»:
Н1М                   — новые клиенты
Н2М                   — вернувшиеся клиенты
Н3М/Н4М               — лояльные клиенты
СД1М/Д1М              — неактивные новые клиенты
СД2М/Д2М              — неактивные вернувшиеся клиенты
СД3М/СД4М/Д3М/Д4М     — неактивные лояльные клиенты
ОД1М                  — ушедшие новые клиенты
ОД2М                  — ушедшие вернувшиеся клиенты
ОД3М/ОД4М             — ушедшие лояльные клиенты

«Стандарт»:
Н1НС/Н1С              — новые клиенты
Н2НС/Н2С              — вернувшиеся клиенты
Н3НС/Н3С/Н4НС/Н4С     — лояльные клиенты
СД1НС/СД1С/Д1НС/Д1С   — неактивные новые клиенты
СД2НС/СД2С/Д2НС/Д2С   — неактивные вернувшиеся клиенты
СД3НС/СД3С/СД4НС/СД4С
/Д3НС/Д3С/Д4НС/Д4С    — неактивные лояльные клиенты
ОД1НС/ОД1С            — ушедшие новые клиенты
ОД2НС/ОД2С            — ушедшие вернувшиеся клиенты
ОД3НС/ОД3С/ОД4НС/ОД4С — ушедшие лояльные клиенты

«VIP»:
Н1Б                   — новые клиенты
Н2Б                   — вернувшиеся клиенты
Н3Б/Н4Б               — лояльные клиенты
СД1Б/Д1Б              — неактивные новые клиенты
СД2Б/Д2Б              — неактивные вернувшиеся клиенты
СД3Б/СД4Б/Д3Б/Д4Б     — неактивные лояльные клиенты
ОД1Б                  — ушедшие новые клиенты
ОД2Б                  — ушедшие вернувшиеся клиенты
ОД3Б/ОД4Б             — ушедшие лояльные клиенты
# In[79]:


df_rfm_value_counts = df_rfm.rfm.value_counts().to_frame().reset_index()

# датасет для визуализации графика распределения количества клиентов по микросегментам


# In[80]:


sns.barplot(data=df_rfm_value_counts, x='index', y='rfm')
plt.xticks(rotation=90)
plt.title('График распределения количества клиентов по микросегментам')
plt.xlabel('RFM-микросегмент')
plt.ylabel('Количество клиентов')

# строим график


# In[81]:


df_rfm_monetary_sum = df_rfm.groupby('rfm',as_index=False).agg({'monetary' : 'sum'}).sort_values('monetary', ascending=False)

# датасет для визуализации графика распределения общей суммы покупок по микросегментам


# In[82]:


sns.barplot(data=df_rfm_monetary_sum, x='rfm', y='monetary')
plt.xticks(rotation=90)
plt.title('График распределения общей суммы покупок по микросегментам')
plt.xlabel('RFM-микросегмент')
plt.ylabel('Сумма покупок')

# строим график


# In[83]:


def segment(monetary_rate):
    if monetary_rate == 'М':
        return 'Бюджетный'
    elif monetary_rate == 'НС' or monetary_rate == 'С':
        return 'Стандарт'
    elif monetary_rate == 'Б':
        return 'VIP'
    
# функция для определения сегмента    


# In[84]:


df_rfm['segment'] = df_rfm.monetary_rate.apply(lambda x: segment(x))

# применяем функцию, рассчитываем сегмент


# In[85]:


df_rfm.head()


# In[86]:


df_rfm_segments_sum = df_rfm.groupby('segment',as_index=False).agg({'monetary' : 'sum'}).sort_values('monetary', ascending=False)

# датасет для визуализации графика распределения суммы покупок по сегментам


# In[87]:


df_rfm_segments_sum.head()


# In[88]:


sns.barplot(data=df_rfm_segments_sum, x='segment', y='monetary')
plt.xticks(rotation=90)
plt.title('График распределения общей суммы покупок по сегментам')
plt.xlabel('Сегмент')
plt.ylabel('Сумма покупок')

# строим график


# # Ключевые выводы по данным:

# - Большинство пользователей делает не более одной покупки, нужно разработать стратегию для удержания клиентов
# - Относительно небольшой доход от клиентов сегмента "VIP", нужно разработать систему лояльности
# - Высокодоходные клиенты не задерживаются, нужно разработать более гибкую систему лояльности для высокодоходных клиентов

# In[ ]:





# In[ ]:




