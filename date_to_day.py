import numpy as np

def date_to_day(str_date):
    date = np.array(str_date).astype(np.int)
    year = (np.floor(date / 10000) - 2000).astype(np.int)
    m_d = date % 10000
    month = np.floor(m_d / 100).astype(np.int)
    day = m_d % 100
    check_leaf_month = month > 2
    extra = np.logical_not(year % 4)
    leaf = check_leaf_month * extra
    dict_month = {1: 0, 2: 31, 3: 28, 4: 31, 5: 30, 6: 31, 7: 30, 8: 31, 9: 31, 10: 30, 11: 31, 12: 30}
    dict_acc = {}
    temp = 0
    for keys in dict_month:
        dict_acc[keys] = temp + dict_month[keys]
        temp += dict_month[keys]
    year_to_day = year * 365
    month_to_day = np.vectorize(dict_acc.get)(month)
    res = year_to_day + month_to_day + day + leaf
    return res

print(date_to_day(['20171231']))
