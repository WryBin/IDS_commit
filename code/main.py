import os
import numpy as np
import pandas as pd
import lightgbm as lgb

def LightGBM(train_data, val_data, pred_data, features):

    params = {
    'boosting': 'gbdt',
    'objective': 'rmse',
    'num_leaves': 300,
    'learning_rate': 0.1,
    'metric': {'rmse'},
    'verbose': -1,
    'min_data_in_leaf': 6,
    'max_depth':30,
    'seed':42, 
    }

    lgb_train = lgb.Dataset(train_data[features], train_data['Steam_flow'].values)
    lgb_eval = lgb.Dataset(val_data[features], val_data['Steam_flow'].values, reference=lgb_train)

    ### 模型训练
    gbm = lgb.train(params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=300, 
                    callbacks=[lgb.early_stopping(stopping_rounds=100)],
                    )
    ### 模型预测
    Y_train = gbm.predict(train_data[features], num_iteration=gbm.best_iteration)
    Y_val = gbm.predict(val_data[features], num_iteration=gbm.best_iteration)
    Y_pred = gbm.predict(pred_data[features], num_iteration=gbm.best_iteration)


    return Y_train, Y_val, Y_pred, gbm


def adv_val(Train_data, pred_data, features):

    # 对抗验证
    df_adv =  pd.concat([Train_data, pred_data])

    adv_data = lgb.Dataset(
        data=df_adv[features], label=df_adv.loc[:, 'Is_Test']
    )

    # 定义模型参数
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 300,
    'learning_rate': 0.1,
    'metric': 'auc',
    'verbose': -1,
    'min_data_in_leaf': 6,
    'max_depth':30,
    'seed':42, 
    'sub_feature': 0.7, 
    }

    adv_cv_results = lgb.cv(
        params, 
        adv_data,
        num_boost_round=30, 
        nfold=5, 
        early_stopping_rounds=10, 
        # verbose_eval=True, 
        seed=42)

    # print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
    #     adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))

    # print('模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))

    # 使用训练好的模型，对所有的样本进行预测，得到各个样本属于测试集的概率
    params['n_estimators'] = len(adv_cv_results['auc-mean'])

    model_adv = lgb.LGBMClassifier(**params)
    model_adv.fit(df_adv[features], df_adv.loc[:, 'Is_Test'])

    preds_adv = model_adv.predict_proba(df_adv[features])[:, 1]
    
    return preds_adv, adv_cv_results


# 数据导入
Train_data = pd.read_csv("raw_data/train/outputs/主蒸汽流量.csv")

for path, dirs, files in os.walk("raw_data/train/inputs"):
    for file in files:
        temp = pd.read_csv(os.path.join(path, file))
        temp.drop_duplicates('时间', inplace=True)
        Train_data = Train_data.merge(temp, on=['时间'], how='left')

Test_data = pd.read_csv("raw_data/test/二次风调门.csv")['时间'].to_frame()
for path, dirs, files in os.walk("raw_data/test"):
    for file in files:
        temp = pd.read_csv(os.path.join(path, file))
        temp.drop_duplicates('时间', inplace=True)
        Test_data = Test_data.merge(temp, on=['时间'], how='left')

Train_data['Is_Test'] = 0
Test_data['Is_Test'] = 1

combi = pd.concat([Train_data, Test_data], axis=0, ignore_index=True)

combi.rename(columns={'时间':'Time', '主蒸汽流量':'Steam_flow', '二次风调门':'2_air_door', '炉排启停':'Grate_switch', '二次风量':'2wind',\
     '一次风量':'1wind', '一次风调门':'1_air_door', '氧量设定值':'O2', '推料器自动投退信号':'Push_auto_feed', 'SO2含量':'SO2',
      '推料器手动指令':'Manual_feed', 'CO含量':'CO', '主蒸汽流量设定值':'Steam_flow_set', '汽包水位':'Water_level', '推料器自动指令':'Push_auto',
       'HCL含量':'HCL','NOx含量':'NOx', '炉排实际运行指令':'Grate_run', '炉排手动指令':'Grate_manual', '给水流量':'Water_flow',
       '炉排自动投退信号':'Grate_auto', '推料器启停':'Feed_switch', '引风机转速':'Fan_speed'}, inplace=True)

# 缺失值处理
combi.dropna(axis=0,how='any',subset=['Grate_switch'], inplace=True)

# 生成一些新的信息
Water_level_cum = (np.cumsum(combi['Water_level']-combi['Water_level'].mean())/200).shift(-61)
combi['Water_level_cum'] = Water_level_cum + combi['Water_flow']
combi['Water_level_cum'][-61:] = combi['Water_level_cum'][258992:259053]+14.3352
combi['Water_level_cum_rolling'] = combi['Water_level_cum'].rolling(3000, center=True, min_periods=1).mean()
combi['Steam_flow_rolling'] = combi['Steam_flow'].rolling(3000, center=True, min_periods=1).mean()
combi['simu'] = combi['Water_level_cum'] - combi['Water_level_cum_rolling'] + combi['Steam_flow_rolling']
combi['simu'][-1800:] = combi['Water_level_cum'][-1800:]

# 数据集划分
Train_data = combi[(combi['Time'] > '2021-12-20 08:00:00') & (combi['Time'] < '2021-12-22 23:30:00')]
pred_data = combi[combi['Time'] >= '2021-12-22 23:30:00']

# 特征选定
features = ['2wind', 'SO2','NOx', 'Fan_speed','simu']


# redivision
sample_weight, adv_cv_results = adv_val(Train_data, pred_data, features)
Train_data['sample_weight'] = sample_weight[:-1800]
val_data = Train_data[Train_data['sample_weight'] >= 0.015]
Train_data.loc[Train_data['sample_weight'] >= 0.015, 'sample_weight'] = None
Train_data['Is_Train'] = Train_data.loc[:, 'sample_weight'].rolling(200, center=True).sum()
train_data = Train_data[Train_data['Is_Train'] > 0]

# LightGBM
Y_train, Y_val, Y_pred, model= LightGBM(train_data, val_data, pred_data, features)

pred_data['ID'] = np.arange(1, len(pred_data)+1)
pred_data['Steam_flow'] = Y_pred
pred_data = pred_data[['ID', 'Time', 'Steam_flow']]
pred_data.to_csv("result/result.csv", index=False, sep=',')