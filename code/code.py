# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:52:53 2019

@author: renhao
"""

import pandas as pd
import datetime
import csv
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn import preprocessing

def get_data(path):
    
    
    features=['title_similary','body_similary','r_rate','r_one_rate','p_rate','p_one_rate','all_r_one_rate','all_r_rate','all_p_one_rate','all_p_rate',
            
            'related_reporter_issue_q_rate','related_reporter_commit_q_rate','related_reporter_issue_r_rate','related_reporter_commit_r_rate','related_reporter_r_count','related_reporter_p_count','related_reporter_r_r_count','related_reporter_p_r_count',
            
            'related_part_issue_q_rate','related_part_commit_q_rate','related_part_issue_r_rate','related_part_commit_r_rate','related_part_r_count','related_part_p_count','related_part_r_r_count','related_part_p_r_count']
        
    
    
    issues=[]
    related_issues=[]
    datas=[]
    with open(path) as file:
        reader=csv.DictReader(file)
        for row in reader:
            if('/' in row['issue'] and '#' in row['issue']):
                issues.append(row['issue'])
                related_issues.append(row['related_issue'])
                temp=[]
                for feature in features:
                    temp.append(float(row[feature]))
                datas.append(temp)
        file.close()
    
    return (issues,related_issues,datas)


def main():
    
    repos=['scipy','numpy','matplotlib','scikit-learn','pandas','ipython','astropy']
    reposs=['scipy/scipy','numpy/numpy','matplotlib/matplotlib','scikit-learn/scikit-learn','pandas-dev/pandas','ipython/ipython','astropy/astropy']

    train_data_path='G:/paper_test2/'+repos[i]+'/new_dataset/cross-project_train_datas.csv'
    test_path='G:/paper_test2/'+repos[j]+'/new_dataset/cross-project_test_issues.csv'
    test_data_path=root_path+repos[j]+'/test_data_info/'+issue.replace('/','_').replace('#','_')+'_issue_info.csv'
                
            
    save_path='G:/paper_test2/'+repos[i]+'/new_dataset/'+repos[j]+'/'
        
    
    for i in range(5,len(repos)):
        
        
        
        file_data=pd.read_csv(train_data_path)
        train_data=np.array(file_data.loc[:,:])
        #key_data=key_data[:,1:]
        labels=train_data[:,-1]
        train_datas=train_data[:,1:-1]
        
        print(repos[i],train_datas.shape)
        
        svm_clf = svm.SVC(kernel='rbf', C=1,probability=True)
        rf_clf=RandomForestClassifier(n_estimators=100)
        lr_clf=LogisticRegression(C=1)
        
        min_max_scaler = preprocessing.MinMaxScaler()
        train_datas=min_max_scaler.fit_transform(train_datas)
                
        svm_clf.fit(train_datas,labels)
        rf_clf.fit(train_datas,labels)
        lr_clf.fit(train_datas,labels)
        
        for j in range(len(repos)):
            
            if(i==j):
                continue
            
            
            test_issues=[]
            test_related_issues=[]
            with open(test_path) as file:
                reader=csv.DictReader(file)
                for row in reader:
                    test_issues.append(row['issue'])
                    test_related_issues.append(row['related_issue'])
                file.close()
            
            print(repos[i],repos[j],len(test_issues))
            
            for issue in set(test_issues):
                
                temp=[]
                for k in range(len(test_issues)):
                    if(test_issues[k]==issue):
                        temp.append(test_related_issues[k])
                #test_issue_related[issue]=temp
                
                t,q,test_datas=get_data(test_data_path)
                        
                min_max_scaler = preprocessing.MinMaxScaler()
                test_datas=min_max_scaler.fit_transform(test_datas)
                        
                print(issue,len(test_datas),len(test_datas[0]))
                
                svm_results=svm_clf.predict_proba(test_datas)
                #print(svm_results)
                file=open(save_path+issue.replace('/','_').replace('#','_')+'_svm_test_result.csv','w',newline='',encoding='utf8')
                header=['issue','related_issue','0_prob','1_prob']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                        
                for k in range(len(svm_results)):
                    file_csv.writerow({'issue':t[k],'related_issue':q[k],'0_prob':svm_results[k][0],'1_prob':svm_results[k][1]})
                        
                dt_results=dt_clf.predict_proba(test_datas)
                        #print(svm_results)
                   
                rf_results=rf_clf.predict_proba(test_datas)
                        #print(svm_results)
                file=open(save_path+issue.replace('/','_').replace('#','_')+'_random_forest_test_result.csv','w',newline='',encoding='utf8')
                header=['issue','related_issue','0_prob','1_prob']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                        
                for k in range(len(rf_results)):
                    file_csv.writerow({'issue':t[k],'related_issue':q[k],'0_prob':rf_results[k][0],'1_prob':rf_results[k][1]})
                        
                lr_results=lr_clf.predict_proba(test_datas)
                        #print(svm_results)
                file=open(save_path+issue.replace('/','_').replace('#','_')+'_logistic_regression_test_result.csv','w',newline='',encoding='utf8')
                header=['issue','related_issue','0_prob','1_prob']
                file_csv=csv.DictWriter(file,header)
                file_csv.writeheader()
                        
                for k in range(len(lr_results)):
                    file_csv.writerow({'issue':t[k],'related_issue':q[k],'0_prob':lr_results[k][0],'1_prob':lr_results[k][1]})

if __name__=='__main__':
    main()