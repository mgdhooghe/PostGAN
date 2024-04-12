#!/bin/bash

## [directory] [dataset_name] [real$size_file] [protected feature] [privileged value] [predicted feature] [preferred value] [selected_data_percentage (ex. .5)] ["gpu"/None]]

python_code=main-parallel-ablation.py
size=-SPLIT-VALID-20

## German
dataset=german
real=german/GERMAN
prot=gender
priv=female
pred=labels
pref=1

python $python_code ../german/german-VGAN-1hot-split-validation-best/9/ $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
python $python_code ../german/german-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 
python $python_code ../german/german-VGAN-WTRAIN-split/  $dataset-WGAN-norm $real$size.csv $prot $priv $pred $pref  
python $python_code ../german/german-VGAN-1hot-gmm-split-60-WTrain/8-9/  $dataset-WGAN-gmm $real$size.csv $prot $priv $pred $pref 

python $python_code ../german/german-tabfairgan/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
python $python_code ../german/german-FAIRGAN/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  


## Compas-Race

dataset=compas-race
real=propublica-compas/PROPUBLICA-COMPAS
prot=race
priv=Caucasian
pred=two_year_recid
pref=0

python $python_code ../compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
python $python_code ../compas/compas-pro-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 
python $python_code ../compas/compas-pro-VGAN-WTRAIN-split/  $dataset-WGAN-norm $real$size.csv $prot $priv $pred $pref  
python $python_code ../compas/compas-pro-VGAN-1hot-gmm-split-60-WTrain/8-9/  $dataset-WGAN-gmm $real$size.csv $prot $priv $pred $pref  

python $python_code ../compas/compas-pro-TABFAIRGAN/race/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
python $python_code ../compas/compas-pro-FAIRGAN/race/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  

## Compas-Gender

dataset=compas-gender
prot=sex
priv=Female

python $python_code ../compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
python $python_code ../compas/compas-pro-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 
python $python_code ../compas/compas-pro-VGAN-WTRAIN-split/  $dataset-WGAN-norm $real$size.csv $prot $priv $pred $pref  
python $python_code ../compas/compas-pro-VGAN-1hot-gmm-split-60-WTrain/8-9/  $dataset-WGAN-gmm $real$size.csv $prot $priv $pred $pref  

python $python_code ../compas/compas-pro-TABFAIRGAN/gender/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
python $python_code ../compas/compas-pro-FAIRGAN/gender/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  

## Medical

dataset=medical
real=medical/meps21
prot=RACE
priv=1
pred=UTILIZATION
pref=1

#python $python_code ../medical/medical-VGAN-1hot-norm-split-60/8-9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
#python $python_code ../medical/medical-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 
#python $python_code ../medical/medical-WTRAIN-VGAN-1hot-norm-split-60/8-9/  $dataset-WGAN-norm $real$size.csv $prot $priv $pred $pref  
#python $python_code ../medical/medical-WTRAIN-VGAN-1hot-gmm-split-60-best/8-9/  $dataset-WGAN-gmm $real$size.csv $prot $priv $pred $pref  

#python $python_code ../medical/medical-TABFAIRGAN/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
#python $python_code ../medical/medical-FAIRGAN/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  

## Bank

dataset=bank
real=bank/bank-full
prot=age
priv=1
pred=y
pref=yes

#python $python_code ../bank/bank-VGAN-best/9-age/  $dataset-VGAN-norm $real$size-age.csv $prot $priv $pred $pref 
#python $python_code ../bank/bank-full-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size-age.csv $prot $priv $pred $pref 
#python $python_code ../bank/bank-VGAN-WTRAIN-split/  $dataset-WGAN-norm $real$size-age.csv $prot $priv $pred $pref  
#python $python_code ../bank/bank-full-VGAN-1hot-gmm-split-60-WTrain/8-9/  $dataset-WGAN-gmm $real$size-age.csv $prot $priv $pred $pref  
#python $python_code ../bank/bank-TABFAIRGAN/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
#python $python_code ../bank/bank-FAIRGAN/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  

## Adult

dataset=adult
real=adult/ADULT
prot=sex
priv=Male
pred=income
pref='>50K'


#python $python_code ../adult/adult-VGAN-1hot-norm-split-60/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
#python $python_code ../adult/adult-VGAN-1hot-gmm-split-60/8-9/ $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 
#python $python_code ../adult/adult-VGAN-WTRAIN-split/  $dataset-WGAN-norm $real$size.csv $prot $priv $pred $pref  
#python $python_code ../adult/adult-VGAN-1hot-gmm-split-60-WTrain-best/8-9/  $dataset-WGAN-gmm $real$size.csv $prot $priv $pred $pref  


#python $python_code ../adult/adult-TABFAIRGAN/  $dataset-tabfairgan $real$size.csv $prot $priv $pred $pref  
#python $python_code ../adult/adult-FAIRGAN/  $dataset-fairgan $real$size.csv $prot $priv $pred $pref  

