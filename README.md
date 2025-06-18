# AlphaForge-MAA(AFF-MAA)

### Run Our Model

#### stage1: Minning alpha factors
```shell
python train_AFF.py --instruments=all --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --zoo_size=20 --is_use_multi_agent=True --is_distill=True --is_cross_finetune=True
```

Here,
- `instruments` is the dataset to use.
- `seeds` is random seed list, e.g., `[0,1,2]` or `[0]`. 
- `train_end_year` is the last year of training set, when train_end_year is 2020,the train,valid and test set is seperately: `2010-01-01 to 2020-12-31`,`2021-01-01 to 2021-12-31`,`2022-01-01 to 2022-12-31`
- `save_name` is the prefix when saving running results. `zoo_size` is the num of factors to save at stage 1 mining model.

#### stage2: Combining alpha factors
```shell
python combine_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --n_factors=10 --window=inf
```
Here `instruments,train_end_year,seeds,save_name`,` must be the same as it in stage 1
- `n_factors` is the num of factors used at each day, it should be less than or equal to `zoo_size` in stage 1.
- `window` is the slicing window that is used to evaluate the alpha factors in order to dynamicly select and cobine.

#### stage3: Show the results

You could run the ipython notebook file 

```shell
exp_AFF_calc_result.ipynb
```

to generate and concat experiment result.

#### ML models including XGBoost, LightGBM and MLP:

train & show results: `exp_ML_train_and_result.ipynb`

#### attention
version of cuda maybe more than 11.3 (recommend torch 1.11 + cuda 11.3)



