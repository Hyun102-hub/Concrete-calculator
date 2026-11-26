"""1,000회 반복 7:3 검증을 다시 돌려 회차별·배합별 예측값을 모두 저장 (validation_runs.csv와 같은 결과인지 확인)"""
import numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
N=["시멘트","고로슬래그","플라이애시","물","고성능감수제","굵은골재","잔골재","재령","압축강도"]; M,F=N[:7],N[:8]
raw=pd.read_excel('/home/user/concrete_strength/data_build/raw/Concrete_Data.xls'); raw.columns=N
d1=raw.drop_duplicates().reset_index(drop=True); dc=d1[~d1.duplicated(subset=F,keep=False)].reset_index(drop=True)
dc["배합번호"]=dc.groupby(M).ngroup()+1
def mk(seed):
    pre=ColumnTransformer([("재료","passthrough",M),("재령",FunctionTransformer(np.log),["재령"])])
    return Pipeline([("pre",pre),("scale",MinMaxScaler()),("mlp",MLPRegressor(hidden_layer_sizes=(64,),activation="tanh",alpha=1e-4,learning_rate_init=0.01,max_iter=5000,early_stopping=True,validation_fraction=0.15,n_iter_no_change=50,random_state=seed))])
def run(seed,sch):
    if sch=="무작위": a,b=train_test_split(np.arange(len(dc)),test_size=0.3,random_state=seed)
    else: a,b=next(GroupShuffleSplit(1,test_size=0.3,random_state=seed).split(dc,groups=dc["배합번호"]))
    m=mk(seed).fit(dc.iloc[a][F],dc.iloc[a]["압축강도"]); p=m.predict(dc.iloc[b][F])
    return pd.DataFrame({"방식":sch,"회차":seed+1,"행":b,"예측":p.astype("float32")})
out=pd.concat(Parallel(n_jobs=-1)(delayed(run)(s,sch) for sch in ["무작위","배합 기준"] for s in range(1000)),ignore_index=True)
out.to_parquet('/home/user/concrete_strength/validation/validation_predictions.parquet',index=False)
dc.to_csv('/home/user/concrete_strength/validation/validation_data_987.csv',index=False,encoding='utf-8-sig')
# 확인: 회차별 R²가 저장된 결과와 같은지
runs=pd.read_csv('/home/user/concrete_strength/validation/validation_runs.csv')
y=dc["압축강도"].values; chk=out.assign(y=y[out["행"].values]).groupby(["방식","회차"]).apply(lambda g: r2_score(g.y,g.예측.astype(float)),include_groups=False).rename("R2").reset_index()
mm=chk.merge(runs[["방식","회차","R²"]],on=["방식","회차"]); print("회차 수",len(mm),"R² 최대 차이",float((mm.R2-mm["R²"]).abs().max()))
