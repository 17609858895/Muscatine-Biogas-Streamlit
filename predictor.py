"""Offline inference; only supplied origin history and explicit plans are read."""
from pathlib import Path
from io import StringIO
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent
HORIZONS=[1,2,3,4,6,9,12,18,24]
BINS=[(0,0),(1,2),(3,5),(6,11),(12,23),(24,47),(48,71),(72,167)]
LAB=['0','1-2','3-5','6-11','12-23','24-47','48-71','72-167']
# CSV column, native training variable, SI/native multiplier, SI additive offset.
SCHEMA=[('biogas_m3_h','Biogas',1.69901,0),('cover_d1_m','H-Dig1_FT',.3048,0),('cover_d2_m','H-Dig2_FT',.3048,0),
 ('hsw_d1_m3_h','Q-HSW1_GPM',.227125,0),('hsw_d2_m3_h','Q-HSW2_GPM',.227125,0),('hsw_tank_m','H-HSW_ft',.3048,0),
 ('hsw_d2_previous_day_m3','V-HSW-DIG2-yest_gal',.003785,0),('ps_on_fraction','Q-PS_on',1,0),('twas_m3_h','Q-TWAS_GPM',.227125,0),
 ('temperature_d1_c','D1_TEMPERATURE',5/9,-32*5/9),('temperature_d2_c','D2_TEMPERATURE',5/9,-32*5/9),
 ('influent_m3_h','Q-influent_MGD',3785.41/24,0),('aeration_m3_h','Q-Aeration_SCFM',1.69901,0),
 ('ras_m3_h','Q-RAS_MGD',3785.41/24,0),('was1_m3_h','Q-WAS1_MGD',3785.41/24,0),('was2_m3_h','Q-WAS2_MGD',3785.41/24,0)]
MODELS={}
def csv_frame(text,required):
    try:d=pd.read_csv(StringIO(text),float_precision='round_trip')
    except Exception:raise ValueError('Cannot read CSV. Use the downloaded UTF-8 template.')
    missing=set(['timestamp',*required])-set(d.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    try:t=pd.to_datetime(d.pop('timestamp'),errors='raise'); d[required]=d[required].apply(pd.to_numeric,errors='raise')
    except Exception:raise ValueError('Timestamps and numeric values must be valid.')
    if t.dt.tz is not None:raise ValueError('Use the same local, timezone-free timestamps as the SCADA history.')
    d.index=pd.DatetimeIndex(t)
    if not d.index.is_unique or not d.index.is_monotonic_increasing:raise ValueError('Timestamps must be unique and increasing.')
    if (d.index.minute!=0).any() or (d.index.second!=0).any():raise ValueError('All timestamps must be on the hour.')
    if len(d)>1 and not (d.index.to_series().diff().iloc[1:]==pd.Timedelta(hours=1)).all():raise ValueError('Hourly data must be consecutive, without gaps.')
    if not np.isfinite(d[required].values).all():raise ValueError('Missing or non-finite values are not accepted. No values are filled automatically.')
    return d[required]
def read_history(text):
    d=csv_frame(text,[s[0] for s in SCHEMA])
    if len(d)<168:raise ValueError('Provide at least 168 consecutive hourly records ending at the forecast origin.')
    d=d.iloc[-168:]
    if ((d.ps_on_fraction<0)|(d.ps_on_fraction>1)).any():raise ValueError('PS on-time fraction must be between 0 and 1.')
    for col,_,_,_ in SCHEMA:
        if 'temperature' not in col and (d[col]<0).any():raise ValueError(col+' must be non-negative.')
    return pd.DataFrame({v:(d[col]-off)/factor for col,v,factor,off in SCHEMA},index=d.index)
def feature_row(history,h,plan=None):
    vals={}
    for _,v,_,_ in SCHEMA:
        for (a,b),lab in zip(BINS,LAB):vals[f'{v}|L{lab}']=float(history[v].shift(a).rolling(b-a+1).mean().iloc[-1])
    t=history.index[-1]+pd.Timedelta(hours=h)
    vals.update(hod_sin=np.sin(2*np.pi*t.hour/24),hod_cos=np.cos(2*np.pi*t.hour/24),dow_sin=np.sin(2*np.pi*t.dayofweek/7),dow_cos=np.cos(2*np.pi*t.dayofweek/7))
    if plan is not None:
        vals.update({'Planned HSW|F':float(plan.hsw_m3_h.iloc[:h].mean()/.227125),'Planned PS|F':float(plan.ps_on_fraction.iloc[:h].mean()),'Planned TWAS|F':float(plan.twas_m3_h.iloc[:h].mean()/.227125)})
    # LightGBM native files normalise spaces in the three plan feature names.
    return {k.replace(' ','_'):v for k,v in vals.items()}
def predict(history_text,plan_text=None):
    history=read_history(history_text); origin=history.index[-1];plan=None
    if plan_text:
        plan=csv_frame(plan_text,['hsw_m3_h','ps_on_fraction','twas_m3_h'])
        expected=pd.date_range(origin+pd.Timedelta(hours=1),periods=24,freq='h')
        if len(plan)!=24 or not plan.index.equals(expected):raise ValueError('The plan must contain exactly 24 rows, from origin + 1 h through origin + 24 h.')
        if (plan[['hsw_m3_h','twas_m3_h']]<0).any().any() or ((plan.ps_on_fraction<0)|(plan.ps_on_fraction>1)).any():raise ValueError('Plan flows must be non-negative and PS fractions between 0 and 1.')
    calibration=json.loads((ROOT/'models'/'calibration.json').read_text(encoding='utf8'))
    rows=[]
    for h in HORIZONS:
        row={'horizon_h':h,'target_time':str(origin+pd.Timedelta(hours=h)),'persistence_m3_h':float(history.Biogas.iloc[-1]*1.69901)}
        for reg,p in [('past',None),('feed',plan)]:
            if reg=='feed' and p is None:continue
            key=f'{reg}_{h}'
            if key not in MODELS:MODELS[key]=lgb.Booster(model_str=(ROOT/'models'/f'{key}.txt').read_text(encoding='utf8'))
            m=MODELS[key]; fv=feature_row(history,h,p); x=np.array([[fv[n] for n in m.feature_name()]])
            value=float((m.predict(x,num_threads=1)[0]+history.Biogas.iloc[-1])*1.69901); q=calibration[key]['halfwidth_m3_h']
            row.update({reg+'_m3_h':value,reg+'_lower_m3_h':value-q,reg+'_upper_m3_h':value+q})
        rows.append(row)
    ranges=json.loads((ROOT/'models'/'history_bounds.json').read_text(encoding='utf8'))
    outside=[v for v in history.columns if ((history[v]<ranges[v][0])|(history[v]>ranges[v][1])).any()]
    return {'origin':str(origin),'mode':'History + supplied plan' if plan is not None else 'History only','model':'Persistence-anchored LightGBM (LGB-Δ)','interval':'Nominal 90% validation-calibrated interval','rows':rows,'out_of_training_range':outside,'history_chart':[{'timestamp':str(t),'biogas_m3_h':float(v*1.69901)} for t,v in history.Biogas.iloc[-24:].items()]}
