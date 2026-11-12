"""Pasta & Co first-draft inventory model.
Assumptions: Daily orders are 3 Rustic, 8 Demi, 1 Como; missing waste rows mean 0;
9/30/26 second full set of waste entries is a suspected duplicate; recorded sales
on sellout days are lower bounds for actual demand (we explore k extra units).
"""
import pandas as pd
import numpy as np

start = '2026-09-13'; end = '2026-10-13'
Q0 = {'Rustic':3,'Demi':8,'Como':1}
COST = {'Rustic':2.49,'Demi':0.97,'Como':3.45}
# Entries transcribed from photos of pages 1-12 (9/13 - 10/13).
# Key: date -> (Rustic, Demi, Como) waste units.
observed_waste = {
 '2026-09-13':(2,0,1), '2026-09-14':(0,2,1), '2026-09-15':(2,0,1),
 '2026-09-16':(2,0,0), '2026-09-17':(1,0,0), '2026-09-18':(0,1,1),
 '2026-09-19':(2,0,0), '2026-09-20':(3,0,0),
 '2026-09-23':(1,6,0), '2026-09-24':(0,0,1),
 '2026-09-27':(0,0,1), '2026-09-28':(2,1,0), '2026-09-29':(3,5,1),
 '2026-09-30':(2,2,0), '2026-10-01':(2,5,1), '2026-10-06':(0,1,0),
 '2026-10-07':(2,3,1), '2026-10-08':(0,2,0), '2026-10-09':(2,0,0),
 '2026-10-11':(0,5,0), '2026-10-12':(3,2,1), '2026-10-13':(3,0,1),
}
rows=[]
for date in pd.date_range(start,end):
    counts=observed_waste.get(str(date.date()),(0,0,0))
    for b,w in zip(Q0,counts):
        rows.append({'date':date,'weekday':date.day_name(),'bread':b,'current_order':Q0[b],
                     'waste':w, 'estimated_sales':Q0[b]-w, 'wholesale_cost':COST[b],
                     'logged_waste':date.date().isoformat() in observed_waste})
df=pd.DataFrame(rows)
if (df.estimated_sales<0).any(): print('WARNING: Negative sales under order assumption!')

# Minimum integer empirical 80%-quantile of demand distribution
# assuming censored sales are equal to true demand (optimistic, lower-bound scenario).
def minimum_order(values, service=.8):
    vals=sorted(values)
    return int(vals[int(np.ceil(service*len(vals)))-1])

def make_policies(sellout_extra=0, service=.8):
    d=df.copy()
    d['assumed_demand'] = d['estimated_sales'] + np.where(d.waste == 0, sellout_extra, 0)
    fixed = d.groupby('bread').assumed_demand.apply(lambda x:minimum_order(x,service)).to_dict()
    by_day = d.groupby(['bread','weekday']).assumed_demand.apply(lambda x:minimum_order(x,service)).to_dict()
    d['current_Q']=d['current_order']
    d['fixed_Q']=d['bread'].map(fixed)
    d['weekday_Q']=[by_day[(b,w)] for b,w in zip(d.bread,d.weekday)]
    def metric(d,col):
        leftover=np.maximum(d[col]-d.assumed_demand,0)
        shortage=np.maximum(d.assumed_demand-d[col],0)
        return {'daily_waste_units':leftover.sum()/31,
                'month_waste_units':leftover.sum(),
                'month_waste_cost':float((leftover*d.wholesale_cost).sum()),
                'fulfilled_bread_days_pct':float(np.mean(d.assumed_demand<=d[col])*100),
                'days_with_unmet_demand_pct':float(np.mean(shortage>0)*100),
                'unmet_units_month':int(shortage.sum())}
    return fixed,by_day,pd.DataFrame({p:metric(d,p+'_Q') for p in ['current','fixed','weekday']}).T,d

if __name__=='__main__':
    df.to_csv('/mnt/data/pasta_bread_daily_data.csv', index=False)
    totals = df.groupby('bread').agg(wasted=('waste','sum'), avg_waste=('waste','mean'))
    totals['waste_cost']=[totals.loc[b,'wasted']*COST[b] for b in totals.index]
    print('TOTALS\n',totals, '\n')
    print('BY WEEKDAY (total all bread products, average per weekday)\n',
        df.groupby(['weekday','date']).waste.sum().groupby('weekday').mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']), '\n')
    for extra in [0,1,2]:
        fixed,by_day,summary,d=make_policies(extra)
        print('\n==== EXTRA DEMAND ON SELL-OUT DAYS:', extra,'====')
        print('Fixed:',fixed)
        print('Weekday:')
        print(pd.Series(by_day).unstack(0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))
        print('Metrics:',summary.round(2).to_string())
    fixed,by_day,summary,d=make_policies(0)
    summary.to_csv('/mnt/data/pasta_bread_results_low_demand.csv')
    pd.Series(by_day).unstack(0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).to_csv('/mnt/data/pasta_bread_ordering_schedule_low_demand.csv')
