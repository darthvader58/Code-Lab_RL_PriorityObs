"""Scientific protocol tests: membership, fitted preprocessing, budgets and uncertainty."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from baselines.core import LearnedPolicy, fit_seasonal_policy, seasonal_folds
from baselines.seasonal import _weighted_ranking, scored_rows


def weather_frame():
    time=pd.date_range('2005-06-01','2006-02-01 12:00',freq='30min')
    return pd.DataFrame({'g5_month':time.strftime('%Y-%m'),'g5_step':np.arange(len(time)),
                         'x':np.arange(len(time),dtype=float),'target':np.arange(len(time))%2})


class SeasonalProtocolTests(unittest.TestCase):
    def test_folds_weather_month_buffer_and_february(self):
        df=weather_frame()
        # Deliberately misleading orbit timestamps must not determine folds.
        df['time']=pd.Timestamp('2030-01-01')
        folds=list(seasonal_folds(df))
        self.assertEqual(len(folds),8)
        for f in folds:
            tr,te=f.frames(df)
            self.assertEqual(len(f.train_months),7)
            self.assertNotIn(f.held_out_month,f.train_months)
            self.assertFalse(set(tr.index)&set(te.index))
            self.assertFalse(set(tr.g5_step)&set(f.buffer_steps))
            self.assertFalse(set(te.g5_step)&set(f.buffer_steps))
            self.assertEqual(set(te.g5_month.replace({'2006-02':'2006-01'})),{f.held_out_month})
            # Every test block is separated from training by both boundary weather steps.
            self.assertTrue(all(abs(int(a)-int(b))>=3 for a in (tr.g5_step.min(),tr.g5_step.max())
                                for b in (te.g5_step.min(),te.g5_step.max())))
        self.assertIn('2006-02',set(df.loc[folds[-1].test_index,'g5_month']))
        jan=df.index[df.g5_month=='2006-01'][-1]
        feb=df.index[df.g5_month=='2006-02'][0]
        self.assertIn(jan,folds[-1].test_index)
        self.assertIn(feb,folds[-1].test_index)
        self.assertNotIn('2006-02',{f.held_out_month for f in folds})

    def test_invalid_labels_do_not_move_boundary(self):
        df=weather_frame(); f=next(seasonal_folds(df))
        df.loc[df.g5_step.isin(f.buffer_steps),'target']=np.nan
        after=next(seasonal_folds(df))
        self.assertEqual(f.buffer_steps,after.buffer_steps)
        tr,te=after.frames(df,['target'])
        self.assertTrue(tr.target.notna().all() and te.target.notna().all())

    def test_training_only_preprocessing_and_thresholds(self):
        train=pd.DataFrame({'x':[0.,1.,np.nan,3.,4.,5.], 'target':[0,0,0,1,1,1]})
        test=pd.DataFrame({'x':[1000.,2000.,np.nan], 'target':[0,1,0]},index=[10,11,12])
        label=lambda f:f.target.to_numpy(bool)
        p,st,se=fit_seasonal_policy(LearnedPolicy('lr',['x']),train,test,label)
        imputer=p.model['pre'].named_transformers_['x']['impute']
        self.assertEqual(imputer.statistics_[0],3.)
        self.assertEqual(imputer.indicator_.features_.tolist(),[0])
        before=scored_rows(st,se,label(test),np.ones(3),(.2,))[0]
        test.target=1-test.target
        p2,st2,se2=fit_seasonal_policy(LearnedPolicy('lr',['x']),train,test,label)
        after=scored_rows(st2,se2,label(test),np.ones(3),(.2,))[0]
        self.assertEqual(before['score_threshold'],after['score_threshold'])
        self.assertEqual(before['realized_duty'],after['realized_duty'])
        self.assertEqual(before['nominal_budget'],.2)
        self.assertNotEqual(before['realized_duty'],.2)
        with self.assertRaises(ValueError):
            fit_seasonal_policy(p,train,train,label)

    def test_test_only_missingness_cannot_create_training_indicator(self):
        train=pd.DataFrame({'x':[0.,1.,2.,3.], 'target':[0,0,1,1]})
        test=pd.DataFrame({'x':[np.nan,1000.], 'target':[0,1]},index=[10,11])
        p,_,_=fit_seasonal_policy(LearnedPolicy('lr',['x']),train,test,lambda f:f.target.to_numpy(bool))
        imputer=p.model['pre'].named_transformers_['x']['impute']
        self.assertEqual(imputer.indicator_.features_.tolist(),[])
        self.assertEqual(imputer.statistics_[0],1.5)

    def test_exact_day_bootstrap_matches_physical_resampling_with_ties(self):
        y=np.array([0,1,1,0,0,1],bool)
        scores=np.array([.1,.5,.5,.8,.1,.9])
        days=np.array([0,0,1,1,2,2])
        weights=np.array([[1,1,1],[0,2,1],[2,1,0],[0,0,3]])
        auc,ap=_weighted_ranking(y,scores,days,weights)
        for k,w in enumerate(weights):
            ix=np.repeat(np.arange(len(y)),w[days])
            self.assertAlmostEqual(auc[k],roc_auc_score(y[ix],scores[ix]),places=14)
            self.assertAlmostEqual(ap[k],average_precision_score(y[ix],scores[ix]),places=14)

    def test_oracle_is_reference_and_uses_exact_rank_budget(self):
        y=np.array([0,1,1,0],bool); rain=np.array([0,2,3,0.])
        r=scored_rows(None,rain,y,rain,(.5,),oracle=True)[0]
        self.assertEqual(r['capture'],1.)
        self.assertEqual(r['realized_duty'],.5)
        self.assertEqual(r['threshold_source'],'oracle_held_out_truth_reference')


if __name__=='__main__':
    unittest.main()
