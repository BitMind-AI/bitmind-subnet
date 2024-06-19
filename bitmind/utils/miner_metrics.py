from collections import defaultdict
import pandas as pd
import numpy as np


def miner_metrics(results_df, transpose=True):
    print(f'Miner Statistics | {results_df.time.min()} - {results_df.time.max()}')
    
    miner_preds_df = results_df.groupby('miner_uid')[['label', 'pred']].agg(list)
    metrics_df = compute_metrics(miner_preds_df)
    
    image_sources = results_df.image_source.unique()
    for image_source in image_sources:
        image_source_df = results_df[results_df.image_source == image_source]
        miner_preds_df = image_source_df.groupby('miner_uid')[['label', 'pred']].agg(list)
        if miner_preds_df.shape[0] == 0:
            continue
        perf_df = compute_metrics(miner_preds_df)[['miner_uid', 'accuracy', 'num_predictions']]
        perf_df = perf_df.rename({
            'accuracy': f'{image_source} | accuracy',
            'num_predictions': f'{image_source} | num_preds'
        }, axis=1)
        metrics_df = metrics_df.merge(perf_df, on='miner_uid')

    if transpose:
        metrics_df = metrics_df.T
        metrics_df.columns = metrics_df.loc['miner_uid'].astype(int)
        metrics_df = metrics_df[1:]
        for i, metric in enumerate(metrics_df.index):
            if 'num_pred' in metric:
                metrics_df.loc[metric] = metrics_df.loc[metric].apply(int)  # TODO why doesn't this work

    return metrics_df


def compute_metrics(miner_preds_df):
    perf_data = defaultdict(list)
    perf_data['miner_uid'] = miner_preds_df.index.unique()
    for uid in perf_data['miner_uid']:
        miner_row = miner_preds_df[miner_preds_df.index == uid]
        if miner_row.shape[0] == 0:
            perf_data['accuracy'].append(0)
            perf_data['precision'].append(0)
            perf_data['recall'].append(0)
            perf_data['f-1'].append(0)
            perf_data['num_predictions'].append(0)
        else:
            labels = np.array(miner_row.label.tolist()[0])
            preds = np.array(miner_row.pred.tolist()[0])
            tp = sum(preds[labels==1] > 0.5)
            fp = sum(preds[labels==0] > 0.5)
            tn = sum(preds[labels==0] <= 0.5)
            fn = sum(preds[labels==1] <= 0.5)
        
            prec = tp / (tp + fp + 1e-5)
            rec = tp / (tp + fn + 1e-5)
            perf_data['accuracy'].append((tp + tn) / len(labels))
            perf_data['precision'].append(prec)
            perf_data['recall'].append(rec)
            perf_data['f-1'].append((2 * prec * rec)/(prec + rec + 1e-5))
            perf_data['num_predictions'].append(len(labels))
        
    return pd.DataFrame(perf_data)