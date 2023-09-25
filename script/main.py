import os
import pandas as pd
import numpy as np
import datetime 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def _import_data():

    fPATH = '../data/Token_PA_Data/'
    folders = ['Bitcoin','Native tokens','Stablecoins','Algo-Stables','CDP','Cross Chain','Derivatives','DEXes','Farm','Indexes','Insurance','Launchpad','Lending','Leveraged Farming','Liquid Staking','Liquid Staking - GOV','Liquidity manager','NFT','Options','Options Vault','Payments','Prediction Market','Privacy','RWA','Services','Staking Pool','Synthetics','Uncollateralized Lending','Yield','Yield Aggregator']
    aggregator = pd.DataFrame()
    for folder_name in folders:
        fPATH_folder_lvl = fPATH + folder_name
        print(fPATH_folder_lvl)
        data = []
        dirs = os.listdir(fPATH_folder_lvl)
        for file in dirs:
            fPATH_file_lvl = fPATH_folder_lvl + '/'+ file
            print(fPATH_file_lvl)
            with open(fPATH_file_lvl) as fp:
                xls_received = fp.readlines()
                data = []
                for line in xls_received:
                    data.append(line.replace('\n', '').split(','))
            staticdata = pd.DataFrame(data) 
            staticdata.columns=staticdata.iloc[0]
            staticdata = staticdata.drop(index=[0])
            staticdata['pair'] = file
            staticdata['folder'] = folder_name
            aggregator = pd.concat([aggregator, staticdata], ignore_index=True)
    return aggregator

def indicate_time_bound(aggregator,selloff_rally):
    aggregator['snapped_at']= pd.to_datetime(aggregator['snapped_at']).dt.date
    pair_list = aggregator['pair'].unique()
    for pair in pair_list:
        if selloff_rally == 'selloff':
            aggregator[aggregator['pair']==pair] = aggregator.loc[(aggregator['snapped_at'] >= datetime.date(2021,11,30)) & (aggregator['snapped_at'] <= datetime.date(2022,11,30))]
        elif selloff_rally == 'rally':
            aggregator[aggregator['pair']==pair] = aggregator.loc[(aggregator['snapped_at'] >= datetime.date(2019,12,31)) & (aggregator['snapped_at'] <= datetime.date(2020,12,31))]
        print('Date bounds for', pair, 'are set.')
    return aggregator  

def build_indexes(aggregator):
    index_list = aggregator['folder'].unique()
    pair_list = aggregator['pair'].unique()
    indexed_data = pd.DataFrame()
    #aggregator = aggregator.drop(columns=['pair'])
    aggregator[['price','market_cap','total_volume']] = aggregator[['price','market_cap','total_volume']].apply(pd.to_numeric, errors='coerce')
    summary_df = pd.DataFrame()
    pair_agg_data = pd.DataFrame()
    for pair in pair_list:
        print('Calculating performance for '+pair+'...')
        aggregator_temp = aggregator[aggregator['pair'] == pair]
        aggregator_temp['price_rel_delta'] = (aggregator_temp['price']/aggregator_temp['price'].shift(1))-1
        aggregator_temp['rel_delta_stdev'] = aggregator_temp['price_rel_delta'].std()
        aggregator_temp = std_perf(aggregator_temp, 'p')
        pair_agg_data = pd.concat([aggregator_temp, pair_agg_data],ignore_index=False)
    for index in index_list:
        print('Building index for '+index+'...')
        aggregator_temp = pair_agg_data[pair_agg_data['folder'] == index]
        get_corr_within_index(aggregator_temp) #creating heatmap with assets within the index 
        aggregator_temp = std_perf(aggregator_temp, 'f')
        aggregator_temp['price_rel_delta'] = (aggregator_temp['performance']/aggregator_temp['performance'].shift(1))-1
        aggregator_temp['rel_delta_stdev'] = aggregator_temp['price_rel_delta'].std()
        aggregator_temp['avg_delta'] = aggregator_temp['price_rel_delta'].mean()
        aggregator_temp = get_dd(aggregator_temp, 'folder')
        #indexed_data = indexed_data.reset_index()
        indexed_data = pd.concat([aggregator_temp, indexed_data], ignore_index=True)
        summary_df = pd.concat([summary_df, indexed_data[indexed_data['folder']==index].iloc[-1:]], ignore_index=True) 
        #print(len(indexed_data['folder'].unique()))
    indexed_data = indexed_data[['snapped_at','folder','performance','price_rel_delta','rel_delta_stdev', 'dd', 'max_dd']]
    summary_df['CALMAR'] = (summary_df['performance']-1)/abs(summary_df['max_dd'])
    summary_df = summary_df[['snapped_at','folder','performance','avg_delta','rel_delta_stdev', 'max_dd', 'CALMAR']]
    return [indexed_data, summary_df]

def std_perf(df, mode):
    df_temp = pd.DataFrame()
    df_n = pd.DataFrame()
    df_temp = df
    
    if mode == 'f':
        i = 0
        df_temp['snapped_at']
        date_list = []        
        for date in df['snapped_at']:
            if date in date_list:
                pass
            else:            
                date_list.append(date)
                df_temp = df[df['snapped_at']==date]
                p_count = len(df_temp['pair'].unique())
                weight = 1/p_count
                df_temp['performance'] = (df_temp['performance']*weight)
                df_n = pd.concat([df_n, df_temp],ignore_index=True)
            i+=1
        df_n = df_n.groupby(['snapped_at','folder'], as_index=False)['performance'].sum()
        i = 0
        df_n['dd'] = 0
        for row in df_n.iterrows():
            if i == 0:
                pass
            else: 
                df_n['dd'].iloc[i] = (df_n['performance'].iloc[i]/max(df_n['performance'].iloc[0:(i+1)]))-1
            i+=1
        return df_n
    elif mode == 'p':
        i = 0
        perf = 1
        df['performance'] = 0
        df['dd'] = 0
        for row in df.iterrows():    
            if pd.isna(df['price_rel_delta'].iloc[i]):
                pass
            else:
                perf = perf*(1+df['price_rel_delta'].iloc[i])
            df['performance'].iloc[i] = perf
            if i == 0:
                pass
            else: 
                df['dd'].iloc[i] = (df['performance'].iloc[i]/max(df['performance'].iloc[0:(i+1)]))-1
            i+=1 
    return df


def get_dd(df, prameter):
    prms = df[prameter].unique()
    n_df = pd.DataFrame()
    for prmtr in prms:
        df_temp = df[df[prameter]==prmtr]
        df_temp['max_dd'] = min(df_temp['dd'])
        n_df= pd.concat([n_df, df_temp], ignore_index=True)
    return n_df

def get_corr_within_index(df):
    indexes = df['folder'].unique() 
    for index in indexes:
        print('Calculating correlation within index '+index+'...')
        output_path = '../output/heatmaps/'+index+'/'
        indexes_corr = create_corr_df(df, 'pair')
        build_histogram(indexes_corr.corr(), output_path)

def create_output(o_path,aggregated_data_list):
    if  os.path.isdir(o_path):
        pass
    else:    
        os.mkdir(o_path)
    i = 1
    for d_list in aggregated_data_list:
        d_list.to_csv(o_path+'/agg_output'+str(i)+'.csv')
        i+=1
    print('Output successfully generated in ', o_path)

def create_corr_df(df, prmtr):
    #transforming the indexed data first, building correl matrix next
    df = df[['snapped_at', prmtr, 'performance']]
    dates = df['snapped_at'].unique()
    indexes = df[prmtr].unique()
    df_temp = pd.DataFrame()
    df_n = pd.DataFrame()
    for idx in indexes:    
        df_temp = df[df[prmtr] == idx]  
        df_temp['performance_'+idx] = df_temp['performance']
        df_temp = df_temp.reset_index()  
        df_temp = df_temp['performance_'+idx]
        df_n[idx] = df_temp  
        
    corr_df = df_n
    print('Correlation matrix has been generated.')
    return corr_df

def build_histogram(corr_df, output_path):
    plt.figure(figsize=(18, 18))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    annot_kws={'fontsize':8, 
           'fontstyle':'normal',  
           #'color':"w",
           'alpha':0.75, 
           'rotation':"horizontal",
           'verticalalignment':'center',
           #'backgroundcolor':'w'
           }
    heatmap = sns.heatmap(corr_df, vmin=-1, vmax=1, annot=True, mask = mask, annot_kws = annot_kws) 
    
    heatmap.tick_params(length=0)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=14)
    i=1
    if os.path.isdir('../output/heatmaps/'):
        pass
    else:    
        os.mkdir('../output/heatmaps/')
    while  os.path.isdir(output_path):
        i +=1
        output_path = output_path+str(i) 
    else:    
        os.mkdir(output_path)
        plt.savefig(output_path+'_heatmap.png', dpi=500)

def SelloffRally(selloff_rally):
    output_path = '../output/'+selloff_rally
    aggregated_data_list = [] #list containing output dataframes
    temp_list = [] 
    aggregated_data = _import_data()
    aggregated_data = indicate_time_bound(aggregated_data, selloff_rally)
    aggregated_data = aggregated_data[aggregated_data['price'].isna()==False]
    aggregated_data_list.append(aggregated_data)
    temp_list.append(build_indexes(aggregated_data))
    for item in temp_list:
        for i in item:
            aggregated_data_list.append(i)
    agg_corr = create_corr_df(aggregated_data_list[1], 'folder')    
    build_histogram(agg_corr.corr(), output_path)
    create_output(output_path, aggregated_data_list)
    for column in agg_corr.columns:
        plt.plot(agg_corr[column])

def main():
    print('Sell off scenraio initiated')
    SelloffRally('selloff')
    print('Rally scenraio initiated')
    SelloffRally('rally')
    #print(aggregated_data.head())
    print('All output has been successfully generated. You can close the window.')
    return 0

main()