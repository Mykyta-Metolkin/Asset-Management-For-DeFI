import os
import pandas as pd
import numpy as np
import datetime 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import kstest, norm


warnings.filterwarnings("ignore")

# OOP
class Asset_Class:
    def __init__(self, asset_class_name, risk_factors, population_sample):
        self.asset_class_name = asset_class_name #just a string to define name
        self.risk_factors = risk_factors # pass a dict of risk_factors predefines based on asset_class_name
        self.population_sample = population_sample # even though it's named a sample, this variable is expected to contaion all the available static pairs that relate to that asset class
        #to make things simpler, the base asset class will be parsed from folders
class Asset(Asset_Class):
    def __init__(self, asset_class_name, risk_factors, population_sample, asset_pair, price, log_returns, mean_of_annual_log_return, performance, max_dd):
        self.asset_pair = asset_pair.iloc[0]
        self.price = price
        self.log_return = log_returns
        self.mean_of_annual_log_return = mean_of_annual_log_return.iloc[0]
        self.performance = performance.iloc[-1]
        self.max_dd = max_dd.iloc[0]
        super().__init__(asset_class_name, risk_factors, population_sample)
class Portfolio:
    def __init__(self, pair_perf_data, weight):
        self.pair_perf_data = pair_perf_data
        self.weight = weight
    def calc_performance(self):
        self.performance = np.sum(self.pair_perf_data.mean()*self.weight)*365
        self.vol = np.sqrt(np.dot(self.weight.T, np.dot(self.pair_perf_data.cov() * 365, self.weight)))


        
# Functions

def asset_obj_convert(pairs_data):
    asset_classes = {}
    assets = {}
    folder_list = pairs_data[0]['folder'].unique()
    pair_list = pairs_data[0]['pair'].unique()

    for folder in folder_list:
        pairs_folder_lvl_data = pairs_data[0][pairs_data[0]['folder'] == folder]
        asset_classes[folder] = Asset_Class(folder, [], pairs_folder_lvl_data['pair'].unique())
    for pair in pair_list:
        pair_lvl_data = pairs_data[0][pairs_data[0]['pair'] == pair]
        current_asset_class = asset_classes[pair_lvl_data['folder'].iloc[0]]
        assets[pair] = Asset(current_asset_class.asset_class_name, current_asset_class.risk_factors, current_asset_class.population_sample, pair_lvl_data['pair'], pair_lvl_data['price'], pair_lvl_data['log_return'], pair_lvl_data['mean_of_annual_log_returns'], pair_lvl_data['performance'], pair_lvl_data['max_dd'])
    return assets, asset_classes

def generate_portfolios(assets, pair_perf_data):
    portfolios = []
    weights = []
    max_portfolio_population = len(assets)
    portfolio_returns = []
    portfolio_vols = []
    portfolio_weights = []
    i = 0

    for _ in range(10000):  # for 10000 portfolios
        weight = np.random.random(max_portfolio_population)
        weight /= np.sum(weight)  # normalizing the weight
        weights.append(weight)
        portfolio = Portfolio(pair_perf_data, weight)
        portfolio.calc_performance()
        
        portfolios.append(portfolio)
        vol = portfolio.vol
        portfolio_returns.append(portfolio.performance)
        portfolio_vols.append(vol)
        portfolio_weights.append(portfolio.weight)
        i += 1

    portfolio_returns = np.array(portfolio_returns)
    portfolio_vols = np.array(portfolio_vols)
    portfolio_weights = np.array(portfolio_weights)

    portfolio_data = {
        'returns': portfolio_returns,
        'vols': portfolio_vols,
        'weights': portfolio_weights
    }

    sharpe_ratios = portfolio_data['returns'] / portfolio_data['vols']
    
    # Sharpe ratio maximum
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = portfolio_data['returns'][max_sharpe_idx]
    max_sharpe_vol = portfolio_data['vols'][max_sharpe_idx]

    # Volatility minimum
    min_vol_idx = np.argmin(portfolio_data['vols'])
    min_vol_return = portfolio_data['returns'][min_vol_idx]
    min_vol_vol = portfolio_data['vols'][min_vol_idx]

    portfolio_data['max_sharpe_return'] = max_sharpe_return
    portfolio_data['max_sharpe_vol'] = max_sharpe_vol
    portfolio_data['min_vol_return'] = min_vol_return
    portfolio_data['min_vol_vol'] = min_vol_vol

    # Efficient frontier calculation
    mean_returns = pair_perf_data.mean() * 365
    cov_matrix = pair_perf_data.cov() * 365
    frontier_vols, frontier_returns = calculate_efficient_frontier(mean_returns, cov_matrix)
    portfolio_data['frontier_vols'] = frontier_vols
    portfolio_data['frontier_returns'] = frontier_returns

    return portfolio_data


def calculate_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    """
    Calculate the efficient frontier by minimizing volatility for a range of target returns.
    """
    num_assets = len(mean_returns)
    results = {'volatilities': [], 'returns': []}
    
    # Generate target returns
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    target_returns = np.linspace(min_return, max_return, num_points)
    
    # Constraints and bounds
    weights_sum_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    for target_return in target_returns:
        # Target return constraint
        return_constraint = {'type': 'eq', 
                             'fun': lambda weights: np.dot(weights, mean_returns) - target_return}
        
        # Minimize portfolio volatility
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        result = minimize(portfolio_volatility, 
                          x0=np.ones(num_assets) / num_assets, 
                          constraints=[weights_sum_constraint, return_constraint], 
                          bounds=bounds)
        
        if result.success:
            results['volatilities'].append(result.fun)
            results['returns'].append(target_return)
        else:
            print(f"Optimization failed for target return: {target_return}")
    
    return results['volatilities'], results['returns']


def generate_portfolio_chart(portfolio_data):
    """
    Plot the portfolio data and efficient frontier.
    """
    print('Generating portfolio chart... ')
    plt.close("all")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of random portfolios
    plt.scatter(portfolio_data['vols'], portfolio_data['returns'], 
                c=portfolio_data['returns'] / portfolio_data['vols'], marker='o', label='Random Portfolios')
    
    # Efficient frontier
    plt.plot(portfolio_data['frontier_vols'], portfolio_data['frontier_returns'], 
             color='red', linewidth=2, label='Efficient Frontier')
    
    # Markers for min volatility and max Sharpe ratio
    plt.scatter(portfolio_data['min_vol_vol'], portfolio_data['min_vol_return'], 
                color='blue', label='Minimum Volatility', marker='X', s=100)
    plt.scatter(portfolio_data['max_sharpe_vol'], portfolio_data['max_sharpe_return'], 
                color='green', label='Maximum Sharpe Ratio', marker='X', s=100)

    # Labels, grid, and color bar
    plt.grid(True)
    plt.xlabel('Expected Historical Vol')
    plt.ylabel('Expected Historical Returns')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Portfolio Optimization and Efficient Frontier')
    plt.legend()
    plt.show()
    
def _import_data(selloff_rally):

    # Data Input logic: use Coingecko data for selloff and Dune for rally - due to shift of rally to 31.12.2022 -> 31.12.2023
    # data for 31.12.2022 - 31.12.2023 is not present in coingecko sample, coingecko has data for more pairs though
    fPATH = '../data/pa_data/'
    if selloff_rally == 'rally':
        fPATH = '../data/pa_data_new/'
    folders = ['Bitcoin','Native tokens','Stablecoins','Algo-Stables','CDP','Cross Chain','Derivatives','DEXes','Farm','Indexes','Insurance','Launchpad','Lending','Leveraged Farming','Liquid Staking','Liquid Staking - GOV','Liquidity manager','NFT','Options','Options Vault','Payments','Prediction Market','Privacy','RWA','Services','Staking Pool','Synthetics','Uncollateralized Lending','Yield','Yield Aggregator']
    aggregator = pd.DataFrame()
    for folder_name in folders:
        fPATH_folder_lvl = fPATH + folder_name
        print(fPATH_folder_lvl)
        data = []
        dirs = os.listdir(fPATH_folder_lvl+'')
        for file in dirs:
            fPATH_file_lvl = fPATH_folder_lvl + '/'+ file
            print(fPATH_file_lvl)
            if selloff_rally == 'rally':
                data = pd.read_excel(fPATH_file_lvl,  engine='openpyxl')
                data =  data.rename(columns={'date': 'snapped_at'})
                staticdata = pd.DataFrame(data) 
                staticdata = staticdata.drop(index=[0])
                staticdata['snapped_at'] = data['snapped_at']
                staticdata['price'] = data['price']
                staticdata['pair'] = file
                staticdata['folder'] = folder_name
                aggregator = pd.concat([aggregator, staticdata], ignore_index=True)
            elif selloff_rally == 'selloff':
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
    print(aggregator.info())
    return aggregator

def indicate_time_bound(aggregator,selloff_rally):
    aggregator['snapped_at']= pd.to_datetime(aggregator['snapped_at']).dt.date
    pair_list = aggregator['pair'].unique()
    for pair in pair_list:
        if selloff_rally == 'selloff':
            aggregator[aggregator['pair']==pair] = aggregator.loc[(aggregator['snapped_at'] >= datetime.date(2021,11,30)) & (aggregator['snapped_at'] <= datetime.date(2022,11,30))]
        elif selloff_rally == 'rally':
            aggregator[aggregator['pair']==pair] = aggregator.loc[(aggregator['snapped_at'] >= datetime.date(2022,12,31)) & (aggregator['snapped_at'] <= datetime.date(2023,12,31))]
        print('Date bounds for', pair, 'are set.')
    return aggregator  

def build_indexes(aggregator):
    index_list = aggregator['folder'].unique()
    pair_list = aggregator['pair'].unique()
    indexed_data = pd.DataFrame()
    #aggregator = aggregator.drop(columns=['pair'])
    aggregator[['price']] = aggregator[['price']].apply(pd.to_numeric, errors='coerce')
    summary_df = pd.DataFrame()
    pair_summary_df = pd.DataFrame()
    pair_agg_data = pd.DataFrame()
    for pair in pair_list:
        print('Calculating performance for '+pair+'...')
        aggregator_temp = aggregator[aggregator['pair'] == pair]
        aggregator_temp['price_rel_delta'] = (aggregator_temp['price']/aggregator_temp['price'].shift(1))-1
        aggregator_temp['rel_delta_stdev'] = aggregator_temp['price_rel_delta'].std()
        aggregator_temp['log_return'] = np.log(aggregator_temp['price']/aggregator_temp['price'].shift(1))
        aggregator_temp = std_perf(aggregator_temp, 'p')
        aggregator_temp['mean_of_annual_log_returns'] = aggregator_temp['log_return'][1:].mean()*365
        aggregator_temp = get_dd(aggregator_temp, 'pair')
        print('Time series rows found: ',len(aggregator_temp['snapped_at'][1:]))
        #aggregator_temp_cleaned['annual_cov'] = aggregator_temp_cleaned['log_return'].cov()*len(aggregator_temp['snapped_at'])
        pair_agg_data = pd.concat([aggregator_temp, pair_agg_data],ignore_index=False)
        pair_summary_df = pd.concat([pair_summary_df, pair_agg_data[pair_agg_data['pair']==pair].iloc[-1:]], ignore_index=True)
    pair_summary_df['CALMAR'] = (pair_summary_df['performance']-1)/abs(pair_summary_df['max_dd'])
    #pair_agg_data.to_csv('../pair_agg_data-test.csv')
    #pair_summary_df.to_csv('../pair_summary_df-test.csv')
    
    for index in index_list:
        print('Building index for '+index+'...')
        aggregator_temp = pair_agg_data[pair_agg_data['folder'] == index]
        get_corr_within_index(aggregator_temp) #creating heatmap with assets within the index 
        aggregator_temp = std_perf(aggregator_temp, 'f')
        aggregator_temp['price_rel_delta'] = (aggregator_temp['performance']/aggregator_temp['performance'].shift(1))-1
        aggregator_temp['rel_delta_stdev'] = aggregator_temp['price_rel_delta'].std()
        aggregator_temp['log_return'] = np.log(aggregator_temp['performance']/aggregator_temp['performance'].shift(1))
        aggregator_temp['avg_log_delta'] = aggregator_temp['log_return'].mean()
        aggregator_temp['avg_delta'] = aggregator_temp['price_rel_delta'].mean()
        aggregator_temp = get_dd(aggregator_temp, 'folder')
        #indexed_data = indexed_data.reset_index()
        indexed_data = pd.concat([aggregator_temp, indexed_data], ignore_index=True)
        summary_df = pd.concat([summary_df, indexed_data[indexed_data['folder']==index].iloc[-1:]], ignore_index=True) 
        #print(len(indexed_data['folder'].unique()))
    indexed_data = indexed_data[['snapped_at','folder','performance', 'log_return', 'price_rel_delta','rel_delta_stdev', 'dd', 'max_dd']]
    summary_df['CALMAR'] = (summary_df['performance']-1)/abs(summary_df['max_dd'])
    summary_df = summary_df[['snapped_at','folder','performance','avg_delta','avg_log_delta','rel_delta_stdev', 'max_dd', 'CALMAR']]
    return [indexed_data, summary_df], [pair_agg_data, pair_summary_df]

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
        indexes_corr = create_corr_df(df, 'pair', 'performance')
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

def create_corr_df(df, prmtr, perf_str):
    #transforming the indexed data first, building correl matrix next
    df = df[['snapped_at', prmtr, perf_str]]
    dates = df['snapped_at'].unique()
    indexes = df[prmtr].unique()
    df_temp = pd.DataFrame()
    df_n = pd.DataFrame()
    for idx in indexes:    
        df_temp = df[df[prmtr] == idx]  
        df_temp[perf_str+'_'+idx] = df_temp[perf_str]
        df_temp = df_temp.reset_index()  
        df_temp = df_temp[perf_str+'_'+idx]
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
    if os.path.isdir(output_path):
        pass
    else:    
        os.mkdir(output_path)
        plt.savefig(output_path+'_heatmap.png', dpi=500)

def SelloffRally(selloff_rally):
    # testing the code with equity portfolio
    stocks = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'AVGO', 'TSLA', 'GOOG', 'BRK-B', 'JPM', 'LLY', 'V', 'XOM', 'UNH']
    stock_data = pd.DataFrame()
    for stock in stocks:
        start_date = '2022-12-31'
        end_date = '2023-12-31'
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start = start_date, end = end_date)['Close']
    log_retruns_stocks = np.log(stock_data/stock_data.shift(1))

    portfolio_data = generate_portfolios(log_retruns_stocks.columns, log_retruns_stocks)

    generate_portfolio_chart(portfolio_data)

    output_path = '../output/'+selloff_rally
    index_data_list = [] #list containing output dataframes
    temp_list = [] 
    aggregated_data = _import_data(selloff_rally)
    aggregated_data = indicate_time_bound(aggregated_data, selloff_rally)
    aggregated_data = aggregated_data[aggregated_data['price'].isna()==False]
    index_data_list.append(aggregated_data)
    
    indexes_pairs_data = build_indexes(aggregated_data)
    index_data = indexes_pairs_data[0]
    pairs_data = indexes_pairs_data[1]
    pairs_data[0].to_csv(output_path+'/'+'pair_agg_data.csv')
    pairs_data[1].to_csv(output_path+'/'+'pair_summary_df.csv')
    temp_list.append(index_data)
    for item in temp_list:
        for i in item:
            index_data_list.append(i)

    print(index_data_list[1].head())
    agg_corr = create_corr_df(index_data_list[1], 'folder', 'log_return') 

    pair_perf_data = create_corr_df(pairs_data[0], 'pair', 'log_return') 

    print(len(pair_perf_data.columns))
    agg_cov_pairs = pair_perf_data.cov()*365
    assets, asset_classes = asset_obj_convert(pairs_data)
    #K-S test
    K_S_results = pd.DataFrame()
    K_S_results_temp = pd.DataFrame()
    for asset in assets:
        statistic, p_value = kstest(assets[asset].log_return.loc[1:], 'norm', args=(assets[asset].log_return.loc[1:].mean(), assets[asset].log_return.loc[1:].std()))
        assets[asset].KStatistics = statistic
        assets[asset].PValue = p_value
        print(asset, ' ', statistic, ' ', p_value)
        K_S_results_temp.at[0,'asset_class_name'] = assets[asset].asset_class_name
        K_S_results_temp.at[0,'asset_pair'] = assets[asset].asset_pair
        K_S_results_temp.at[0,'KStatistics'] = statistic
        K_S_results_temp.at[0,'PValue']  = p_value
        K_S_results = pd.concat([K_S_results, K_S_results_temp])
    K_S_results.to_csv('./K_S_results_output_'+selloff_rally+'.csv')
    pair_perf_data = pair_perf_data.loc[:, pair_perf_data.notna().sum() >= 364]
    plt.close("all")
    fig2 = plt.figure(figsize=(15, 7))
    ax2 = fig2.add_subplot(1, 1, 1)
    calculated_half =  len(pair_perf_data.columns)//2
    pair_perf_data.iloc[:,:calculated_half].hist(bins=50, ax=ax2)
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Portfolio Return Distribution')
    fig2.tight_layout()
    plt.show()

    portfolio_data = generate_portfolios(agg_corr.columns, agg_corr)
    generate_portfolio_chart(portfolio_data)
    build_histogram(agg_corr.corr(), output_path)
    create_output(output_path, index_data_list)
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