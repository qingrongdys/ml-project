# Utility Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import comb
from scipy import stats

# Weighted correlation
def weighted_corr(x, y, w):
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    var_x = np.sum(w * (x - x_mean) ** 2)
    var_y = np.sum(w * (y - y_mean) ** 2)
    
    return cov_xy / np.sqrt(var_x * var_y)

# Weighted r2
def weighted_r2(y_true, y_pred, w):
    y_mean = np.average(y_true, weights=w)
    sst = np.sum(w * (y_true - y_mean) ** 2)
    ssr = np.sum(w * (y_true - y_pred) ** 2)
    
    return 1 - ssr / sst if sst > 0 else np.nan


# Customized colors
ys_color_blues   = [plt.cm.get_cmap('Blues' )((i+1)/11) for i in range(10)]
ys_color_greens  = [plt.cm.get_cmap('Greens')((i+1)/11) for i in range(10)]
ys_color_yellows = [plt.cm.get_cmap('Wistia')((i+1)/11) for i in range(10)]
ys_color_reds    = [plt.cm.get_cmap('Reds'  )((i+1)/11) for i in range(10)]
ys_colors        = [plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in [1,2,3,4,6,0,5,7]]   # default color in matplotlib


# Matplotlib theme function
def ys_theme(ax):
    """
    Apply customized Yishu Theme to matplotlib charts.
    
    Example:
    Add the following code before plt.show()
    - ys_theme(ax)
    """
    
    # 将图例放在作图框外的右侧, 删除图例边框
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False, fontsize=9)
    
    # 设置边框和tick的灰度颜色
    gray_color = (0, 0, 0, 0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(gray_color)
    ax.tick_params(colors=gray_color)
    ax.tick_params(axis='both', which='major', labelsize=9)  # 修改字号大小
    
    # 在y轴上添加虚线
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)

    # 在x轴上添加虚线
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
    plt.tight_layout()


# Cumulative return
def ys_cum_ret(input_df,
               log_input  = False,
               log_output = True):
    
    """
    Calculate the cumulative returns from daily returns.
    
    Parameters:
    - input_df: DataFrame
        DataFrame containing a 'Date' column and daily returns for various assets.
    - log_input: bool, default False
        If True, assumes that the daily returns in input_df are in logarithmic form.
    - log_output: bool, default True
        If True, the returned cumulative returns will be in logarithmic form.

    Returns:
    - DataFrame
        DataFrame with the same columns as input_df, but with daily cumulative returns.
    """
    
    # set 'Date' as index
    if input_df.index.name != 'Date':
        df = input_df.set_index('Date')
    else:
        df = input_df
    
    # log input
    if not log_input:
        df = np.log(df + 1)
    
    # output dataframe
    if log_output:
        df = df.cumsum()
    else:
        df = (1 + df).cumprod() - 1
    
    # 假如日期列不是index, 返回的dataframe日期列也不是index
    if input_df.index.name != 'Date':
        df = df.reset_index()
    
    return df


# Cumulative return chart
def ys_cum_ret_chart(input_df, 
                     log_input      = False, 
                     log_output     = True,
                     title_input    = 'Cumulative Returns', 
                     subtitle_input = f'{datetime.now().strftime("%Y-%m-%d")} by Yishu Dai', 
                     xlabel         = '', 
                     ylabel         = '',
                     colors_input   = ys_color_blues + ys_colors,
                     fig_size       = (10, 6) ):
    
    """
    Plot a chart of cumulative returns based on daily returns.

    Parameters:
    -----------
    input_df : DataFrame. With 'Date' as column or index and daily returns for various assets.
    log_input : bool, default False
        Whether input logarithmic daily return.
    log_output : bool, default True
        Whether the returned cumulative returns will be in logarithmic form.
    title_input : str, default 'Cumulative Returns'.
        Title for the plot.
    subtitle_input : str, default '%Y-%m-%d by Yishu Dai'.
        Subtitle for the plot, displayed just below the main title.
    xlabel : str, default 'Date'
        Label for the x-axis.
    ylabel : str, default 'Cumulative Return'
        Label for the y-axis.
    colors_input : list, optional
        A list of colors to use for plotting each asset's returns. 
    fig_size : tuple, default (10, 6)

    Returns:
    fig, ax
    """
    
    # 创建图表, 并设置图面大小
    fig, ax = plt.subplots(figsize = fig_size)
    
    # 获取每日累计收益率
    cumulative_returns = ys_cum_ret(input_df, log_input=log_input, log_output=log_output)
    
    # 假如日期列是index, 取消索引
    if input_df.index.name == 'Date':
        cumulative_returns = cumulative_returns.reset_index()
    
    
    # 绘制每个资产的累计收益率
    for idx, column in enumerate(cumulative_returns.columns):
        if column != 'Date':
            ax.plot(cumulative_returns['Date'], cumulative_returns[column], label=column, color=colors_input[(idx-1) % len(colors_input)])
    
    # 添加标题和子标题
    ax.set_title(title_input, fontsize=12, ha='center', y=1.05)
    subtitle_y_pos = 1.02  # 调整y值以确保子标题出现在主标题下方
    ax.text(0.5, subtitle_y_pos, subtitle_input, transform=ax.transAxes, ha='center', fontsize=10)
    
    # 设置x和y轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ys_theme(ax)

    return fig, ax


# Calculate max drawdown for SINGLE column input
def ys_max_drawdown( input_df,
                     column_input,
                     cum_input  = False,
                     log_input  = False,
                     log_output = True ):
    """
    Calculate the max drawdown for a SINGLE column in a dataframe with 'Date' column.
    
    Parameters:
    - input_df: input dataframe, 'Date' as column or index
    - column_input: column name
    - cum_input: bool, whether cumulative return or not.
    - log_input: bool, whether logarithmic input or not.
    - log_output: bool, whether logarithmic output or not.
    
    Return a SINGLE number.
    """
    
    # If 'Date' column is not index, set it as index
    if input_df.index.name == 'Date':
        series = input_df[column_input]
    else:
        series = input_df.set_index('Date')[column_input]
    
    # Log input
    if not log_input:
        series = np.log(series + 1)
    
    # Calculate cumulative return
    if cum_input:
        cumulative_return = series
    else:
        cumulative_return = series.cumsum()
    
    # Calculate max drawdown
    max_return = cumulative_return.cummax()
    drawdown = (cumulative_return - max_return).min()
    
    # Convert to simple form if output is not log
    if not log_output:
        drawdown = np.exp(drawdown) - 1
    
    return drawdown


# Calculate annualized return for SINGLE column input
def ys_ann_ret( input_df,
                column_input,
                cum_input  = False,
                log_input  = False,
                log_output = True ):
    """
    Calculate the annualized return for a SINGLE DAILY return column in a dataframe with 'Date' column.
    
    Parameters:
    - input_df: input dataframe, 'Date' as column or index
    - column_input: column name
    - cum_input: bool, whether cumulative return or not.
    - log_input: bool, whether logarithmic input or not.
    - log_output: bool, whether logarithmic output or not.
    
    Return a SINGLE number.
    """
    
    # If 'Date' column is not index, set it as index
    if input_df.index.name != 'Date':
        series = input_df.set_index('Date')[column_input]
    else:
        series = input_df[column_input]

    # Calculate the total period in years
    total_period = (series.index[-1] - series.index[0]).days / 365
    
    # If not cumulative, sum/prod the return
    if not cum_input:
        if not log_input:
            annual_return = np.log(series + 1).sum() / total_period
        else:
            annual_return = series.sum() / total_period
    else:
        if not log_input:
            annual_return = ( np.log(1 + series.iloc[-1]) - np.log(1 + series.iloc[0]) ) / total_period
        else:
            annual_return = (series.iloc[-1] - series.iloc[0]) / total_period

    # If output not in log form, convert it
    if not log_output:
        annual_return = np.exp(annual_return) - 1

    return annual_return

# Calculate annualized volatility
def ys_ann_vola( input_df,
                 column_input,
                 log_input  = False ):
    """
    Calculate the annualized volatility for a SINGLE column in a dataframe with 'Date' column.
    Only input daily returns. If not, convert by multiplying sqrt(T) as you want.
    
    Parameters:
    - input_df: input dataframe, 'Date' as column or index.
    - column_input: column name
    - log_input: bool, whether logarithmic input or not.
    
    Return a SINGLE number.
    """
    
    # If 'Date' column is not index, set it as index
    if input_df.index.name != 'Date':
        series = input_df.set_index('Date')[column_input]
    else:
        series = input_df[column_input]
    
    # Convert the series into log returns if necessary
    if not log_input:
        vola = np.log(1 + series).std() * np.sqrt(252)
    else:
        vola = series.std() * np.sqrt(252)
    
    return vola

# Calculate Sharpe Ratio
def ys_sharpe_ratio( input_df,
                     column_input,
                     risk_free  = 0,
                     log_input  = False,
                     log_output = True ):
    """
    Calculate Sharpe ratio for a SINGLE column in a dataframe with 'Date' column.
    Only input daily returns.
    
    Parameters:
    - input_df: input dataframe, 'Date' as column or index.
    - column_input: column name
    - risk_free: risk free rate, default 0
    - log_input: bool, whether logarithmic input or not.
    - log_output: bool, whether logarithmic output or not.
    
    Return a SINGLE number.
    """
    
    return (ys_ann_ret(  input_df,
                         column_input,
                         cum_input  = False,
                         log_input  = log_input,
                         log_output = log_output ) - risk_free) / \
            ys_ann_vola( input_df,
                         column_input,
                         log_input  = log_input )

# Performance Matrix
def ys_performance_matrix( input_df,
                           risk_free  = 0,
                           log_input  = False,
                           log_output = True,
                           in_percent = True,
                           rounding   = 2 ):
    """
    Calculate Performance Matrix for a dataframe with 'Date' column and several different DAILY return columns.
    Include Annualized Returns, Max Drawdown, Annualized Volatility, and Sharpe Ratio.
    
    Parameters:
    - input_df: input dataframe, 'Date' as column or index.
    - risk_free: risk free rate, default 0
    - log_input: bool, whether logarithmic input or not.
    - log_output: bool, whether logarithmic output or not.
    - in_percent: bool, whether output in percent.
    - rounding: int, default 2. Output rounding.
    
    Return a Dataframe.
    """
    result = []

    for column in input_df.columns:
        if column != 'Date':
            result.append([
                column,
                ys_ann_ret(      input_df, column,
                                 log_input  = log_input,
                                 log_output = log_output,
                                 cum_input  = False ),
                ys_max_drawdown( input_df, column,
                                 log_input  = log_input,
                                 log_output = log_output,
                                 cum_input  = False ),
                ys_ann_vola(     input_df, column,
                                 log_input  = log_input ),
                ys_sharpe_ratio( input_df, column,
                                 risk_free  = risk_free,
                                 log_input  = log_input,
                                 log_output = log_output )
            ])
    
    result_df = pd.DataFrame(result, columns=['Column', 'Ann. Return', 'Max Drawdown', 'Ann. Volatility', 'Sharpe Ratio'])
    
    if in_percent:
        result_df['Ann. Return']     = result_df['Ann. Return'].apply(lambda x: f"{round(x * 100, rounding)}%")
        result_df['Max Drawdown']    = result_df['Max Drawdown'].apply(lambda x: f"{round(x * 100, rounding)}%")
        result_df['Ann. Volatility'] = result_df['Ann. Volatility'].apply(lambda x: f"{round(x * 100, rounding)}%")
    else:
        result_df['Ann. Return']     = result_df['Ann. Return'].round(rounding)
        result_df['Max Drawdown']    = result_df['Max Drawdown'].round(rounding)
        result_df['Ann. Volatility'] = result_df['Ann. Volatility'].round(rounding)

    result_df['Sharpe Ratio'] = result_df['Sharpe Ratio'].round(rounding)
    result_df.set_index('Column', inplace = True)
    
    return result_df


# assign factor quantile groups, weighted
def factor_quantile_weighted(input_df, factor_column, weight_column, n_quantile=10):
    """
    Calculate factor quantile group for every day
    """
    if 'Date' not in input_df.columns or 'Id' not in input_df.columns:
        raise ValueError("Input DataFrame must contain 'Date' and 'Id' columns.")

    input_df = input_df.sort_values(['Date', factor_column])
    
    def calculate_weighted_quantiles(group):
        group = group.sort_values(factor_column).copy()
        total_weight = group[weight_column].sum()
        bin_size = total_weight / n_quantile
        # Use cumsum to calculate quantile group
        group['cumulative_weight'] = group[weight_column].cumsum()
        # Calculate quantile group
        group[f'{factor_column}_quantile'] = np.floor(group['cumulative_weight'] / bin_size).astype(int)
        group[f'{factor_column}_quantile'] = group[f'{factor_column}_quantile'].clip(upper=n_quantile-1)
        # Clear helping column
        group.drop(columns='cumulative_weight', inplace=True)
        return group

    # Calculate weighted quantile group for every day
    output_df = (
        input_df
        .groupby('Date', group_keys=False)
        .apply(calculate_weighted_quantiles)
        .reset_index(drop=True)
    )
    
    # Sort by date and id
    output_df['Date'] = pd.to_datetime(output_df['Date'])
    output_df = output_df.sort_values(['Date', 'Id']).reset_index(drop=True)

    return output_df


# Calculate weighted factor returns for every quantile group
def factor_return_weighted(quantile_df, factor_column, weight_column, n_quantile=10):
    """
    Calculate weighted returns for every quantile
    """
    quantile_column = f"{factor_column}_quantile"

    if quantile_column not in quantile_df.columns or 'y' not in quantile_df.columns or 'Date' not in quantile_df.columns:
        raise ValueError(f"Columns {quantile_column} and 'y' and 'Date' must exist in the input DataFrame.")

    if weight_column not in quantile_df.columns:
        raise ValueError(f"Weight column {weight_column} must exist in the input DataFrame.")

    result = (
        quantile_df.groupby(['Date', quantile_column])
        .apply(lambda x: np.average(x['y'], weights=x[weight_column]))
        .unstack(fill_value=0)
    ).reindex(columns=range(n_quantile)).reset_index()
    
    result['19LS'] = result[9] - result[0]
    result['33LS'] = result[9] + result[8] + result[7] - result[0] - result[1] - result[2]
    
    return result