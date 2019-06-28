import sys
import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats

from lifetimes.utils import calculate_alive_path, expected_cumulative_transactions

glovo_green = '#26A69A'
glovo_yellow = '#FFD335'
glovo_gray = '#666666'

__all__ = [
    'plot_period_transactions',
    'plot_calibration_purchases_vs_holdout_purchases',
    'plot_frequency_recency_matrix',
    'plot_probability_alive_matrix',
    'plot_expected_repeat_purchases',
    'plot_history_alive',
    'plot_cumulative_transactions',
    'plot_incremental_transactions',
    'plot_transaction_rate_heterogeneity',
    'plot_dropout_rate_heterogeneity'
]


def coalesce(*args):
    return next(s for s in args if s is not None)


def plot_period_transactions(model,
                             max_frequency=7,
                             title='Frequency of Repeat Transactions',
                             xlabel='Number of Calibration Period Transactions',
                             ylabel='Customers',
                             **kwargs):
    """
    Plot a figure with period actual and predicted transactions.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    max_frequency: int, optional
        The maximum frequency to plot.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt
    labels = kwargs.pop('label', ['Actual', 'Model'])

    n = model.data.shape[0]
    simulated_data = model.generate_new_data(size=n)

    model_counts = pd.DataFrame(model.data['frequency'].value_counts().sort_index().iloc[:max_frequency])
    simulated_counts = pd.DataFrame(simulated_data['frequency'].value_counts().sort_index().iloc[:max_frequency])
    combined_counts = model_counts.merge(simulated_counts, how='outer', left_index=True, right_index=True).fillna(0)
    combined_counts.columns = labels

    ax = combined_counts.plot(kind='bar', **kwargs)

    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return ax


def plot_calibration_purchases_vs_holdout_purchases(model,
                                                    calibration_holdout_matrix,
                                                    kind="frequency_cal",
                                                    n=7,
                                                    **kwargs):
    """
    Plot calibration purchases vs holdout.

    This currently relies too much on the lifetimes.util calibration_and_holdout_data function.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    calibration_holdout_matrix: pandas DataFrame
        Dataframe from calibration_and_holdout_data function.
    kind: str, optional
        x-axis :"frequency_cal". Purchases in calibration period,
                 "recency_cal". Age of customer at last purchase,
                 "T_cal". Age of customer at the end of calibration period,
                 "time_since_last_purchase". Time since user made last purchase
    n: int, optional
        Number of ticks on the x axis
    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    x_labels = {
        "frequency_cal": "Purchases in calibration period",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase"
    }
    summary = calibration_holdout_matrix.copy()
    duration_holdout = summary.iloc[0]['duration_holdout']

    summary['model_predictions'] = summary.apply(lambda r: model.conditional_expected_number_of_purchases_up_to_time(duration_holdout, r['frequency_cal'], r['recency_cal'], r['T_cal']), axis=1)

    if kind == "time_since_last_purchase":
        summary["time_since_last_purchase"] = summary["T_cal"] - summary["recency_cal"]
        ax = summary.groupby(["time_since_last_purchase"])[['frequency_holdout', 'model_predictions']].mean().iloc[:n].plot(**kwargs)
    else:
        ax = summary.groupby(kind)[['frequency_holdout', 'model_predictions']].mean().iloc[:n].plot(**kwargs)

    plt.title('Actual Purchases in Holdout Period vs Predicted Purchases')
    plt.xlabel(x_labels[kind])
    plt.ylabel('Average of Purchases in Holdout Period')
    plt.legend()

    return ax


def plot_frequency_recency_population(summary_data,
                                      max_frequency=None,
                                      max_recency=None,
                                      title=None,
                                      xlabel="Customer's Historical Frequency",
                                      ylabel="Customer's Recency",
                                      ax=None,
                                      **kwargs):
    """
    Plot distribution of customers in a recency frequecy matrix as heatmap.

    Parameters
    ----------
    summary_data: RF dataframe
        Dataframe containing recency-frequency feature vectors for all users.
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the customer.
        Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if max_frequency is None:
        max_frequency = int(summary_data['frequency_cal'].max())

    if max_recency is None:
        max_recency = int(summary_data['T_cal'].max())

    population_matrix = summary_data.groupby(['frequency_cal','recency_cal']
                                   )['T_cal'].count(
                                   ).reset_index(
                                   ).rename(columns={'T_cal':'num_customers'})
    population_matrix['num_customers'] = np.log10(population_matrix['num_customers'])
    Z = pd.merge(pd.DataFrame(np.transpose([list(range(max_recency)), [1]*max_recency]),
                              columns=['recency_cal','dummy']),
                 pd.DataFrame(np.transpose([list(range(max_frequency)), [1]*max_frequency]),
                              columns=['frequency_cal','dummy']),
                 on='dummy')
    Z.drop('dummy',1,inplace=True)
    Z = pd.merge(Z, population_matrix, on=['recency_cal','frequency_cal'], how='left')
    # Z.fillna(0,inplace=True)
    # Z.loc[(Z['frequency_cal']==0)&(Z['recency_cal']==0),'num_customers'] = 0
    interpolation = kwargs.pop('interpolation', 'none')

    if ax is None:
        ax = plt.subplot(111)
    PCM = ax.imshow(Z.pivot(index='recency_cal',
                            columns='frequency_cal',
                            values='num_customers').values,
                            interpolation=interpolation, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = 'Number of Customers (log)\nby Frequency and Recency'
    ax.set_title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    cb = plt.colorbar(PCM, ax=ax)
    cb.set_ticklabels([str(round(np.power(10,x),1)) for x in cb.get_ticks()])
    cb.set_label('Number of customers')

    return ax


def plot_frequency_recency_matrix(model,
                                  T=1,
                                  max_frequency=None,
                                  max_recency=None,
                                  title=None,
                                  xlabel="Customer's Historical Frequency",
                                  ylabel="Customer's Recency",
                                  ax=None,
                                  **kwargs):
    """
    Plot recency frequecy matrix as heatmap.

    Plot a figure of expected transactions in T next units of time
    by a customer's frequency and recency.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    T: fload, optional
        Next units of time to make predictions for
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the
        customer. Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if max_frequency is None:
        max_frequency = int(model.data['frequency'].max())

    if max_recency is None:
        max_recency = int(model.data['T'].max())

    Z = np.zeros((max_recency + 1, max_frequency + 1))
    for i, recency in enumerate(np.arange(max_recency + 1)):
        for j, frequency in enumerate(np.arange(max_frequency + 1)):
            Z[i, j] = model.conditional_expected_number_of_purchases_up_to_time(
                T, frequency, recency, max_recency)

    interpolation = kwargs.pop('interpolation', 'none')

    if ax is None:
        ax = plt.subplot(111)
    PCM = ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = 'Expected Number of Future Purchases for {} Unit{} of Time,'. \
            format(T, "s"[T == 1:]) + '\nby Frequency and Recency of a Customer'
    plt.title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(PCM, ax=ax)

    return ax


def plot_rfm_matrix(model,
                    rfm_data,
                    horizon=1,
                    T=None,
                    revenue_type='gross',
                    max_frequency=None,
                    max_recency=None,
                    title=None,
                    xlabel="Customer's Historical Frequency",
                    ylabel="Customer's Recency",
                    ax=None,
                    log=True,
                    **kwargs):
    """
    Plot recency frequecy matrix as heatmap, with color indicating RLTV
    per customer.

    Plot a figure of expected transactions in T next units of time
    by a customer's frequency and recency.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    horizon: float, optional
        Next units of time to make predictions for
    T: integer
        age of the user
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the
        customer. Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if max_frequency is None:
        max_frequency = int(model.data['frequency'].max())

    if max_recency is None:
        max_recency = int(model.data['T'].max())

    mean_monetary_value = rfm_data.groupby(['frequency_cal','recency_cal']
                                 )['monetary_value_cal'
                                 ].mean(
                                 ).reset_index()
    mean_margin = rfm_data.groupby(['frequency_cal','recency_cal']
                         )['margin_cal'
                         ].mean(
                         ).reset_index()
    Z = np.zeros((max_recency + 1, max_frequency + 1))
    if T is None:
        T = max_recency
    for i, recency in enumerate(np.arange(max_recency + 1)):
        for j, frequency in enumerate(np.arange(max_frequency + 1)):
            exp_purchases = model.conditional_expected_number_of_purchases_up_to_time(
                horizon, frequency, recency, T)
            money = mean_monetary_value[
                        (mean_monetary_value['frequency_cal']==frequency)&
                        (mean_monetary_value['recency_cal']==recency)
                    ]['monetary_value_cal']
            if not money.empty:
                money = money.values[0]
            else:
                money = 0.
            margin = mean_margin[
                        (mean_margin['frequency_cal']==frequency)&
                        (mean_margin['recency_cal']==recency)
                    ]['margin_cal']
            if not margin.empty:
                margin = margin.values[0]
            else:
                margin = 0.
            Z[i, j] = exp_purchases * money
            if revenue_type == 'net':
                Z[i, j] *= 0.01 * margin
            if log:
                if Z[i, j] > 0.01:
                    Z[i, j] = np.log10(Z[i, j])
                else:
                    Z[i, j] = None
            elif Z[i, j] <= 0.01:
                Z[i, j] = None
    interpolation = kwargs.pop('interpolation', 'none')
    if ax is None:
        ax = plt.subplot(111)
    PCM = ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        revenue_type = revenue_type[0].upper()+revenue_type[1:].lower()
        title = 'Expected {} Revenue for {} Unit{} of Time,'. \
            format(revenue_type, horizon, "s"[horizon == 1:]) + '\nby Frequency and Recency of a Customer'
    plt.title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    cb = plt.colorbar(PCM, ax=ax)
    cb.set_ticklabels([str(round(np.power(10,x),1)) for x in cb.get_ticks()])
    cb.set_label('Customer RLTV @ 3 months')
    plt.show()
    return ax, Z


def plot_integral_rfm_matrix(model,
                             rfm_data,
                             horizon=1,
                             revenue_type='gross',
                             max_frequency=None,
                             max_recency=None,
                             title=None,
                             xlabel="Customer's Historical Frequency",
                             ylabel="Customer's Recency",
                             ax=None,
                             log=True,
                             **kwargs):
    """
    Plot recency frequecy matrix as heatmap, for all ages according to
    the distribution in our user base and color indicating total RLTV
    at the provided horizon.

    Plot a figure of expected transactions in T next units of time
    by a customer's frequency and recency.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    horizon: float, optional
        Next units of time to make predictions for
    T: integer
        age of the user
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the
        customer. Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if max_frequency is None:
        max_frequency = int(model.data['frequency'].max())

    if max_recency is None:
        max_recency = int(model.data['T'].max())

    mean_monetary_value = rfm_data.groupby(['frequency_cal','recency_cal']
                                 )['monetary_value_cal'
                                 ].mean(
                                 ).reset_index()
    mean_margin = rfm_data.groupby(['frequency_cal','recency_cal']
                         )['margin_cal'
                         ].mean(
                         ).reset_index()
    age_count = rfm_data.reset_index(
                       ).groupby(['frequency_cal','recency_cal','T_cal']
                       )['id'
                       ].count(
                       ).reset_index()
    N_total = age_count['id'].sum()
    Z = np.zeros((max_recency + 1, max_frequency + 1))
    for i, recency in enumerate(np.arange(max_recency + 1)):
        for j, frequency in enumerate(np.arange(max_frequency + 1)):
            exp_purchases = 0.
            partial_age_count = age_count[
                (age_count['frequency_cal']==frequency)&
                (age_count['recency_cal']==recency)
            ]
            for T,N in partial_age_count[['T_cal','id']].values:
                exp_purchases += N * model.conditional_expected_number_of_purchases_up_to_time(
                    horizon, frequency, recency, T)
            # exp_purchases /= (N_total+0.)
            money = mean_monetary_value[
                        (mean_monetary_value['frequency_cal']==frequency)&
                        (mean_monetary_value['recency_cal']==recency)
                    ]['monetary_value_cal']
            if not money.empty:
                money = money.values[0]
            else:
                money = 0.
            margin = mean_margin[
                        (mean_margin['frequency_cal']==frequency)&
                        (mean_margin['recency_cal']==recency)
                    ]['margin_cal']
            if not margin.empty:
                margin = margin.values[0]
            else:
                margin = 0.
            Z[i, j] = exp_purchases * money
            if revenue_type == 'net':
                Z[i, j] *= 0.01 * margin
            if log:
                if Z[i, j] > 0.01:
                    Z[i, j] = np.log10(Z[i, j])
                else:
                    Z[i, j] = None
            elif Z[i, j] <= 0.01:
                Z[i, j] = None
    interpolation = kwargs.pop('interpolation', 'none')
    if ax is None:
        ax = plt.subplot(111)
    PCM = ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        revenue_type = revenue_type[0].upper()+revenue_type[1:].lower()
        title = 'Expected {} Revenue for {} Unit{} of Time,'. \
            format(revenue_type, horizon, "s"[horizon == 1:]) + '\nby Frequency and Recency of a Customer'
    plt.title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    cb = plt.colorbar(PCM, ax=ax)
    print(cb.get_ticks())
    cb.set_ticklabels([str(round(np.power(10,x),1)) for x in cb.get_ticks()])
    print(cb.get_ticks())
    cb.set_label('Total RLTV @ 3 months')

    return ax, Z

def plot_probability_alive_matrix(model,
                                  max_frequency=None,
                                  max_recency=None,
                                  title='Probability Customer is Alive,\nby Frequency and Recency of a Customer',
                                  xlabel="Customer's Historical Frequency",
                                  ylabel="Customer's Recency",
                                  ax=None,
                                  **kwargs):
    """
    Plot probability alive matrix as heatmap.

    Plot a figure of the probability a customer is alive based on their
    frequency and recency.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    T: fload, optional
        Next units of time to make predictions for
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the
        customer. Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    z = model.conditional_probability_alive_matrix(max_frequency, max_recency)

    interpolation = kwargs.pop('interpolation', 'none')

    if ax is None:
        ax = plt.subplot(111)
    PCM = ax.imshow(z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    cb = plt.colorbar(PCM, ax=ax)
    cb.set_label('Survival probability')
    return ax


def plot_expected_repeat_purchases(model,
                                   title='Expected Number of Repeat Purchases per Customer',
                                   xlabel='Time Since First Purchase',
                                   ax=None,
                                   label=None,
                                   **kwargs):
    """
    Plot expected repeat purchases on calibration period .

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    max_frequency: int, optional
        The maximum frequency to plot.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    label: str, optional
        Label for plot.
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    if plt.matplotlib.__version__ >= "1.5":
        color_cycle = ax._get_lines.prop_cycler
        color = coalesce(kwargs.pop('c', None),
                         kwargs.pop('color', None),
                         next(color_cycle)['color'])
    else:
        color_cycle = ax._get_lines.color_cycle
        color = coalesce(kwargs.pop('c', None),
                         kwargs.pop('color', None), next(color_cycle))

    max_T = model.data['T'].max()

    times = np.linspace(0, max_T, 100)
    ax.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, label=label, **kwargs)

    times = np.linspace(max_T, 1.5 * max_T, 100)
    ax.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, ls='--', **kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    if label is not None:
        plt.legend(loc='lower right')
    return ax


def plot_history_alive(model, t, transactions, datetime_col, freq='D',
                       start_date=None, end_date=None, ax=None,
                       title='Evolution of survival probability', **kwargs):
    """
    Draw a graph showing the probablility of being alive for a customer in time.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
    t: int
        the number of time units since the birth we want to draw the p_alive
    transactions: pandas DataFrame
        DataFrame containing the transactions history of the customer_id
    datetime_col: str
        The column in the transactions that denotes the datetime the purchase
        was made
    freq: str, optional
        Default 'D' for days. Other examples= 'W' for weekly
    start_date: datetime, optional
        Limit xaxis to start date
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    title: str, optional
        Title of the plot.
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    first_transaction_date = transactions[datetime_col].min()
    if start_date is None:
        start_date = dt.datetime.strptime(first_transaction_date,"%Y-%m-%d") - \
                                          dt.timedelta(days=7)
    if ax is None:
        ax = plt.subplot(111)
    # Get purchasing history of user
    customer_history = transactions[[datetime_col]].copy()
    customer_history.index = pd.DatetimeIndex(customer_history[datetime_col])

    # Add transactions column
    customer_history['transactions'] = 1
    customer_history = customer_history.resample(freq).sum()

    current_date = dt.datetime.strftime(dt.datetime.now(),"%Y-%m-%d")
    refresh_date = dt.datetime.strftime(
        dt.datetime.strptime(first_transaction_date,"%Y-%m-%d") + dt.timedelta(days=t),
        "%Y-%m-%d")
    periods_to_plot = (dt.datetime.strptime(end_date,"%Y-%m-%d") \
                       - start_date
                      ).days
    if freq == 'W':
        periods_to_plot = int(periods_to_plot/7.)
    # plot alive_path
    path = pd.concat(
        [pd.Series([None]*7),
         calculate_alive_path(model, transactions, datetime_col, periods_to_plot, freq)*100.])
    path_dates = pd.date_range(start=start_date, periods=len(path), freq=freq)
    past_dates_path = path_dates[path_dates<=refresh_date]
    past_path = path[path_dates<=refresh_date]
    future_dates_path = path_dates[path_dates>=refresh_date]
    future_path = path[path_dates>=refresh_date]
    plt.plot(past_dates_path, past_path, color=glovo_gray, ls='-', label='P_alive')
    plt.plot(future_dates_path, future_path, color=glovo_gray, ls='--')

    # plot buying dates
    payment_dates = customer_history[customer_history['transactions'] >= 1].index
    plt.vlines(payment_dates.values, ymin=0, ymax=100,
               colors=glovo_green, linestyles='dashed', label='Orders')
    plt.vlines(refresh_date, ymin=0, ymax=100,
               colors=glovo_yellow, linestyles='solid', label='Last refresh')
    plt.vlines(current_date, ymin=0, ymax=100,
               colors='red', linestyles='solid', label='Today')
    plt.ylim(0, 105)
    plt.yticks(np.arange(0, 110, 10))
    plt.xticks(rotation=-20.)
    plt.xlim(start_date, path_dates[-1])
    plt.legend(loc=3)
    plt.ylabel('P_alive (%)')
    plt.title(title)

    return ax


def plot_cumulative_transactions(model, transactions, datetime_col, customer_id_col, t, t_cal,
                                 datetime_format=None, freq='D', set_index_date=False,
                                 title='Tracking Cumulative Transactions',
                                 xlabel='day', ylabel='Cumulative Transactions',
                                 ax=None, **kwargs):
    """
    Plot a figure of the predicted and actual cumulative transactions of users.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model
    transactions: pandas DataFrame
        DataFrame containing the transactions history of the customer_id
    datetime_col: str
        The column in transactions that denotes the datetime the purchase was made.
    customer_id_col: str
        The column in transactions that denotes the customer_id
    t: float
        The number of time units since the begining of
        data for which we want to calculate cumulative transactions
    t_cal: float
        A marker used to indicate where the vertical line for plotting should be.
    datetime_format: str, optional
        A string that represents the timestamp format. Useful if Pandas
        can't understand the provided format.
    freq: str, optional
        Default 'D' for days, 'W' for weeks, 'M' for months... etc.
        Full list here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    set_index_date: bool, optional
        When True set date as Pandas DataFrame index, default False - number of time units
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    kwargs
        Passed into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    df_cum_transactions = expected_cumulative_transactions(model, transactions, datetime_col,
                                                           customer_id_col, t,
                                                           datetime_format=datetime_format, freq=freq,
                                                           set_index_date=set_index_date)

    ax = df_cum_transactions.plot(ax=ax, title=title, **kwargs)

    if set_index_date:
        x_vline = df_cum_transactions.index[int(t_cal)]
        xlabel = 'date'
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color='r', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_incremental_transactions(model, transactions, datetime_col, customer_id_col, t, t_cal,
                                  datetime_format=None, freq='D', set_index_date=False,
                                  title='Tracking Daily Transactions',
                                  xlabel='day', ylabel='Transactions',
                                  ax=None, **kwargs):
    """
    Plot a figure of the predicted and actual cumulative transactions of users.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model
    transactions: pandas DataFrame
        DataFrame containing the transactions history of the customer_id
    datetime_col: str
        The column in transactions that denotes the datetime the purchase was made.
    customer_id_col: str
        The column in transactions that denotes the customer_id
    t: float
        The number of time units since the begining of
        data for which we want to calculate cumulative transactions
    t_cal: float
        A marker used to indicate where the vertical line for plotting should be.
    datetime_format: str, optional
        A string that represents the timestamp format. Useful if Pandas
        can't understand the provided format.
    freq: str, optional
        Default 'D' for days, 'W' for weeks, 'M' for months... etc.
        Full list here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    set_index_date: bool, optional
        When True set date as Pandas DataFrame index, default False - number of time units
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    kwargs
        Passed into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.subplot(111)

    df_cum_transactions = expected_cumulative_transactions(model, transactions, datetime_col,
                                                           customer_id_col, t,
                                                           datetime_format=datetime_format, freq=freq,
                                                           set_index_date=set_index_date)

    # get incremental from cumulative transactions
    df_cum_transactions = df_cum_transactions.apply(lambda x: x - x.shift(1))
    ax = df_cum_transactions.plot(ax=ax, title=title, **kwargs)

    if set_index_date:
        x_vline = df_cum_transactions.index[int(t_cal)]
        xlabel = 'date'
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color='r', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_transaction_rate_heterogeneity(model,
                                        title='Heterogeneity in Transaction Rate',
                                        xlabel='Transaction Rate',
                                        ylabel='Density',
                                        suptitle_fontsize=14,
                                        ax=None,
                                        **kwargs):
    """
    Plot the estimated gamma distribution of lambda (customers' propensities to purchase).

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model, for now only for BG/NBD
    suptitle: str, optional
        Figure suptitle
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    #TODO Include a text with the PDF with the fitted values
    from matplotlib import pyplot as plt

    r, alpha = model._unload_params('r', 'alpha')
    rate_mean = r / alpha
    rate_var = r / alpha ** 2

    rv = stats.gamma(r, scale=1 / alpha)
    lim = rv.ppf(0.99)
    x = np.linspace(0, lim, 100)

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_title(title+'\nmean: {:.3f}, var: {:.3f}'.format(rate_mean, rate_var))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.plot(x, rv.pdf(x), **kwargs)
    return ax


def plot_dropout_rate_heterogeneity(model,
                                    title='Heterogeneity in Dropout Probability',
                                    xlabel='Dropout Probability p',
                                    ylabel='Density',
                                    suptitle_fontsize=14,
                                    ax=None,
                                    **kwargs):
    """
    Plot the estimated gamma distribution of p.

    p - (customers' probability of dropping out immediately after a transaction).

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model, for now only for BG/NBD
    suptitle: str, optional
        Figure suptitle
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    kwargs
        Passed into the matplotlib.pyplot.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    """
    #TODO Include a text with the PDF with the fitted values
    from matplotlib import pyplot as plt

    a, b = model._unload_params('a', 'b')
    beta_mean = a / (a + b)
    beta_var = a * b / ((a + b) ** 2) / (a + b + 1)

    rv = stats.beta(a, b)
    lim = rv.ppf(0.99)
    x = np.linspace(0, lim, 100)

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.set_title(title+'\nmean: {:.3f}, var: {:.3f}'.format(beta_mean, beta_var))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.plot(x, rv.pdf(x), **kwargs)
    return ax


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
