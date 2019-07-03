"""Init for fitters."""
import numpy as np
from .base_fitter import BaseFitter
import logging

horizon_dict = {
    28:'1mo',   29:'1mo',   30:'1mo',   31:'1mo',   32:'1mo',   33:'1mo',
    58:'2mo',   59:'2mo',   60:'2mo',   61:'2mo',   62:'2mo',   63:'2mo',
    88:'3mo',   89:'3mo',   90:'3mo',   91:'3mo',   92:'3mo',   93:'3mo',
    179:'6mo',  180:'6mo',  181:'6mo',  182:'6mo',  183:'6mo',  184:'6mo',  185:'6mo',
    270:'9mo',  271:'9mo',  272:'9mo',
    273:'9mo',  274:'9mo',  275:'9mo',  276:'9mo',  277:'9mo',  278:'9mo',  279:'9mo',
    362:'12mo', 363:'12mo', 364:'12mo', 365:'12mo', 366:'12mo', 367:'12mo', 368:'12mo',
    545:'18mo', 546:'18mo', 547:'18mo', 548:'18mo', 549:'18mo', 550:'18mo', 551:'18mo',
    727:'24mo', 728:'24mo', 729:'24mo', 730:'24mo', 731:'24mo', 732:'24mo', 733:'24mo',
}

# Current LTV model was tested with a 91 days horizon, longer term predictions
# are not tested so performance is unknown
max_horizon = 93

class CLTVModel(BaseFitter):
    """
    A sklearn wrapper for CLTV models from lifetimes package, so other sklearn
    tools can be directly applied.
    """
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.X_train = None
        self.y_train = None
        self.freq = None
        self.rec = None
        self.age = None

    def fit(self, X, y, **fit_params):
        self.X_train = X
        self.y_train = y
        self.freq_train = np.array(list(zip(*self.X_train))[0])
        self.rec_train = np.array(list(zip(*self.X_train))[1])
        self.age_train = np.array(list(zip(*self.X_train))[2])
        self.model.fit(
            frequency=self.freq_train, recency=self.rec_train, T=self.age_train)

    def predict(self, X, horizon):
        freq, rec, age = list(zip(*X))
        return self.model.predict(
            t=horizon, frequency=np.array(freq),
            recency=np.array(rec), T=np.array(age))

    def score(self, X, y, horizon):
        y_est = self.predict(X, horizon)
        y_err = y - y_est
        y_err_pct = y_err / y_est * 100.
        y_err_abs = np.abs(y_err)
        me = np.mean(y_err)
        mae = np.mean(y_err_abs)
        rmse = np.sqrt(np.mean(np.square(y_err)))
        pctdeltarep = np.sum(y_err) / np.sum(y_est) * 100
        if self.verbose:
            print((
                "ME: {:.2f}; MAE: {:.2f}; RMSE: {:.2f} ; Order Error: {.2f}$"
                .format(me,mae,rmse,pctdeltarep)
            ))
        return mae

def add_predictions(rfm_data, model, horizon, testing=True, logger=None):
    """
        Given a RFM dataframe, a LTV model and an number of periods,
        attach the following metrics: frequency, orders, gross revenue,
        as well as the current survival probability oft the customer.
        Necessary columns in the input dataframe are: `frequency_cal`,
        `recency_cal`, `T_cal`, `monetary_value_cal`. If column
        `margin_cal` if also present, the estimated net revenue is also
        computed.
        If testing is set to true, columns `frequency_holdout`,
        `monetary_value_holdout` should also be available, so errors
        in estimated frequency and gross revenue can be computed. If
        `margin_holdout` is also available, error in net revenue is also
        computed.
    """
    if logger is None:
        logger = logging.getLogger('execution')
    logger.info("Computing LTV prediction metrics for an horizon of {} days".format(horizon))
    rfm = rfm_data.copy()
    if horizon not in horizon_dict:
        raise ValueError("Horizon value not expected: {}".format(horizon))
    horizon_str = horizon_dict[horizon]
    if horizon > int(max_horizon*1.05):
        logger.warning((
            "You are estimating LTV in an untested horizon! ({})"
            .format(horizon_dict[horizon])
        ))
        # Add flag to warn about using LTV values for longer periods than tested
        horizon_str += '_UNTESTED'
    money = False
    # Feature vectors, all users
    frequency_history = rfm['frequency_cal']
    recency_history = rfm['recency_cal']
    age_history = rfm['T_cal']
    orders_per_period_history = rfm['orders_per_period_cal']
    if testing:
        # Target vectors, all users
        frequency_horizon = rfm['frequency_holdout']
    if 'monetary_value_cal' in rfm.columns:
        money = True
        money_history = rfm['monetary_value_cal']
        if testing:
            money_horizon = rfm['monetary_value_holdout']
    freq_est = model.predict(t=horizon, frequency=frequency_history,
                             recency=recency_history, T=age_history)
    rfm['frequency_est_'+horizon_str] = np.round(freq_est,2)
    rfm['orders_est_'+horizon_str] = np.round(freq_est * orders_per_period_history,2)
    logger.info("Future orders estimated")
    if 'survival_probability' not in rfm.columns:
        survival_prob = model.conditional_probability_alive(
            frequency=frequency_history, recency=recency_history, T=age_history)
        rfm['survival_probability'] = np.round(survival_prob * 100.,1)
    logger.info("Survival probability estimated")
    if testing:
        freq_err = frequency_horizon - freq_est
        freq_err_pct = freq_err / frequency_horizon * 100.
        freq_err_abs = np.abs(freq_err)
        rfm['frequency_err_'+horizon_str] = np.round(freq_err,4)
        logger.info("Frequency error computed")
    if money:
        revenue_est = freq_est * money_history
        rfm['gross_revenue_est_'+horizon_str] = np.round(revenue_est,2)
        logger.info("Future gross revenue estimated")
        if testing:
            revenue_horizon = frequency_horizon * money_horizon
            revenue_err = revenue_horizon - revenue_est
            revenue_err_abs = np.abs(revenue_horizon - revenue_est)
            rfm['gross_revenue_err_'+horizon_str] = np.round(revenue_err,4)
            logger.info("Gross revenue error computed")
        if 'margin_cal' in rfm.columns:
            margin = rfm['margin_cal']
            rfm['net_revenue_est_'+horizon_str] = np.round(
                revenue_est * margin / 100.,2)
            logger.info("Future new revenue estimated")
        if 'margin_holdout' in rfm.columns:
            margin_horizon = rfm['margin_holdout']
            rfm['net_revenue_err_'+horizon_str] = \
                np.round((revenue_horizon*margin_horizon - revenue_est*margin)/100.,4)
            logger.info("Net revenue error computed")
    # Add additional metrics for 3mo LTV
    rfm['ltv_'+horizon_str] = (rfm['frequency_cal'] + 1.) * \
        rfm['monetary_value_cal'] * rfm['margin_cal']/100. + \
        rfm['net_revenue_est_'+horizon_str]
    rfm['rltv_pct_'+horizon_str] = rfm['net_revenue_est_'+horizon_str] / \
        rfm['ltv_'+horizon_str] * 100.
    return rfm
