import numpy as np

epsilon = np.finfo(float).eps


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def WAPE(y_pred, y):
    """Weighted Average Percentage Error metric in the interval [0; 100]"""
    nominator = np.sum(np.abs(np.subtract(y, y_pred)))
    denominator = np.add(np.sum(np.abs(y)), epsilon)
    wape = np.divide(nominator, denominator)*100.0
    return wape

def NSE(y_pred, y):
    return (1-(np.sum((y_pred-y)**2)/np.sum((y-np.mean(y))**2)))

def PFE (y_pred, y):
    return abs(y.max() - y_pred.max())/y.max()

def TPE (y_pred, y):
    return abs(np.argmax(y) - np.argmax(y_pred)) * 60

def r_factor(y_pred, y):
    """
    Calculate the R-factor (reliability factor) for time series predictions.
    
    Parameters:
    y_pred (numpy.array): Predicted values
    y (numpy.array): Observed (true) values
    
    Returns:
    float: R-factor value
    """
    if len(y_pred) != len(y):
        raise ValueError("Predicted and observed arrays must have the same length")
    
    n = len(y)
    numerator = np.sum((y_pred - y)**2)
    denominator = np.sum((y - np.mean(y))**2)
    
    r_factor = np.sqrt(numerator / denominator)
    
    return r_factor

def calculate_r_factor(y_pred, y):
    # Calculate the standard deviation of observed values
    std_dev_y = np.std(y)
    
    # Calculate the lower and upper bounds of the 95PPU band
    lower_bound = np.percentile(y_pred, 2.5, axis=0)
    upper_bound = np.percentile(y_pred, 97.5, axis=0)
    
    # Calculate the average width of the 95PPU band
    average_width_95ppu = np.mean(upper_bound - lower_bound)
    
    # Calculate the R-Factor
    r_factor = (average_width_95ppu / std_dev_y) * 100
    
    return r_factor


def p_factor_95(y_pred, y, alpha=0.05):
    """
    Calculate the 95% p-factor for time series predictions.
    
    Parameters:
    y_pred (numpy.array): Predicted values
    y_obs (numpy.array): Observed (true) values
    alpha (float): Significance level (default is 0.05 for 95% confidence)
    
    Returns:
    float: 95% p-factor value (between 0 and 1)
    """
    if len(y_pred) != len(y):
        raise ValueError("Predicted and observed arrays must have the same length")
    
    # Calculate prediction error
    error = y_pred - y
    
    # Calculate the 95% confidence interval of the error
    lower_bound = np.percentile(error, alpha/2 * 100)
    upper_bound = np.percentile(error, (1 - alpha/2) * 100)
    
    # Count how many observed values fall within the 95% prediction interval
    within_bounds = np.sum((error >= lower_bound) & (error <= upper_bound))
    
    # Calculate 95% p-factor
    p_factor = within_bounds / len(y)
    
    return p_factor

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    wape = WAPE(pred, true)
    nse = NSE(pred, true)
    pfe = PFE(pred, true)
    tpe = TPE(pred, true)
    rfactor = r_factor(pred, true)
    pfactor95 = p_factor_95(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, wape, nse, pfe, tpe, rfactor, pfactor95
