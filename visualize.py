import matplotlib.pyplot as plt

def plot_predictions(dates, actual, predicted):
    plt.figure()
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.legend()
    plt.title('Sales Forecast')
    plt.show()