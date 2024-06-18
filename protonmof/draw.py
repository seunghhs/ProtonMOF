import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

myfonts = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = myfonts
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    cmap=plt.cm.get_cmap('summer')
    ax.scatter( x, y, c=z, cmap=cmap, alpha=0.6,**kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax, cmap=cmap)
    #cbar.ax.set_ylabel('Density', fontsize=16)

    return ax
    
def draw_result_graph( x_train, x_test,y_train, y_test,
                      best_model= None,
                      label='proton conductivity(S/cm)', save_true=False, save_name=''):
    if best_model is not None:    
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)
    else:
        y_train_pred = x_train
        y_test_pred = x_test
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    mse_train = mean_squared_error(y_train, y_train_pred) 
    mse_test = mean_squared_error(y_test,y_test_pred)

    print('R2 Score - Train : %.3f, Test : %3f' %(r2_train, r2_test))
    print('MSE - Train : %.3f, Test : %3f' %(mse_train, mse_test))
    print('RMSE - Train : %.3f, Test : %3f' %(np.sqrt(mse_train), np.sqrt(mse_test)))
    print(f'MAE - Train: {mean_absolute_error(y_train, y_train_pred)}, Test: {mean_absolute_error(y_test, y_test_pred)}')

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train,y_train_pred,c='lightgray',marker='o', label='Training')
    #density_scatter(y_test, y_test_pred)
    plt.scatter(y_test,y_test_pred,c='green',marker='s', label='Test')
    
    y_train_pred=np.expand_dims(y_train_pred, axis=1)
    y_test_pred=np.expand_dims(y_test_pred, axis=1)
    
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1,1)
        y_train_pred = y_train_pred.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        y_test_pred = y_test_pred.reshape(-1,1)
    
    mini=np.min(np.concatenate([y_train,y_train_pred, y_test,y_test_pred]))
    maxi=np.max(np.concatenate([y_train,y_train_pred, y_test,y_test_pred]))
    a = np.linspace(mini,maxi)
    plt.plot(a,a,c='black',linestyle='dashed',linewidth=1)

    plt.ylabel(label+' (Predicted)', fontsize=20)
    plt.xlabel(label+' (Origin)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    if save_true:
        plt.tight_layout()   
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()


def draw_result_graph_density(x_train, x_test,y_train, y_test,best_model = None,  save_name='', title_name=''):
    from scipy.stats import gaussian_kde
    
    myfonts = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = myfonts
    plt.rcParams['figure.figsize'] = (4, 4)
    plt.rcParams['font.size'] = 12
    plt.rcParams['pdf.fonttype'] = 42
    
    palette = sns.set_palette('Spectral')
    fig, axe = plt.subplots(1,1, )
    if best_model is not None:    
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)
    else:
        y_train_pred = x_train
        y_test_pred = x_test
    
    xy = np.vstack([y_test, y_test_pred])
    z = gaussian_kde(xy)(xy)

    # y_train_pred_tmp=np.expand_dims(y_train_pred, axis=1)
    # y_test_pred_tmp=np.expand_dims(y_test_pred, axis=1)
    
    if y_train.ndim == 1:
        y_train_tmp = y_train.reshape(-1,1)
        y_train_pred_tmp = y_train_pred.reshape(-1,1)
        y_test_tmp = y_test.reshape(-1,1)
        y_test_pred_tmp = y_test_pred.reshape(-1,1)
        mini=np.min(np.concatenate([y_train_tmp,y_train_pred_tmp, y_test_tmp,y_test_pred_tmp]))
        maxi=np.max(np.concatenate([y_train_tmp,y_train_pred_tmp, y_test_tmp,y_test_pred_tmp]))
    else:
        mini=np.min(np.concatenate([y_train,y_train_pred, y_test,y_test_pred]))
        maxi=np.max(np.concatenate([y_train,y_train_pred, y_test,y_test_pred]))
    a = np.linspace(mini,maxi)
    axe.plot(a,a,c='gray',linestyle='dashed',linewidth=1)
    

    
    sns.scatterplot(x=y_train, y=y_train_pred, ax=axe, color='gray')

    sns.scatterplot(x=y_test, y=y_test_pred, ax=axe, c=z, cmap='summer')
    

    axe.set_xlabel('log $\mathrm{\sigma}$ (S/cm) (Actual)', fontsize=12)
    axe.set_ylabel('log $\mathrm{\sigma}$ (S/cm) (Predicted)')
    if title_name:
        plt.title(title_name, fontsize=13)
    else:
        plt.title('Proton Conductivity', fontsize=13)
        
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, transparent=True, dpi=600)