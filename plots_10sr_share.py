import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import io
from matplotlib.colors import ListedColormap
from sklearn import linear_model
import scipy as sp

### Functions ####


def make_figure2a(datafiles,plotname):
    # Load Data
    cvdata=io.loadmat(datafiles[0])
    cvdata=cvdata['behav_struct_norm']
    cvdata_extra=io.loadmat(datafiles[1])
    cvdata_extra=cvdata_extra['behav_struct_norm']

    # Set labels
    cvtypes=['k2','k5','k10','loo']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']
    
    # Aggregate predicted and actual behavioral scores for first 100 iterations
    pred_data_dict={cvt:cvdata[0][cvt][0]['predbehavpos'][0][0] for cvt in cvtypes}
    real_data_dict={cvt:cvdata[0][cvt][0]['testbehav'][0][0] for cvt in cvtypes}

    # Aggregate predicted and actual behavioral scores for second 100 iterations
    pred_data_dict_extra={cvt:cvdata_extra[0][cvt][0]['predbehavpos'][0][0] for cvt in cvtypes}
    real_data_dict_extra={cvt:cvdata_extra[0][cvt][0]['testbehav'][0][0] for cvt in cvtypes}

    # Calculate MSE and aggregate into lists
    cllct=[]
    for cvt in cvtypes:
        mse1=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        mse2=np.mean(np.array(pred_data_dict_extra[cvt]-real_data_dict_extra[cvt])**2,axis=1)
        mse=np.concatenate([mse1,mse2])
        cllct.append(mse)

    # Convert to array
    mse_tot=np.array(cllct)
    
    # Embed in dataframe
    ipdata=pd.DataFrame(mse_tot.T,columns=cvtypes_plots)

    # Create and save plot
    cv_effect_plot(ipdata,plotname)



def make_figure2b(datafile,plotname):
    # Load data
    ctrain_data=io.loadmat(datafile)
    cvdata=ctrain_data['behav_struct_norm_st']
 
    # List labels
    cvtypes=['k2','k5','k10','loo']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']


    # Aggregate predicted and actual behavioral scores
    pred_data_dict={cvt:cvdata[0][cvt][0]['predbehavpos'][0][0] for cvt in cvtypes}
    real_data_dict={cvt:cvdata[0][cvt][0]['testbehav'][0][0] for cvt in cvtypes}

    # Calculate and gather MSE
    cllct=[]
    for cvt in cvtypes:
    
        mse=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        cllct.append(mse)

    # Convert to array
    mse_tot=np.array(cllct)
    
    # Embed in dataframe
    ipdata=pd.DataFrame(mse_tot.T,columns=cvtypes_plots)
    ipdata=ipdata[['Split-half', '5-fold', '10-fold', 'LOO']]

    # Plot
    const_trainsize_plot(ipdata,plotname)

def make_figure3(datafile,plotname):

    # Load data
    biasstuff=io.loadmat(datafile,mat_dtype=True)
    bias_data=biasstuff['rsn_tot'][0]


    # List CV labels in data structure, and what we would like the labels to be in the plots
    cvtypes=['k2','k5','k10','loo']
    cvtypes_plots=['Split-half', '5-fold', '10-fold', 'LOO']


    # Aggregate both measures, for each CV method, into one array
    Rpos=np.concatenate([bias_data[0][cvt][:,0] for cvt in cvtypes])
    Rmse=np.concatenate([bias_data[0][cvt][:,4] for cvt in cvtypes])
    labels=np.concatenate([np.repeat(cv,200) for cv in cvtypes_plots])
    dfarr=np.concatenate([np.vstack(labels).T,np.vstack(Rpos).T,np.vstack(Rmse).T]).T

    # Column headers and plot labels
    n1="Rsq(observed,yhat)"
    n2="Rsq(observed,pred)"

    ipdata=pd.DataFrame(dfarr,columns=['Labels',n1,n2])

    # Ensure data cast as float
    ipdata[n1]=ipdata[n1].astype('float')
    ipdata[n2]=ipdata[n2].astype('float')

    # Rsq(observed,yhat) was orginally calculated as square root of MSE
    # Adjust here for current comparison with Rsq
    ipdata[n1]=ipdata[n1]**2

    # Create 45 deg line for plotting
    ipdata['X']=np.concatenate([np.linspace(0,0.3,200) for n in range(0,len(cvtypes_plots))])
    ipdata['Y']=np.concatenate([np.linspace(0,0.3,200) for n in range(0,len(cvtypes_plots))])


    # Generate plot
    g=sns.FacetGrid(ipdata,col="Labels",col_wrap=2,hue="Labels",palette='vlag',despine=True)
    axes = g.axes.flatten()
    g=g.map(plt.plot,"X","Y")
    g=g.map(plt.scatter,n1,n2).set_axis_labels(r'$R^2(explanatory)$', r'$R^2(prediction)$')

    # Subplot titles
    axes[0].set_title("Split-half")
    axes[1].set_title("5-fold")
    axes[2].set_title("10-fold")
    axes[3].set_title("LOO")


    # Save plot
    plt.savefig(plotname, dpi=300, facecolor='w', edgecolor='w',
         orientation='portrait', papertype=None, format=None,
         transparent=False, bbox_inches=None, pad_inches=0.1,
         frameon=None)   


def make_figure4(datafile,plotname):
    # Load data
    nsubsdata=io.loadmat(datafile)
    nsubsdata=nsubsdata['behav_struct_nsubs']
    
    # Setup labels of dataframe and for plots
    train_lbls=['train'+str(num) for num in np.arange(25,401,25)]    
    col_lbls=[str(num) for num in np.arange(25,401,25)]

    # Pull out predicted behavioural trait and actual behavioural trait
    pred_data_dict={cvt:nsubsdata[0][cvt][0]['predbehavpos'][0][0] for cvt in train_lbls}
    real_data_dict={cvt:nsubsdata[0]['testbehav'][0] for cvt in train_lbls}
    

    # Calculate MSE and R and aggregate into lists
    cllct=[]
    for cvt in train_lbls:
        mse=np.mean(np.array(pred_data_dict[cvt]-real_data_dict[cvt])**2,axis=1)
        cllct.append(mse)

    # Convert list to array
    mse_tot=np.array(cllct)

    # Create dataframe
    ipdata=pd.DataFrame(mse_tot.T,columns=col_lbls)
    # Plot and save plot
    train_subs_plot(ipdata, plotname)




def cv_effect_plot(ipdata,opname):
    # Setup plot environment
    plt.clf()
    sns.set(style='ticks')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=ipdata,palette='vlag')
    sns.despine(offset=10, trim=True)

    # Title and labels
    plt.title('Variable train size',fontsize='large')
    plt.ylabel('Normalized MSE',weight='bold')
    plt.xlabel('Cross-validation method',weight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)



def train_subs_plot(ipdata,opname):
    # Create plot environment
    plt.clf()
    sns.set(style='ticks')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=ipdata,palette='vlag')
    sns.despine(offset=20, trim=True)

    # Title and labels
    plt.title('Effect of number of training individuals on prediction',fontsize='large')
    plt.ylabel('Normalized MSE',weight='bold')
    plt.xlabel('Number of training individuals',weight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


def const_trainsize_plot(ipdata,opname):
    # Create plot environment
    plt.clf()
    sns.set(style='ticks')
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=ipdata,palette='vlag')
    sns.despine(offset=10, trim=True)

    # Title and labels
    plt.title('Constant train size (n=180)',fontsize='large')
    plt.ylabel('Normalized MSE',weight='bold')
    plt.xlabel('Cross-validation method',weight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(opname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

if __name__ == '__main__':
    data_dir='data'
    fig_dir='figs'


    data2a=[data_dir+'/data_figure2a_part1.mat',data_dir+'/data_figure2a_part2.mat']

    make_figure2a(data2a,fig_dir+'/figure2a.tiff')
    make_figure2b(data_dir+'/data_figure2b.mat',fig_dir+'/figure2b.tiff')
    make_figure3(data_dir+'/data_figure3.mat',fig_dir+'/figure3.tiff')
    make_figure4(data_dir+'/data_figure4.mat', fig_dir+'/figure4.tiff')

