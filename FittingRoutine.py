#%%
import numpy as np
from inspect import signature
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def derivative(params,func,x):
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + x)
    return (func(x + eps,*params) - func(x - eps,*params)) / (2.0 * eps)

def Chi2_no_error(func,x,y,*args):
    Chi2 = 0
    return Chi2

def NormRes_error_x(params,func,x,y,error_x):
    if error_x.any() == 0:
        return 0
    else:
        return np.nan_to_num((y - func(x,*params)) / (error_x *  derivative(params,func,x)), nan=0.0, posinf=0.0, neginf=0.0)

def Chi2_error_x(params,func,x,y,error_x):
    Chi2 = 0
    for i,j,k in zip(x,y,error_x):
        Chi2 += NormRes_error_x(params,func,i,j,k)**2
    return Chi2

def NormRes_error_y(params,func,x,y,error_y):
    if error_y.any() == 0:
        return 0
    else:
        return (y - func(x,*params)) / (error_y)
    

def Chi2_error_y(params,func,x,y,error_y):
    Chi2 = 0
    for i,j,k in zip(x,y,error_y):
        Chi2 += NormRes_error_y(params, func, i, j, k)**2
    return Chi2

def NormRes_error_both(params,func,x,y,error_x,error_y):
    if (error_y**2 + (error_x *  derivative(params,func,x))**2).any() == 0:
        return 0
    else:
        return (y - func(x,*params)) / (error_y**2 + (error_x *  derivative(params,func,x))**2)**(1/2)

def Chi2_error_both(params,func,x,y,error_x,error_y):
    Chi2 = 0
    for i,j,k,l in zip(x,y,error_x,error_y):
        Chi2 += NormRes_error_both(params, func, i, j, k, l)**2
    return Chi2




class FittingRoutine:

    def __init__(self, function, x, y, error_y = np.array([None]), error_x = np.array([None]), P0 = np.array([None]), method = None, bounds = None, *kargs):

        '''Check if function is a function'''
        if callable(function) != 1:
            raise SyntaxError('First argument must be a callable function')
        else:
            pass

        '''Check if function is compatible'''
        if len(signature(function).parameters) <= 1:
            raise SyntaxError('Function has to few arguments')
        else:
            pass

        '''Check if data has the right format'''
        if isinstance(x, (np.ndarray)) != 1:
            raise SyntaxError('x has to be a numpy array')
        else:
            pass
        if isinstance(y, (np.ndarray)) != 1:
            raise SyntaxError('y has to be a numpy array')
        else:
            pass
        if isinstance(error_x, (np.ndarray)) != 1:
            raise SyntaxError('error_x has to be a numpy array')
        else:
            self.error_x = error_x
        
        if isinstance(error_y, (list, tuple, np.ndarray)) != 1:
            raise SyntaxError('error_y has to be a numpy array')
        else:
            self.error_y = error_y

        '''Check if data is compatible'''
        if (error_x.any() == None and error_y.any() == None):
            if (len(x)-len(y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(y)))
            else:
                pass

        elif (error_x.any() != None and error_y.any() == None):
            if (len(x)-len(y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(y)))
            elif (len(x)-len(error_x)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(error_x)))
            else:
                pass

        elif (error_x.any() == None and error_y.any() != None):
            if (len(x)-len(y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(y)))
            elif (len(x)-len(error_y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(error_y)))
            else:
                pass

        elif (error_x.any() != None and error_y.any() != None):
            if (len(x)-len(y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(y)))
            elif (len(error_x)-len(error_y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(error_x)) + ' is not compatible with data of length ' + str(len(error_y)))
            elif (len(x)-len(error_y)) != 0:
                raise SyntaxError('Length of data does not match. Data of length ' + str(len(x)) + ' is not compatible with data of length ' + str(len(error_y)))
            else:
                pass

        self.func = function
        self.x = x
        self.y = y 
        
        
        '''Check compatibility of P0'''
        if P0.any() == None:
            self.params = np.ones(len(signature(function).parameters)-1)
            self.initparams = self.params
        else:
            if isinstance(P0, (np.ndarray)) != 1:
                raise SyntaxError('P0 has to be a numpy array')
            elif (len(signature(function).parameters)-1) != len(P0):
                raise SyntaxError('Length of P0 ' + str(len(P0)) + ' is not compatible with ' + str(len(signature(function).parameters)-1) + ' parameters in function')
            else:
                self.initparams = P0
                self.params = P0

        self.df = len(x)-len(self.params)


        '''Fitting with Minimize'''

        if (error_x.any() == None and error_y.any() == None):
            raise NotImplementedError("No errors on x or y?")

        elif (error_x.any() != None and error_y.any() == None):
            self.FitResult = minimize(Chi2_error_x,self.params,
                                    args=(self.func,self.x,self.y,self.error_x),
                                    method = method, bounds = bounds, *kargs)
            self.params = self.FitResult.x
            self.Chi2 = self.FitResult.fun
            self.Cov = self.FitResult.hess_inv * 2
            self.Pval = chi2.sf(self.Chi2,self.df)
            self.NormRes = NormRes_error_x(self.params,self.func,self.x,self.y,self.error_x)
            if self.FitResult.success!=1:
                print(self.FitResult.message+' P-value at '+str(self.Pval))

        elif (error_x.any() == None and error_y.any() != None):
            self.FitResult = minimize(Chi2_error_y,self.params,
                                    args=(self.func,self.x,self.y,self.error_y),
                                    method = method, bounds = bounds, *kargs)
            self.params = self.FitResult.x
            self.Chi2 = self.FitResult.fun
            self.Cov = self.FitResult.hess_inv * 2
            self.Pval = chi2.sf(self.Chi2,self.df)
            self.NormRes = NormRes_error_y(self.params,self.func,self.x,self.y,self.error_y)
            if self.FitResult.success!=1:
                print(self.FitResult.message+' P-value at '+str(self.Pval))

        elif (error_x.any() != None and error_y.any() != None):
            self.FitResult = minimize(Chi2_error_both,self.params,
                                    args=(self.func,self.x,self.y,self.error_x,self.error_y),
                                    method = method, bounds = bounds, *kargs)
            self.params = self.FitResult.x
            self.Chi2 = self.FitResult.fun
            self.Cov = self.FitResult.hess_inv * 2
            self.Pval = chi2.sf(self.Chi2,self.df)
            self.NormRes = NormRes_error_both(self.params,self.func,self.x,self.y,self.error_x,self.error_y)
            if self.FitResult.success!=1:
                print(self.FitResult.message+' P-value at '+str(self.Pval))

        else:
            print("Something is wrong")


    def Plot(self, ms=5, capsize = 3, xlabel='x', ylabel='y', figsize=(6,6), init=False, legend=True, stats=True, legend_loc = 'upper right', save=None):
        fig, axes = plt.subplots(2,1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

        if (self.error_x.any() == None and self.error_y.any() == None):
            axes[0].plot(self.x,self.y,'ko', ms=ms, capsize=capsize)
        elif (self.error_x.any() == None and self.error_y.any() != None):
            axes[0].errorbar(self.x,self.y,yerr=self.error_y,fmt='ko', ms=ms, capsize=capsize)
        elif (self.error_x.any() != None and self.error_y.any() == None):
            axes[0].errorbar(self.x,self.y,xerr=self.error_x,fmt='ko', ms=ms, capsize=capsize)
        elif (self.error_x.any() != None and self.error_y.any() != None):
            axes[0].errorbar(self.x,self.y,yerr=self.error_y,xerr=self.error_x,fmt='ko', ms=ms, capsize=capsize)
        else:
            print("Something is wrong")

        xp = np.linspace(np.min(self.x),np.max(self.x),1000,endpoint=True)
        axes[0].plot(xp,self.func(xp,*self.params),color='b', label = 'Fit')
        axes[0].set_ylabel(ylabel)
        if init:
            axes[0].plot(xp,self.func(xp,*self.initparams),color='r', label='Initial')
        if legend:
            if stats:
                handles, labels = axes[0].get_legend_handles_labels()
                patch = mpatches.Patch(linewidth=0,fill=False,label=f'P-val: {self.Pval:.2e}')
                handles.append(patch) 
                axes[0].legend(handles=handles,loc = legend_loc)
            else:
                axes[0].legend(loc = legend_loc)


        '''Residual plot'''
        axes[1].plot(self.x,self.NormRes, 'ko', ms=ms)
        axes[1].hlines(1,np.min(self.x),np.max(self.x), color = 'r', linestyle='dashed')
        axes[1].hlines(-1,np.min(self.x),np.max(self.x), color = 'r', linestyle='dashed')
        axes[1].set_ylabel('Norm. Res.')
        axes[1].set_xlabel(xlabel)
        
        plt.tight_layout()
        if save != None:
            plt.savefig(save)
        plt.show()

# %%
