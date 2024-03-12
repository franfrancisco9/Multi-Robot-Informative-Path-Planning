import GPy
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker, colors

class GPModel():
    '''
    Inspired from informative-path-planning github.
    TODO -> Put Link

    This class works as a wrappes on top of GPy.
    
    Inputs:
    @param ranges
        x and y limits for graphs and workspace
    @param variance (float) 
        the variance parameter of the squared exponential kernel
    @param lengthscale (float) 
        the lengthscale parameter of the squared exponential kernel
    @param noise (float) 
        the sensor noise parameter of the squared exponential kernel
    @param dimension (int) 
        the dimension of the environment 
        2D only currently
    @param kernel (string) 
        the type of kernel 
        rbg only currently
    '''  
    def __init__(self, ranges = [0, 40, 0, 40], variance = 100, 
                    lengthscale = 1.0, noise = 0.0001, dimension = 2, kernel = 'rbf'):
        '''
        Iniatilze the class's attributes
        '''

        self.ranges = ranges
        self.variance = variance
        self.lenghtscale = lengthscale
        self.noise = noise

        if dimension == 2:
            self.dim = dimension
        else:
            raise ValueError('Environment must have dimension 2')
        
        self.kernel = kernel

        if kernel == 'rbf':
            self.kern = GPy.kern.RBF(input_dim = self.dim, 
                                    variance = self.variance,
                                    lengthscale = self.lenghtscale)
        else:
            raise ValueError('Kernel type must be rbf')
        
        self.model = None
        self.xvals = None
        self.zvals = None
         
    def predict_value(self, xvals):
        ''' Public method returns the mean and variance predictions at a set of input locations.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        
        Returns: 
        * mean (float array): an nparray of floats representing predictive mean, with dimension NUM_PTS x 1         
        * var (float array): an nparray of floats representing predictive variance, with dimension NUM_PTS x 1 '''        

        assert(xvals.shape[0] >= 1)            
        assert(xvals.shape[1] == self.dim)    
        
        n_points, input_dim = xvals.shape
        
        # With no observations, predict 0 mean everywhere and prior variance
        if self.model == None:
            return np.zeros((n_points, 1)), np.ones((n_points, 1)) * self.variance
        
        # Else, return 
        mean, var = self.model.predict(xvals, full_cov = False, include_likelihood = True)
        return mean, var        
    
    def set_data(self, xvals, zvals):
        ''' Public method that updates the data in the GP model.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        * zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 ''' 
        
        # Save the data internally
        self.xvals = xvals
        self.zvals = zvals
        
        # If the model hasn't been created yet (can't be created until we have data), create GPy model
        if self.model == None:
            self.model = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
        # Else add to the exisiting model
        else:
            self.model.set_XY(X = np.array(xvals), Y = np.array(zvals))
    
    def add_data(self, xvals, zvals):
        ''' Public method that adds data to an the GP model.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        * zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 ''' 
        
        if self.xvals is None:
            self.xvals = xvals
        else:
            self.xvals = np.vstack([self.xvals, xvals])
            
        if self.zvals is None:
            self.zvals = zvals
        else:
            self.zvals = np.vstack([self.zvals, zvals])

        # If the model hasn't been created yet (can't be created until we have data), create GPy model
        if self.model == None:
            self.model = GPy.models.GPRegression(np.array(self.xvals), np.array(self.zvals), self.kern)
        # Else add to the exisiting model
        else:
            self.model.set_XY(X = np.array(self.xvals), Y = np.array(self.zvals))
            
    def simulate_prediction(self, xvals_add, zvals_add, xvals_test):
        ''' Public method that adds data to an the GP model.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        * zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1 ''' 
        
        if self.xvals is None:
            xvals = xvals_add
        else:
            xvals = np.vstack([self.xvals, xvals_add])
            
        if self.zvals is None:
            zvals = zvals_add
        else:
            zvals = np.vstack([self.zvals, zvals_add])


        temp_model = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
        mean, var = temp_model.predict(xvals_test, full_cov = False, include_likelihood = True)

        return mean, var                   

    def load_kernel(self, kernel_file = 'kernel_model.npy'):
        ''' Public method that loads kernel parameters from file.
        Inputs:
        * kernel_file (string): a filename string with the location of the kernel parameters '''    
        
        # Read pre-trained kernel parameters from file, if avaliable and no training data is provided
        if os.path.isfile(kernel_file):
            print ("Loading kernel parameters from file")
            self.kern[:] = np.load(kernel_file)
        else:
            raise ValueError("Failed to load kernel. Kernel parameter file not found.")
            
        return

    def train_kernel(self, xvals = None, zvals = None, kernel_file = 'kernel_model.npy'):
        ''' Public method that optmizes kernel parameters based on input data and saves to files.
        Inputs:
        * xvals (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2
        * zvals (float array): an nparray of floats representing sensor observations, with dimension NUM_PTS x 1        
        * kernel_file (string): a filename string with the location to save the kernel parameters '''      
        
        # Read pre-trained kernel parameters from file, if avaliable and no training data is provided
        if xvals is not None and zvals is not None:
            print ("Optimizing kernel parameters given data")
            # Initilaize a GP model (used only for optmizing kernel hyperparamters)
            self.m = GPy.models.GPRegression(np.array(xvals), np.array(zvals), self.kern)
            self.m.initialize_parameter()

            # Constrain the hyperparameters during optmization
            self.m.constrain_positive('')
            #self.m['rbf.variance'].constrain_bounded(0.01, 10)
            #self.m['rbf.lengthscale'].constrain_bounded(0.01, 10)
            self.m['Gaussian_noise.variance'].constrain_fixed(self.noise)

            # Train the kernel hyperparameters
            self.m.optimize_restarts(num_restarts = 2, messages = True)

            # Save the hyperparemters to file
            np.save(kernel_file, self.kern[:])
        else:
            raise ValueError("Failed to train kernel. No training data provided.")
            
    def visualize_model(self, x1lim, x2lim, title = ''):
        if self.model is None:
            print ('No samples have been collected. World model is equivalent to prior.')
            return None
        else:
            print ("Sample set size:", self.xvals.shape)
            fig = self.model.plot(figsize=(8, 6), title = title, xlim = x1lim, ylim = x2lim)
            
    def kernel_plot(self):
        ''' Visualize the learned GP kernel '''        
        _ = self.kern.plot()
        plt.ylim([-10, 10])
        plt.xlim([-10, 10])
        plt.show()

# Example usage
if __name__ == "__main__":
    from ipp import InformativePathPlanning
    from radiation import RadiationField
    scenario = RadiationField(num_sources=2, workspace_size=(40, 40))
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    GP = GPModel()
    x1 = np.linspace(0, 40, 20)
    x2 = np.linspace(0, 40, 20)
    x1vals, x2vals = np.meshgrid(x1, x2, sparse = False, indexing = 'xy') # dimension: NUM_PTS x NUM_PTS
    # print(x1vals, x2vals)
    data = np.vstack([x1vals.ravel(), x2vals.ravel()]).T # dimension: NUM_PTS*NUM_PTS x 2
    
    xsamples = np.reshape(np.array(data[0, :]), (1, GP.dim)) # dimension: 1 x 2        
    mean, var = GP.predict_value(xsamples)   
    print(mean, var)
    seed = 95789

    zsamples = np.random.normal(loc = 0, scale = np.sqrt(var))
    zsamples = np.reshape(zsamples, (1,1)) # dimension: 1 x 1 

    # Add new data point to the GP model
    waypoints = ipp.nominal_spread
    measurements = scenario.simulate_measurements(waypoints)
    GP.set_data(waypoints, np.reshape(np.array(measurements), (len(measurements), 1))) # dimension: NUM_PTS x 2

    # Iterate through the rest of the grid sequentially and sample a z values, condidtioned on previous samples
    for index, point in enumerate(data[1:, :]):
        # Get a new sample point
        xs = np.reshape(np.array(point), (1, GP.dim))

        # Compute the predicted mean and variance
        mean, var = GP.predict_value(xs)
        
        # Sample a new observation, given the mean and variance
        if seed is not None:
            np.random.seed(seed)
            seed += 1            
        zs = scenario.simulate_measurements(xs, noise_level = np.sqrt(var))
        
        # Add new sample point to the GP model
        zsamples = np.vstack([zsamples, np.reshape(zs, (1, 1))])
        xsamples = np.vstack([xsamples, np.reshape(xs, (1, GP.dim))])
        GP.set_data(xsamples, zsamples)
    
    visualize = True
    # Plot the surface mesh and scatter plot representation of the samples points
    if visualize == True:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_title('Surface of the Simulated Environment')
        surf = ax.plot_surface(x1vals, x2vals, zsamples.reshape(x1vals.shape), cmap = cm.coolwarm, linewidth = 1)
        
        #ax2 = fig.add_subplot(212, projection = '3d')
        Z_true = scenario.ground_truth()
        if Z_true.max() == 0:
            max_log_value = 1
        else:
            max_log_value = np.ceil(np.log10(Z_true.max()))
        levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
        cmap = plt.get_cmap('Greens_r', len(levels) - 1)
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Countour Plot of the Simulated Environment')     
        plot = ax2.contourf(x1vals, x2vals, zsamples.reshape(x1vals.shape), levels = levels, cmap = cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
        scatter = ax2.scatter(data[:, 0], data[:, 1], c = zsamples.ravel(), s = 4.0, cmap = cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
        maxind = np.argmax(zsamples)
        ax2.scatter(xsamples[maxind, 0], xsamples[maxind,1], color = 'k', marker = '*', s = 500)
        plt.show()           
    
    print ("Environment initialized with bounds X1: (", 0, ",", 40, ")  X2:(", 0, ",", 40, ")" )