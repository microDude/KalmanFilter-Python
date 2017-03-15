'''
Created on Feb 16, 2013
@author: Gregory
Modified Kalman Filter Design
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import sleep as pause

# intial parameters
onlinePlotFlag = False
x = np.sin(np.arange(0,9*np.pi,0.1)) + 5                  # true value
x = x + np.arange(0,5,(5-0) / (np.size(x, 0)))
numSample = np.size(x, 0)        # Number of samples
print('Number of sample points = ',numSample)
z = np.random.normal(x,0.15,size = numSample) # observations (normal about x, sigma=0.1)

# allocate space for arrays
x_est = np.zeros(numSample)      # a posteri estimate of x
P = np.zeros(numSample)         # a posteri error estimate
x_estminus = np.zeros(numSample) # a priori estimate of x
Pminus = np.zeros(numSample)    # a priori error estimate
K = np.zeros(numSample)         # gain or blending factor

Q = 5e-4   # process variance
R = 0.05**2 # estimate of measurement variance, change to see effect

# intial values (guesses)
x_est[0] = 0.0
P[0] = Q + 1

for k in range(1,numSample):
    
    # time update
    x_estminus[k] = x_est[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    x_est[k] = x_estminus[k] + K[k]*(z[k] - x_estminus[k])
    P[k] = (1-K[k])*Pminus[k]
    
    # Online Plotting
    if onlinePlotFlag == True:
        if (k == 1):
            # First, need to create the template for the figure() on the first iteration
            fig = plt.figure(99)
            
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(x_est,'.')
            ax1.set_title('x_est for iteration: ' + str(k))
            ax1.set_xlabel('Sample Index [j]')
            ax1.set_ylabel('x_est')
            
            ax2 = fig.add_subplot(2,1,2)
            ax2.loglog(P)
            ax2.set_xlabel('iteration [k]')
            ax2.set_ylabel('P(x)')
            ax2.grid(True)
            
            plt.show(block=False) # block = False, means that it will draw and continue
            
        else:
            # Update the figure
            print('k =',k)
            
            # Clear the previous plots
            ax1.clear()
            ax2.clear()
            
            # Replot the update
            ax1.plot(x_est,'.')
            ax1.set_title('x_est for iteration: ' + str(k))
            ax2.loglog(P,'b')
            
            # Draw and Pause
            fig.canvas.draw()
            pause(1e-2)
            
    else: # Just to make sure that everything is running
        print('k =',k)


# Plot the results
plt.figure(2)
plt.plot(z,'k.',label='noisy measurements')
plt.plot(x_est,'b-',label='a posteri estimate')
plt.plot(x,'g',label='true value')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Estimate')
plt.title('Online Estimate for x \n' + 'Using first order Kalman Filter')
plt.show(block = False)

plt.figure(3)
plt.loglog(Pminus,label='a priori error estimate')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Online Error Estimate')
plt.legend()
plt.show()