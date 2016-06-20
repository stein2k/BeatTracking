import numpy as np

def main():

    q = 0.01
    r = 0.02

    Phi = np.array(((1,1),(0,1)),dtype=np.float64)
    Q = q*np.array(((1.0/3.0,1.0/2.0),(1.0/2.0,1.0)),dtype=np.float64)
    H = np.array(((1,0),),dtype=np.float64)
    R = np.array((r,),dtype=np.float64)

    state_actual = [None]*1001
    state_observed = [None]*1001
    state_estimate = [None]*1001
    state_cov_estimate = [None]*1001

    state_actual[0] = np.array(((0,),(1,)),dtype=np.float64)
    state_estimate[0] = np.array(((2*np.random.randn(),),
        (10.*np.random.rand(),)))
    #state_estimate[0] = np.array(((0,),(1,)),dtype=np.float64)
    state_cov_estimate[0] = Q

    print 'state_estimate =', state_estimate[0]

    for n in range(1000):

        state_actual[n+1] = (np.dot(Phi,state_actual[n]) + 
            np.array(((q*np.random.randn(1),),(0.,))))
        state_observed[n+1] = (np.dot(H,state_actual[n+1]) + 
            r*np.random.randn())
    
        x001 = np.dot(Phi,state_estimate[n])
        P001 = (np.dot(np.dot(Phi,state_cov_estimate[n]),
            np.transpose(Phi)) + Q)
        K = np.dot(np.dot(P001,np.transpose(H)),
            np.linalg.inv(np.dot(np.dot(H,P001),np.transpose(H))+
            R))
        state_estimate[n+1] = x001 + np.dot(K,(state_observed[n+1]-np.dot(H,x001)))
        state_cov_estimate[n+1] = np.dot((np.eye(2)-np.dot(K,H)),
            P001)
            
    print 'state_estimate = ', state_estimate[1000]
    print 'state_cov_estimate = ', state_cov_estimate[1000]
    print 'state_actual = ', state_actual[1000]

if __name__ == '__main__':
    main()
