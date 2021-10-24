import numpy as np
import math
import scipy
import scipy.linalg



"""
    x_max = x_min + d_x * (x_steps+2)
"""
class EquidistantSpacial:
    def generate_spacial_grid(self, x_min, x_max, x_steps,key_points=None):
        return np.linspace(x_min,x_max,x_steps+2)
    
class EquidistantWithStrikeSpacial:
    def generate_spacial_grid(self, x_min, x_max, x_steps,key_points=None):
        base_points = EquidistantSpacial().GenerateSpacialGrid(x_min, x_max, x_steps)
        if key_points is not None:
            with_key_points = np.concatenate((base_points,key_points))
        else:
            with_key_points = base_points
        return np.array(sorted(with_key_points))
    
"""
    x_max = x_min * factor **(x_steps+2)
    => factor = x_max / x_min 
"""
class ExponentialSpacial:
    def generate_spacial_grid(self, x_min, x_max, x_steps,key_points=None):
        factor = math.exp(  math.log(x_max/x_min) / (x_steps+1) )
        
        tmp = factor**(x_steps+2)
        #print(f"{math.log(factor)}  ~  {math.log(x_max/x_min)/(x_steps+2)}")
        #print(f"factor={factor}")
        #print(f"factor**(x_steps+2)={tmp}")
        #print(f"x_min factor**(x_steps+2)={x_min*tmp}")
        return np.array( [ x_min*factor**i for i in range(0,x_steps+2)])
    

class SteppingConstant:
    def __init__(self, theta = 0.5):
        self.theta = theta
    def stepping(self, n):
        return self.theta
    
class SteppingRannacher:
    def __init__(self, theta=0.5):
        self.theta = theta
    def stepping(self, n):
        if n in [0,1]:
            return 1.0
        return self.theta



def c_j(pde_mu,vol,r,d_x_neg,d_x_pos):
    tmp = ( d_x_pos - d_x_neg ) / d_x_pos / d_x_neg * pde_mu 
    tmp += - 1/d_x_pos/d_x_neg * vol**2 - r
    return tmp
def u_j(pde_mu,vol,d_x_neg,d_x_pos):
    tmp = d_x_neg / ( d_x_pos + d_x_neg ) / d_x_pos * pde_mu
    tmp += 1.0 / ( d_x_pos + d_x_neg ) / d_x_pos * vol**2
    return tmp
def l_j(pde_mu,vol,d_x_neg,d_x_pos):
    tmp = -d_x_pos / ( d_x_pos + d_x_neg ) / d_x_neg * pde_mu
    tmp += 1.0 / ( d_x_pos + d_x_neg ) / d_x_neg * vol**2
    return tmp
    
def make_spacial_matrix(X,r,vol):

    pde_mu = r - 0.5*vol**2

    n = len(X) - 2

    A = np.zeros([n, n], dtype=np.dtype('f8'))
    for j in range(1,n+1):

        d_x_neg = X[j] - X[j-1]
        d_x_pos = X[j+1] - X[j]

        assert(d_x_neg > 0.0)
        assert(d_x_pos > 0.0)

        a_index = j - 1
        A[a_index,a_index] = c_j(pde_mu,vol,r,d_x_neg,d_x_pos)
        if a_index != 0:
            A[a_index,a_index-1] = l_j(pde_mu,vol,d_x_neg,d_x_pos)

        if j != n:
            A[a_index,a_index+1] = u_j(pde_mu,vol,d_x_neg,d_x_pos)

    return A


def pde_crank_nicolson(
    r,
    vol,
    time_to_expiry,
    x_min,
    x_max,
    terminal_condition,
    upper_boundary,
    t_steps,
    x_steps,
    grid_policy,
    stepping_policy,
    key_points=None):
    
    
    
    pde_mu = r - 0.5*vol**2

    TL = np.linspace(0.0,1.0,t_steps)

    X = grid_policy.generate_spacial_grid(x_min,x_max,x_steps,key_points)
    A = make_spacial_matrix(X,r=r,vol=vol)
    V_T = [terminal_condition(x) for x in X[1:-1]]
    S = [ math.exp(x) for x in X[1:-1]]

    upper_d_x_pos = X[-1] - X[-2]
    upper_d_x_neg = X[-2] - X[-3]
    u_m = u_j(pde_mu,vol,upper_d_x_neg,upper_d_x_pos)



    def make_bounary(t):
        B = np.zeros(A.shape[0], dtype=np.dtype('f8'))

        B[-1] = u_m * upper_boundary(t,X[-1])
        return B


    solution_sequence = [ V_T ]
    head = V_T

    problem_seq = []

    for step_index,t_index_and_t in enumerate(list(enumerate(TL))[::-1][1:]):
        t_index,t = t_index_and_t

        theta = stepping_policy.stepping(step_index)

        assert( 0.0 <= theta and theta <= 1.0)



        d_t = TL[t_index+1] - TL[t_index]

        identity = np.identity(A.shape[0], dtype=np.dtype('f8'))
        """ I + ( 1 - \theta)\delta_t A(t_i(theta)) """
        right_op = identity + ( 1 - theta) * d_t * A
        """ I - \theta\delta_t A(t_i(theta)) """
        left_op  = identity - theta * d_t * A

        boundary_cond = d_t * ( ( 1 - theta ) * make_bounary(t+d_t) + theta * make_bounary(t) )

        right_value = np.dot( right_op, head)
        right_value += boundary_cond


        problem_seq.append((left_op,right_value))
        sol = scipy.linalg.solve(left_op, right_value)

        head = sol

        solution_sequence.append(head)
    
    class PDESolution:
        def __init__(child):
            child.solution = np.array(solution_sequence[::-1])
            child.X = X
            child.sol_X = X[1:-1]
            child.S = np.exp(child.sol_X)
            child.TL = TL
            child.problem_seq = problem_seq

            from scipy.interpolate import interp1d
            child.smooth_solution = interp1d(child.S, head)

        def plot(child):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111, projection='3d')

            plot_y_axis = child.TL
            plot_x_axis = [math.exp(x) for x in child.X[1:-1]]

            xx,yy = np.meshgrid(plot_x_axis,plot_y_axis)

            ax.view_init(30, 233)
            surf = ax.plot_surface(xx, 
                                   yy, 
                                   child.solution, 
                                   linewidth=0, antialiased=False,
                                   cmap=plt.cm.coolwarm,
                                   #cmap=cm.CMRmap
                                  )
            plt.show()
            
        def __call__(child, x):
            if type(x) is float:
                return child.smooth_solution(x)[0]
            else:
                return child.smooth_solution(x)



    return PDESolution()

