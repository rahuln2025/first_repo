##### first_repo
### My worked notebooks as per the course: Practical Numerical Computing with Python by Prof. Lorena Barba from the University of Washington

Link to the course material : https://github.com/numerical-mooc/numerical-mooc

The course is aimed for first-year graduate and senior undergraduate students. It gives a foundation in scientific computing and introduces numerical methods for solving differential equations. 
The course material is available in the form of interactive jupyter notebooks with code blocks, along with references for further reading.

Here is a summary of the modules and the code snippets from my assignment solutions: 

## **Module 1: The Phugoid Model of Glider Flight**

- _Problem_: Phugoid model of glider flight described by set of two nonlinear ODEs.
- _Topics_: a) Euler's method, 2nd-order RK, and leapfrog; b) consistency, convergence testing; c) stability Computational techniques: array operations with NumPy; symbolic computing with SymPy; ODE integrators and libraries; writing and using functions.

## **Module 2: Space and Time - Introduction to finite-difference solutions of PDEs**

- _Problem_: Linear convection equation in one dimension.
- _Topics_: Finite differencing in PDEs, CFL condition, numerical diffusion, accuracy of finite difference approximations via Tyalor sieries, consistency and stability, introduction to conservation laws, array operations with Numpy, symbolic computing with SymPy. 

## **Module 3: Riding the wave: Convection Problems**

- _Problem_: Traffic-flow model to study different solutions for problems with shocks.
- _Topics_: Upwind scheme, Lax-Friedrichs scheme, Lax-Wendroff scheme, MacCormack scheme, MUSCL scheme.

- _Code Snippet_: 

   ![image](https://user-images.githubusercontent.com/28657501/148097851-3b7d6ecd-e6fc-4024-90a6-09a0436ecb8e.png)
   ![image](https://user-images.githubusercontent.com/28657501/148097952-8901391c-a982-4c24-8fe8-5aeb163249bd.png)
   ![image](https://user-images.githubusercontent.com/28657501/148101704-82e422bf-5d7a-47fc-adf7-0a637cc43b4f.png)
   ![image](https://user-images.githubusercontent.com/28657501/148101289-848815ab-703b-422d-873c-b5c8945e1e33.png)
   
   **Key Functions**
    ```Python
       #calculate flux for the Richtmyer scheme
       def flux(U):

            """
            Calculates the flux based on u_bar array

            U = 2D numpy array, u_bar values at all x loactions at a time instant
            ------
            returns: 
            F = 2D numpy array, f_bar values at all x locations at a time instant
            """

            F = numpy.array([U[1],
                            ((U[1]**2)/U[0]) + (0.4 * (U[2] - 0.5*((U[1]**2)/U[0]))),
                            (U[2] + 0.4 * (U[2] - (0.5* (U[1]**2/U[0])))) * (U[1]/U[0])])
        return F 
    
        #implementing the Richtmeyr scheme
        def richtmyer(U0, nt, dt, dx):


            U_hist = [U0.copy()]
            U = U0.copy()
            U_star = U.copy() #here U_star = U(n+1/2) values

            for n in range(nt):

                F = flux(U)
                U_star[:,1:] = 0.5*(U[:,1:] + U[:,:-1]) - (dt/(2*dx))*(F[:,1:] - F[:,:-1])
                F = flux(U_star)
                #print(F)
                U[:,:-1] = U[:,:-1] - (dt/dx)*(F[:,1:] - F[:,:-1])
                
                U_hist.append(U.copy())
    
        return U_hist
        
        ```
 _Result snap_:
        
 ![image](https://user-images.githubusercontent.com/28657501/148103938-2162458f-c3f9-4b28-914c-31a5db31dcb4.png)




## **Module 4: Spreading out: Diffusion Problems**

- _Problem_: Diffusion (heat) equation as a problem to solve parabolic PDEs. Start with 1D heat equaiton and move to 2D heat equation.
- _Topics_: Implicit and Explicit Schemes with boundary condition implementation, Crank-Nocolson Scheme

- _Code Snippet_: 
   
  ![image](https://user-images.githubusercontent.com/28657501/148088959-4e4a7b1d-f372-4236-9161-e56512f26b5f.png)
  ![image](https://user-images.githubusercontent.com/28657501/148089684-cf383b2a-6521-4f23-8bab-7943b0e1ebf0.png)

    ```Python
    import urllib.request

    # Download and read the data file.
    url = ('https://github.com/numerical-mooc/numerical-mooc/blob/master/'
           'lessons/04_spreadout/data/uvinitial.npz?raw=true')
    filepath = 'uvinitial.npz'
    urllib.request.urlretrieve(url, filepath);
    # Read the initial fields from the file.
    uvinitial = numpy.load(filepath)
    u0, v0 = uvinitial['U'], uvinitial['V']
    # Plot the initial fields.
    fig, ax = pyplot.subplots(ncols=2, figsize=(9.0, 4.0))
    ax[0].imshow(u0, cmap=cm.RdBu)
    ax[0].axis('off')
    ax[1].imshow(v0, cmap=cm.RdBu)
    ax[1].axis('off');
    ```
    ![image](https://user-images.githubusercontent.com/28657501/148091492-e93e8409-d403-42ff-91e8-6e32943cfdbe.png)
    
    **Key function**
    ```Python
    def ftcs_rxn_dfn(u, v, dx, dy, dt, nt, Du, Dv, F, k):
    u = u0.copy()
    v = v0.copy()
    u_hist = [u0]
    v_hist = [v0]

    for n in range(1,nt):
        un = u.copy()
        vn = v.copy()
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt*(Du*((un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])/(dx**2) + (un[2:,1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])/(dy**2)) - un[1:-1, 1:-1]*(vn[1:-1, 1:-1]**2) + F*(1 - un[1:-1, 1:-1]))
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt*(Dv*((vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])/(dx**2) + (vn[2:,1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])/(dy**2)) + un[1:-1, 1:-1]*(vn[1:-1, 1:-1]**2) - vn[1:-1, 1:-1]*(F + k))
        #Neumann boundary condition on all boundaries
        #for left boundary
        u[:, 0] = u[:, 1]
        v[:, 0] = v[:, 1]
        #for bottom boundary
        u[0, :] = u[1, :]
        v[0, :] = v[1, :]
        #for right boundary
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        #for top boundary
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        #storing values for all time steps
        u_hist.append(u.copy())
        v_hist.append(v.copy())
    return u_hist, v_hist
    
    ```
    _Result animation_: 
    
 https://user-images.githubusercontent.com/28657501/148093795-5d4aee06-4e21-4889-8fb4-c11253080e10.mp4





## **Module 5: Relax and hold stead: Elliptic Problems**

- _Problem_: Laplace and Poisson equations to be solved by iterative methods
- _Topics_: Iterative methods for solving algebraic equations based on disctretization of PDEs, namely, Jacobi method, Gauss-Siedel, Successive over-relaxation and conjugate gradient method.

- _Code Snippet_:

    ![image](https://user-images.githubusercontent.com/28657501/147852530-2ff96d43-9dcd-400f-b4cf-5d1ecdd47ba7.png)

**Key function**
```Python
    @jit(nopython=True)
    def cavity_gauss_siedel1(psi0, omega0, dx, maxiter=20000, rtol = 1e-6):

        nx, ny = psi0.shape
        psi = psi0.copy()
        omega = omega0.copy()
        diff_psi = 1.0
        diff_omega = rtol + 1.0
        ite = 0
        conv_psi = []
        conv_omega = []

        while diff_omega>rtol or diff_psi>rtol and ite<maxiter:

            #Setting up the iterations
            psi_n = psi.copy()

            #iterating for psi
            for j in range(1, ny-1):
                for i in range(1, ny-1):
                    psi[j, i] = 0.25*(psi[j-1, i] + psi[j+1, i] + psi[j, i-1] + psi[j, i+1] + omega[j, i]*(dx**2))

            diff_psi = numpy.sum(numpy.abs(psi[1:-1, 1:-1] - psi_n[1:-1, 1:-1]))

            #iterating for omega
            omega_n = omega.copy()

            for j in range(1, ny-1):
                for i in range(1, ny-1):
                    omega[j, i] = 0.25*(omega[j-1, i] + omega[j+1, i] + omega[j, i-1] + omega[j, i+1])

            #Applying boundary conditions (acc to book ref)
            #left bc
            for j in range(1, ny-1):
                omega[j, 0] = (-2/(dx**2))*psi[j, 1]

            #bottom bc
            for i in range(1, nx-1):
                omega[0, i] = (-2/(dx**2))*psi[1, i]

            #right bc
            for j in range(1, ny-1):
                omega[j, -1] = (-2/(dx**2))*psi[j, -2]

            #top bc
            for i in range(1, nx-1):
                omega[-1, i] = -2/dx - (2/(dx**2))*psi[-2, i]

            diff_omega = numpy.sum(numpy.abs(omega - omega_n))

            #recording l1_norms at each iteration
            conv_psi.append(diff_psi)
            conv_omega.append(diff_omega)

            ite = ite + 1

        return psi, omega, ite, conv_psi, conv_omega
    
  ```
 _Result plot_:
 
   ![image](https://user-images.githubusercontent.com/28657501/147852818-f69099fe-1a82-41fb-9a01-3ea436a61365.png)
