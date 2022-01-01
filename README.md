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

## **Module 4: Spreading out: Diffusion Problems**

- _Problem_: Diffusion (heat) equation as a problem to solve parabolic PDEs. Start with 1D heat equaiton and move to 2D heat equation.
- _Topics_: Implicit and Explicit Schemes with boundary condition implementation, Crank-Nocolson Scheme

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
