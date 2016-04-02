#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <string>
#include <map>
#include <helper_image.h>     // helper for image and data comparison
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

#define BLOCK_SIZE 32 // Number of threads in x and y direction - Maximum Number of threads per block = 32 * 32 = 1024

// Kernel Definitions

//__global__ void sobel( int xd_size, int yd_size, int maxdval, int d_thresh, unsigned int *input , int *output)
//{

int main()
{
    float start_clock = clock();
//    ofstream f("lx4 - Re300 - Fr300 - results.txt"); // Solution Results
      ofstream f("result_cpu4.txt"); // Solution Results
    f.setf(ios::fixed | ios::showpoint);
    f << setprecision(5);

    ofstream g("lx4 - Re300 - Fr300 - convergence.txt"); // Convergence history
    g.setf(ios::fixed | ios::showpoint);
    g << setprecision(5);
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(5);

    float Re, Pr, Fr, T_L, T_0, T_amb, dx, dy, t, ny, nx, dt, eps, abs, beta, iter, maxiter, tf, st, pold, counter, column, u_wind, T_R, Lx, Ly;

    // Input parameters 
    Lx = 2 * 2.0; Ly = 5.0; // Domain dimensions
    nx = Lx * 2.0; ny = Ly * 2.0; // Grid size - Number of nodes
    u_wind = 1; // Reference velocity
//    viscosity = 0.5*(16.97 + 18.90)*pow(10.0, -6.0); // Fluid viscosity
    st = 0.00005; // Total variance criteria
    eps = 0.001; // Pressure convergence criteria
    tf = 100; // Final time step
    Pr = 0.5*(0.709 + 0.711); // Prandtl number
    Re = 30.0; Fr = 0.3; // Non-dimensional numbers for inflow conditions
    dx = Lx / (nx - 1); dy = Ly / (ny - 1); // dx and dy
    beta = 1.4; // Successive over relaxation factor (SOR)
    t = 0; // Initial time step
    T_L = 100.0; // Left wall temperature (C)
    T_R = 50.0; // Right wall temperature (C)
    T_amb = 25.0; // Ambient air temperature (C)
    T_0 = 50.0; // Initial air temperature
    T_L = T_L + 273.15; T_0 = T_0 + 273.15; T_amb = T_amb + 273.15; T_R = T_R + 273.15;// Unit conversion to (K)
    maxiter = 100; // Maximum iteration at each time step
    counter = 0; // initial row for output monitoring
    column = 1; // Column number for output display

    // Records number of clicks a step takes
    std::map<string, uint32_t> stepTimingAccumulator;



//.................................................GPU Version.................................................


thrust::host_vector<float> u(nx*(ny+1));
thrust::host_vector<float> us(nx*(ny+1));
thrust::host_vector<float> uold(nx*(ny+1));

thrust::host_vector<float> v((nx+1)*ny);
thrust::host_vector<float> vs((nx+1)*ny);
thrust::host_vector<float> vold((nx+1)*ny);

thrust::host_vector<float> p((nx+1)*(ny+1));
thrust::host_vector<float> T((nx+1)*(ny+1));
thrust::host_vector<float> Told((nx+1)*(ny+1));

thrust::host_vector<float> sai(nx*ny);
thrust::host_vector<float> omc_gpu(nx*ny);
thrust::host_vector<float> vc_gpu(nx*ny);
thrust::host_vector<float> uc_gpu(nx*ny);

thrust::host_vector<float> pc_gpu(nx*ny);
thrust::host_vector<float> Tc_gpu(nx*ny);

thrust::device_vector<float> u_h(nx*(ny+1));
thrust::device_vector<float> us_h(nx*(ny+1));
thrust::device_vector<float> uold_h(nx*(ny+1));

thrust::device_vector<float> v_h((nx+1)*ny);
thrust::device_vector<float> vs_h((nx+1)*ny);
thrust::device_vector<float> vold_h((nx+1)*ny);

thrust::device_vector<float> p_h((nx+1)*(ny+1));
thrust::device_vector<float> T_h((nx+1)*(ny+1));
thrust::device_vector<float> Told_h((nx+1)*(ny+1));

thrust::device_vector<float> sai_h(nx*ny);
thrust::device_vector<float> omc_h_gpu(nx*ny);
thrust::device_vector<float> vc_h_gpu(nx*ny);
thrust::device_vector<float> uc_h_gpu(nx*ny);

thrust::device_vector<float> pc_h_gpu(nx*ny);
thrust::device_vector<float> Tc_h_gpu(nx*ny);


int wu, wv, wp, wT, wc;

wu = nx; // number of rows of u vector
wv = nx + 1; // number of rows of v vector
wp = nx + 1; // number of rows of p vector
wT = nx + 1; // number of rows of T vector
//wsai = nx; // number of rows of sai vector
wc = nx; // number of rows of collocated vectors

/*  
    vector<vector<float> > u(nx, vector<float>(ny + 1));
    vector<vector<float> > us(nx, vector<float>(ny + 1));
    vector<vector<float> > uold(nx, vector<float>(ny + 1));

    vector<vector<float> > v(nx + 1, vector<float>(ny));
    vector<vector<float> > vs(nx + 1, vector<float>(ny));
    vector<vector<float> > vold(nx + 1, vector<float>(ny));

    vector<vector<float> > p(nx + 1, vector<float>(ny + 1));
    vector<vector<float> > T(nx + 1, vector<float>(ny + 1));
    vector<vector<float> > Told(nx + 1, vector<float>(ny + 1));

    vector<vector<float> > sai(nx, vector<float>(ny));
    vector<vector<float> > omc_cpu(nx, vector<float>(ny));
    vector<vector<float> > vc_cpu(nx, vector<float>(ny));
    vector<vector<float> > uc_cpu(nx, vector<float>(ny));

    vector<vector<float> > pc_cpu(nx, vector<float>(ny));
    vector<vector<float> > Tc_cpu(nx, vector<float>(ny));
*/


    // Time step size stability criterion

    float mt1 = 0.25*pow(dx, 2.0) / (1.0 / Re); float Rer = 1.0 / Re; float mt2 = 0.25*pow(dy, 2.0) / (1.0 / Re);

    if (mt1 > Rer)
    {
        dt = Rer;
    }
    else
    {
        dt = mt1;
    }

    if (dt > mt2)
    {
        dt = mt2;
    }

    
    //......................................................................................
    // Step 0 - It can be parallelized
    // Initializing the flow variable (Temperature)  
    // Boundary conditions for T (Initialization)
    int step0_start = clock();
    for (int i = 0; i < nx + 1; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            T[i * wT + j] = T_0 / T_amb;
        } // end for j
    } // end for i
    //......................................................................................
    int step0_end = clock();
    stepTimingAccumulator["Step 0, Initializing Temperature"] += step0_end - step0_start;
    //......................................................................................

    // Marching in Time - Outermost loop

    while (t <= tf)
    {

        iter = 0;

        int stepi1_start = clock();
        //........................................................................................
        // Step i1 - it can be parallelized 
        // boundary conditions for u velocity

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny + 1; j++)
            {
                if (i == 0 && j > 0 && j < ny)
                {
                    if (j*dy < 2.0)
                    {
                        u[i * wu + j] = 0; // left wall - Final
                    }
                    else
                    {
                        u[i * wu + j] = u_wind; // left inlet - Final
                    }
                }
                else if (i == nx - 1 && j>0 && j < ny)
                {
                    if (j*dy < 2.0)
                    {
                        u[i * wu + j] = 0; // Right wall has 0 horizontal velocity - Final
                    }
                    else
                    {
                        u[i * wu + j] = u[(i - 1) * wu + j]; // right outlet - no velocity change
                    }
                }
                else if (j == 0)
                {
                    u[i * wu + j] = -u[i * wu + j + 1]; // bottom ghost - Final
                }
                else if (j == ny)
                {
                    u[i * wu + j] = u[i * wu + j - 1]; // upper ghost - Final
                }
            } // end for j
        } // end for i
        int stepi1_end = clock();
        stepTimingAccumulator["Step i1 - Set Horizontal Velocity Boundary Conditions"] += stepi1_end - stepi1_start;
        //...............................................................................................

        
        //.........................................................................................
        // Step i2 - it can be parallelized
        // boundary conditions for v velocity
        int stepi2_start = clock();

        for (int i = 0; i < nx + 1; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (j == 0 && i > 0 && i < nx)
                {
                    v[i * wv + j] = 0; // bottom wall - Final
                }
                else if (j == ny - 1 && i > 0 && i < nx)
                {
                    v[i * wv + j] = v[i * wv + j - 1]; // upper wall - Final
                }
                else if (i == 0)
                {
                    v[i * wv + j] = -v[(i + 1) * wv + j]; // left ghost (Left Wall and inlet has 0 vertical velocity) - Final
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        v[i * wv + j] = -v[(i -1) * wv + j]; // right ghost (Right wall has 0 vertical velocity) - Final
                    }
                    else
                    {
                        v[i * wv + j] = v[(i - 1) * wv + j]; // right outlet - no velocity gradient
                    }
                }
            } // end for j
        } // end for I
        int stepi2_end = clock();
        stepTimingAccumulator["Step i2 - Set Vertical Velocity Boundary Conditions"] += stepi2_end - stepi2_start;
        //...............................................................................................

        //...............................................................................................
        int step1_start = clock();
        //.........................................................................................
        // Step 1 - it can be parallelized - Solve for intermediate velocity values

        // u - us - vh - a 

        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                float vh = 1.0 / 4.0*(v[i * wv + j] + v[(i + 1) * wv + j] + v[i * wv + j - 1] + v[(i + 1) * wv + j - 1]); // v hat
                float a = u[i * wu + j] * 1.0 / (2.0*dx)*(u[(i + 1) * wu + j] - u[(i - 1) * wu + j]) + vh*1.0 / (2.0*dy)*(u[i * wu + j + 1] - u[i * wu + j - 1]); // a
                us[i * wu + j] = dt / Re*(1.0 / pow(dx, 2.0)*(u[(i + 1) * wu + j] - 2.0*u[i * wu + j] + u[(i - 1) * wu + j]) + 1.0 / pow(dy, 2.0)*(u[i * wu + j + 1] - 2.0*u[i * wu + j] + u[i * wu + j - 1])) - a*dt + u[i * wu + j]; // u star
            } // end for j
        } // end for i

        //..........................................................................................
        // Step 1 - it can be parallelized
        // v - vs - uh - b
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                float uh = 1.0 / 4.0*(u[i * wu + j] + u[i * wu + j + 1] + u[(i - 1) * wu + j] + u[(i - 1) * wu + j + 1]);
                float b = uh*1.0 / (2.0*dx)*(v[(i + 1) * wv + j] - v[(i - 1) * wv + j]) + v[i * wv + j] * 1.0 / (2.0*dy)*(v[i * wv + j + 1] - v[i * wv + j - 1]); // b
                vs[i * wv + j] = dt / Re*(1.0 / pow(dx, 2.0)*(v[(i + 1) * wv + j] - 2.0*v[i * wv + j] + v[(i - 1) * wv + j]) + 1.0 / pow(dy, 2.0)*(v[i * wv + j + 1] - 2.0*v[i * wv + j] + v[i * wv + j - 1])) + dt / pow(Fr, 2.0)*(0.5*(T[i * wT + j] + T[i * wT + j - 1]) - 1) / (0.5*(T[i * wT + j] + T[i * wT + j - 1])) - b*dt + v[i * wv + j]; // v 
            } // end for j
        } // end for i

        //...........................................................................................
        // vs and us on Boundary conditions

        for (int i = 0; i < nx; i++)
        {
            us[i * wu + 0] = -us[i * wu + 1]; // bottom ghost - Final
        } // end for j

        //...........................................................................................
        for (int j = 0; j < ny + 1; j++)
        {
            if (j*dy < 2.0)
            {
                us[0 * wu + j] = 0; // left wall - FInal
                us[(nx - 1) * wu + j] = 0; // right wall - Final
            }
            else
            {
                us[0 * wu + j] = u_wind; // left inlet - Final
            }
        }
        //...........................................................................................

        for (int j = 0; j < ny; j++)
        {
            vs[0 * wv + j] = -vs[1 * wv + j]; // left ghost (Both wall and inlet have 0 vs) - Final
            if (j*dy < 2.0)
            {
                vs[nx * wv + j] = -vs[(nx - 1) * wv + j]; // right ghost (Only the right wall - Final
            }
            else
            {
                vs[nx * wv + j] = vs[(nx - 1) * wv + j]; // right outlet - no flux
            }
        }
        //............................................................................................

        for (int i = 0; i < nx + 1; i++)
        {
            vs[i * wv + 0] = 0; // Bottom wall - Final
        } // end for i
        //............................................................................................

        int step1_end = clock();
        stepTimingAccumulator["Step 1 - Solve for intermediate velocities"] += step1_end - step1_start;

        //...............................................................................................
        // Step 2 - It can be parallelized 
        // This is the most expensive part of the code
        // Poisson equation for pressure
        int step2_start = clock();

        float error = 1; iter = 0;

	thrust::device_vector<float> (nx*ny);


        // Solve for pressure iteratively until it converges - Using Gauss Seidel SOR 
        while (error > eps)
        {
            error = 0;
            //............................................................................................
            for (int i = 1; i < nx; i++)
            {
                for (int j = 1; j < ny; j++)
                {
                    pold = p[i * wp + j];
                    p[i * wp + j] = beta*pow(dx, 2.0)*pow(dy, 2.0) / (-2.0*(pow(dx, 2.0) + pow(dy, 2.0)))*(-1.0 / pow(dx, 2.0)*(p[(i + 1) * wp + j] + p[(i - 1) * wp + j] + p[i * wp + j + 1] + p[i * wp + j - 1]) + 1.0 / dt*(1.0 / dx*(us[i * wu + j] - us[(i - 1) * wu + j]) + 1.0 / dy*(vs[i * wv + j] - vs[i * wv + j - 1]))) + (1.0 - beta)*p[i * wp + j];
                    abs = pow((p[i * wp + j] - pold), 2.0);
                    error = error + abs;
                } // end for j
            } // end for i
            //............................................................................................
            // boundary conditions for pressure

            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    if (j == 0)
                    {
                        p[i * wp + j] = p[i * wp + j + 1]; // bottom wall - Final
                    }
                    else if (j == ny)
                    {
                        p[i * wp + j] = p[i * wp + j - 1]; // Upper - no flux
                    }
                    else if (i == 0)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i * wp + j] = p[(i + 1) * wp + j]; // left wall - not the inlet - Final
                        }
                        else
                        {
                            p[i * wp + j] = p[(i + 1) * wp + j];
                        }
                    }
                    else if (i == nx)
                    {
                        if (j*dy < 2.0)
                        {
                            p[i * wp + j] = p[(i - 1) * wp + j]; // right wall - not the outlet - Final
                        }
                        else
                        {
                            p[i * wp + j] = -p[(i - 1) * wp + j]; // pressure outlet - static pressure is zero - Final
                        }
                    }
                } // end for j
            } // end for i
            //................................................................................................

            error = pow(error, 0.5);
            iter = iter + 1;
            if (iter > maxiter)
            {
                break;
            }

        } // end while eps

        int step2_end = clock();
        stepTimingAccumulator["Step 2 - Solve for pressure until tolerance or max iterations"] += step2_end - step2_start;
        //...............................................................................................

        //.................................................................................................
        // Step 3 - It can be parallelized 
        // velocity update - projection method
        int step3_start = clock();

        // u

        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                uold[i * wu + j] = u[i * wu + j];
                u[i * wu + j] = us[i * wu + j] - dt / dx*(p[(i + 1) * wp + j] - p[i * wp + j]);
            } // end for j
        } // end for i
        //................................................

        // v

        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                vold[i * wv + j] = v[i * wv + j];
                v[i * wv + j] = vs[i * wv + j] - dt / dy*(p[i * wp + j + 1] - p[i * wp + j]);
            } // end for j
        } // end for i
        int step3_end = clock();
        stepTimingAccumulator["Step 3 - Velocity Update"] += step3_end - step3_start;
        //...............................................................................................

        //...............................................................................................
        // Step 4 - It can be parallelized
        // Solving for temperature
        int step4_start = clock();
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                Told[i * wT + j] = T[i * wT + j];
                T[i * wT + j] = T[i * wT + j] + dt*(-0.5*(u[i * wu + j] + u[(i - 1) * wu + j])*(1.0 / (2.0*dx)*(T[(i + 1) * wT + j] - T[(i - 1) * wT + j])) - 0.5*(v[i * wv + j] + v[i * wv + j - 1])*(1.0 / (2.0*dy)*(T[i * wT + j + 1] - T[i * wT + j - 1])) + 1 / (Re*Pr)*(1 / pow(dx, 2.0)*(T[(i + 1) * wT + j] - 2.0*T[i * wT + j] + T[(i - 1) * wT + j]) + 1 / pow(dy, 2.0)*(T[i * wT + j + 1] - 2 * T[i * wT + j] + T[i * wT + j - 1])));
            } // end for j
        } // end for i
        int step4_end = clock();
        stepTimingAccumulator["Step 4 - Solving for temperature"] += step4_end - step4_start;
        //................................................................................................
        
        //...............................................................................................
        // Step i3 - Initializing boundary conditions for temperature 
        // boundary conditions for Temperature
        int stepi3_start = clock();

        for (int i = 0; i < nx + 1; i++)
        {
            for (int j = 0; j < ny + 1; j++)
            {
                if (j == 0)
                {
                    T[i * wT + j] = T[i * wT + j + 1]; // bottom wall - Insulated - no flux - Final
                }
                else if (j == ny)
                {
                    T[i * wT + j] = 2.0*(T_0) / T_amb - T[i * wT + j - 1]; // upper boundary - lid with ambient temperature (as air) - Final
                }
                else if (i == 0)
                {
                    if (j*dy < 2.0)
                    {
                        T[i * wT + j] = 2.0*T_L / T_amb - T[(i + 1) * wT + j]; // left wall at T_L - Constant Temperature - Final
                    }
                    else
                    {
                        T[i * wT + j] = 2.0*T_0 / T_amb - T[(i + 1) + j]; // left inlet at T_0 (initial temperature) - Final
                    }
                }
                else if (i == nx)
                {
                    if (j*dy < 2.0)
                    {
                        T[i * wT + j] = 2.0*T_R / T_amb - T[(i - 1) * wT + j]; // right wall at T_R - Final
                    }
                }
            } // end for j
        } // end for i
        int stepi3_end = clock();
        stepTimingAccumulator["Step i3 - Initializing boundary conditions for temperature"] += stepi3_end - stepi3_start;
        //...............................................................................................

        //...............................................................................................
        // Step 5 - Checking if solution reached steady state
        // Checking the steady state condition
        int step5_start = clock();

        float TV, abs; TV=0; // float abs, TVt, TV2, TV3; TV = 0; TV2 = 0; TV3 = 0; float abs, abs2, abs3;
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 2; j++)
            {
                abs = v[i * wv + j] - vold[i * wv + j];
                TV = TV + pow(pow(abs, 2), 0.5);
            } // end for i
        } // end for j

        TV = TV / ((nx - 1)*(ny - 2));
	
	float st_time;
        if (TV < st && error < eps)
        {
            st_time = t;
            break;
        }
        counter = counter + 1;
        if (fmod(counter, 10) == 0 || counter == 1)
        {
            //cout << "" << endl;
            //cout << "Column" << setw(30) << "time(s)" << setw(30) << "Iterations on Pressure" << setw(30) << "Pressure Residual" << setw(30) << "Total Variance" << endl;
        } // end if
        int step5_end = clock();
        stepTimingAccumulator["Step 5 - Check for steady state"] += step5_end - step5_start;
        //...............................................................................................


        //cout << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
        g << column << setw(30) << t << setw(30) << iter << setw(30) << error << setw(30) << TV << endl;
        t = t + dt;
        column = column + 1;

    } // end while time

    //........................................................................................................

    // Step 6
    // Co-locate the staggered grid points 
    int step6_start = clock();
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            vc_gpu[i * wc + j] = 1.0 / 2.0*(v[(i + 1) * wv + j] + v[i * wv + j]);
            pc_gpu[i * wc + j] = 1.0 / 4.0*(p[i * wp + j] + p[(i + 1) * wp + j] + p[i * wp + j + 1] + p[(i + 1) * wp + j + 1]);
            uc_gpu[i * wc + j] = 1.0 / 2.0*(u[i * wu + j] + u[i * wu + j + 1]);
            omc_gpu[i * wc + j] = 1.0 / dx*(v[(i + 1) * wv + j] - v[i * wv + j]) - 1.0 / dy*(u[i * wu + j + 1] - u[i * wu + j]);
            Tc_gpu[i * wc + j] = 1.0 / 4.0*(T[i * wT + j] + T[(i + 1) * wT + j] + T[i * wT + j + 1] + T[(i + 1) * wT + j + 1]);

        } // end for j
    } // end for i
    //........................................................................................................
    int step6_end = clock();
    stepTimingAccumulator["Step 6 - Co-locate staggered grid points"] += step6_end - step6_start;

    // Steady state results

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            f << setw(15) << t - dt << setw(15) << i*dx << setw(15) << j*dy << setw(15) << uc_gpu[i * wc + j] << setw(15) << vc_gpu[i * wc + j] << setw(15) << pc_gpu[i * wc + j] << setw(15) << Tc_gpu[i * wc + j] * T_amb - 273.15 << setw(15) << omc_gpu[i * wc + j] << endl;
        } // end for i
    } // end for j
    //.........................................................................................................

    float end_clock = clock();

    for (auto it = stepTimingAccumulator.begin(); it != stepTimingAccumulator.end(); it++)
    {
        float seconds = (float)it->second / CLOCKS_PER_SEC;
        std::cout << it->first << "\t" << seconds << endl;
    }
//.................................................End of GPU Version..........................................


// Compare CPU and GPU results
/*  bool success = true; 
 for (int j = 0; j < ny; j++)
  {
      for (int i = 0; i < nx; i++)
      {
	  if (uc_cpu[i][j] != uc_cpu[i][j] || vc_cpu[i][j] != vc_cpu[i][j] || pc_cpu[i][j] != pc_cpu[i][j] || Tc_cpu[i][j] != Tc_cpu[i][j])
          {
             success = false;
          }
      }
 }
	if (success == true)
	{ 
	   printf("\n");
	   printf("*** kernel PASSED ***\n"); //, kernelName);
	   printf("The outputs of CPU version and GPU version are identical.\n");
	}
	else
	{  
	   printf("\n");
	   printf("*** kernel FAILED ***\n"); //, kernelName);
	}
*/
    cout << "" << endl;
    cout << "Steady state time = " << t << " (s) " << endl;
    cout << "GPU time = " << (end_clock - start_clock) / CLOCKS_PER_SEC << " (s)" << endl;

    return 0;
} // end main
