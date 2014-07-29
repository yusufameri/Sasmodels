__kernel void CylinderKernel(__global const real *qx, global const real *qy, __global real *_ptvalue, __global const real *zip,
const real sub, const real scale, const int count, const int size, const int N)
{
    real weight[N] = {};
    real radius[N] = {};
    real length[N] = {};
    real cyl_theta[N] = {};
    real cyl_phi[N] = {};

    for(k=0;k<N;k++){
        weight[k] = zip[k];
        radius[k] = zip[k+N];
        length[k] = zip[k+2*N];
        cyl_theta[k] = zip[k+3*N];
        cyl_phi[k] = zip[k+4*N];
    }

    int i = get_global_id(0);
    int j = get_global_id(1);

    if(i < count)
    {
        real qq = sqrt(qx[i]*qx[i]+qy[i]*qy[i]);

        real pi = 4.0*atan(1.0);
        real theta = cyl_theta[j]*pi/180.0;
        real phi = cyl_phi[j]*pi/180.0;

        real cyl_x = cos(theta)*cos(phi);
        real cyl_y = sin(theta);
        real cos_val = cyl_x*(qx[i]/qq) + cyl_y*(qy[i]/qq);

        real alpha = acos(cos_val);
        if(alpha == 0.0){
            alpha = 1.0e-26;
        }
        real besarg = qq*radius[j]*sin(alpha);
        real siarg = qq*length[j]/2*cos(alpha);
        real be=0.0; real si=0.0;

        real bj = NR_BessJ1(besarg);

        real d1 = qq*radius[j]*sin(alpha);

        if (besarg == 0.0){
            be = sin(alpha);
        }
        else{
            be = bj*bj*4.0*sin(alpha)/(d1*d1);
        }
        if(siarg == 0.0){
            si = 1.0;
        }
        else{
            si = sin(siarg)*sin(siarg)/(siarg*siarg);
        }

        real form = be*si/sin(alpha);
        real answer = sub*sub*form*acos(-1.0)*radius[j]*radius[j]*length[j]*1.0e8*scale;

        real ret = radius_weight*length_weight*theta_weight*phi_weight*answer*pow(radius[j],2)*length[j];
        if (size>1) {
            ret *= fabs(cos(cyl_theta[j]*pi/180.0));
        }
        //_ptvalue[i] = ret;  // return ptvalue for each p and accumulate on host
        _ptvalue[i] += ret;   // accumulate in kernel, initializing ptvalue before loop and return at end of loop
    }
}





























