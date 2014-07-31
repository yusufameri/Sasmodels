
real f(real qx, real qy, real cyl_theta, real cyl_phi,
       real radius, real length, const real sub, const unsigned int size, const real scale)
{
    real qq = sqrt(qx*qx+qy*qy);

    real pi = 4.0*atan(1.0);
    real theta = cyl_theta*pi/180.0;
    real phi = cyl_phi*pi/180.0;

    real cyl_x = cos(theta)*cos(phi);
    real cyl_y = sin(theta);
    real cos_val = cyl_x*(qx/qq) + cyl_y*(qy/qq);

    real alpha = acos(cos_val);
    if(alpha == 0.0)
    {
        alpha = 1.0e-26;
    }
    real besarg = qq*radius*sin(alpha);
    real siarg = qq*length/2*cos(alpha);
    real be=0.0; real si=0.0;

    real bj = NR_BessJ1(besarg);

    real d1 = qq*radius*sin(alpha);

    if (besarg == 0.0)
    {
        be = sin(alpha);
    }
    else
    {
        be = bj*bj*4.0*sin(alpha)/(d1*d1);
    }
    if(siarg == 0.0)
    {
        si = 1.0;
    }
    else
    {
        si = sin(siarg)*sin(siarg)/(siarg*siarg);
    }

    real form = be*si/sin(alpha);
    real answer = sub*sub*form*acos(-1.0)*radius*radius*length*1.0e8*scale;

    real ret = answer*pow(radius,2)*length;
    if (size>1)
    {
        ret *= fabs(cos(cyl_theta*pi/180.0));
    }

    return ret;
}

__kernel void CylinderKernel(
                            __global const real *qx,
                            __global const real *qy,
                            __global real *_ptvalue,
                            // const unsigned int count,
                            __global const real *zip,
                            const real sub,
                            const real scale,
                            const unsigned int size,
                            const unsigned int N)
{

    int i = get_global_id(0);
    int count = get_global_size(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);
    real ret = 0.0;
#if 0 //This is with private memory
    if (i < count) {
        for (int k=0; k<N; k++) {
            real w = zip[k];
            if (w != 0.0) {
                real radius = zip[k+N];
                real length = zip[k+2*N];
                real cyl_theta = zip[k+3*N];
                real cyl_phi = zip[k+4*N];
                real fk = f(qx[i], qy[i], cyl_theta, cyl_phi, radius, length, sub, size, scale);
                ret += w*fk;
            }
        }
        _ptvalue[i] = ret;
    }
#else //This is with local memory

    //MAX_NLOC is the work group size. In the code_cylinder it is 64
    __local real weight[MAX_NLOC];
    __local real radius[MAX_NLOC];
    __local real length[MAX_NLOC];
    __local real cyl_theta[MAX_NLOC];
    __local real cyl_phi[MAX_NLOC];

    for(int k=0;k<N;k+=nloc)
    {
        //loop and invocate for every work-item in the specific work group
        weight[iloc] = zip[k+iloc];
        radius[iloc] = zip[k+N+iloc];
        length[iloc] = zip[k+2*N+iloc];
        cyl_theta[iloc] = zip[k+3*N+iloc];
        cyl_phi[iloc] = zip[k+4*N+iloc];
        barrier(CLK_LOCAL_MEM_FENCE);
        if(i < count) //if we are working inside of a work-group
        {
            //_ptvalue[i] = ret;  // return ptvalue for each p and accumulate on host
            // accumulate in kernel, initializing ptvalue before loop and return at end of loop
            for(int kpart=0;kpart<nloc;kpart++)
            {
                if (weight[kpart] != 0)
                {
                    real fk = f(qx[i], qy[i], cyl_theta[kpart], cyl_phi[kpart], radius[kpart], length[kpart],sub, size, scale);
                    ret += weight[kpart]*ret;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i<count) {
       _ptvalue[i] += ret;
    }
#endif
}