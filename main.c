#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

/*
 * two sensors case:
 *
 * state variable stays same: x = [x, v]
 *
 * mesurement vector z = [z1, z2]
 *
 * measurement model matrix H = [1, 0; 1, 0] (both measure position)
 *
 * measurement noise covariance R = [r, 0; 0, r] (both r as they are identical sensors)
 *
 * process noise covariance Q = [q, 0; 0, q2] (q2 is smaller than q)
 *
 * Prediction step:
 *
 * state prediction x = F * x
 * covariance prediction P = F * P * FT + Q
 * 
 * Update step:
 *
 * measurement residual (innovation) y = z - H * x
 * innovation covariance S = H * P * HT + R
 * Kalman gain K = P * HT * S^-1
 * state update x = x + K * y
 * covariance update P = (I - K * H) * P
 */

struct kalman {
    double k[2][2];       /* kalman gain */
    double p[2][2];       /* covariance matrix */
    double x[2];       /* extimate of position / offset */
    double r[2][2];       /* measurement variance */
    double q[2][2];       /* process variance */
    int64_t old_t;
};

void kalman_init(struct kalman *k, double r_init, double q_init, double p_init)
{
    k->k[0][0] = 0; k->k[1][1] = 0; k->k[0][1] = 0; k->k[1][0] = 0;
    k->p[0][0] = p_init; k->p[1][1] = p_init; k->p[0][1] = 0; k->p[1][0] = 0;
    k->x[0] = 0; k->x[1] = 0;
    k->q[0][0] = q_init; k->q[1][1] = q_init * 0.0001; k->q[0][1] = 0; k->q[1][0] = 0;
    k->r[0][0] = r_init;
    k->r[0][1] = 0;
    k->r[1][0] = 0;
    k->r[1][1] = r_init * 0.1;
    k->old_t = 0;
}

void matrix_mult_2x2(double a[2][2], double b[2][2], double c[2][2])
{
    double d[2][2];

    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];

    c[0][0] = d[0][0];
    c[0][1] = d[0][1];
    c[1][0] = d[1][0];
    c[1][1] = d[1][1];
}

void matrix_mult_2x2_2x1(double a[2][2], double b[2], double c[2])
{
    c[0] = a[0][0] * b[0] + a[0][1] * b[1];
    c[1] = a[1][0] * b[0] + a[1][1] * b[1];
}

void matrix_transpose_2x2(double a[2][2], double c[2][2])
{
    double x[2][2];

    x[1][0] = a[0][1];
    x[0][1] = a[1][0];
    c[1][0] = x[1][0];
    c[0][1] = x[0][1];
}

void matrix_invert_2x2(double a[2][2], double c[2][2])
{
    double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    double x[2][2];

    x[0][0] = a[1][1] / det;
    x[0][1] = -a[0][1] / det;
    x[1][0] = -a[1][0] / det;
    x[1][1] = a[0][0] / det;

    c[0][0] = x[0][0];
    c[0][1] = x[0][1];
    c[1][0] = x[1][0];
    c[1][1] = x[1][1];
}

void kalman_update(struct kalman *k, double z, int64_t t, int index) 
{
    double dt = (t - k->old_t) / (double)1;
    double p00_temp = k->p[0][0];
    double p01_temp = k->p[0][1];
    double H[2][2] = {{!index, 0}, {index, 0}};
    double s[2][2];
    double p_new[2][2];

    /* innovation covariance S = H * P * HT + R */
    matrix_mult_2x2(H, k->p, s);
    matrix_transpose_2x2(H, H);
    matrix_mult_2x2(s, H, s);
    s[0][0] += k->r[0][0];
    s[0][1] += k->r[0][1];
    s[1][0] += k->r[1][0];
    s[1][1] += k->r[1][1];

    /* K = P * HT * S^-1 */
    matrix_invert_2x2(s, s);
    matrix_mult_2x2(k->p, H, k->k);
    matrix_mult_2x2(k->k, s, k->k);

    /* P = (I - K * H) * P */
    matrix_transpose_2x2(H, H);
    matrix_mult_2x2(k->k, H, p_new);
    p_new[0][0] = 1 - p_new[0][0];
    p_new[0][1] = 0 - p_new[0][1];
    p_new[1][0] = 0 - p_new[1][0];
    p_new[1][1] = 1 - p_new[1][1];
    matrix_mult_2x2(p_new, k->p, k->p);

    /* measurement residual y = z - H * x */
    double y[2]; 
    double x_new[2];
    matrix_mult_2x2_2x1(H, k->x, y);
    y[0] = z - y[0];
    y[1] = z - y[1];

    /* state update x = x + K * y */
    matrix_mult_2x2_2x1(k->k, y, x_new);
    k->x[0] = k->x[0] + x_new[0];
    k->x[1] = k->x[1] + x_new[1];

    k->old_t = t;

}

void kalman_predict2(struct kalman *k, int64_t t, double v)
{
    double dt = (t - k->old_t) / (double)1;
    double F[2][2] = {{1, 1},{0, 1}};

    F[0][1] = dt;

    /* P = F * P * FT + Q */
    //k->p = k->p + k->q * dt;
    matrix_mult_2x2(F, k->p, k->p);
    matrix_transpose_2x2(F, F);
    matrix_mult_2x2(k->p, F, k->p);
    k->p[0][0] += k->q[0][0] * dt;
    k->p[1][1] += k->q[1][1] * dt;

    /* x[k] = F * x[k - 1] */
    //k->x = k->x + k->y * dt;

    k->x[1] = v;
    k->x[0] = k->x[0] + k->x[1] * dt;
    
    k->old_t = t;
}

void kalman_predict(struct kalman *k, int64_t t)
{
    double dt = (t - k->old_t) / (double)1;
    double F[2][2] = {{1, 1},{0, 1}};

    F[0][1] = dt;

    /* P = F * P * FT + Q */
    //k->p = k->p + k->q * dt;
    matrix_mult_2x2(F, k->p, k->p);
    matrix_transpose_2x2(F, F);
    matrix_mult_2x2(k->p, F, k->p);
    k->p[0][0] += k->q[0][0] * dt;
    k->p[1][1] += k->q[1][1] * dt;

    /* x[k] = F * x[k - 1] */
    //k->x = k->x + k->y * dt;
    k->x[0] = k->x[0] + k->x[1] * dt;
    
    k->old_t = t;
}

void test()
{
    struct kalman k;
    int i;
    double vel = 0.02;
    double v = 5;
    double v2;

    kalman_init(&k, 16.7 * 16.7, 1e-3, 10e+2);

    for (i = 0; i < 10000; i++) {
        if (i == 5500) vel = 0.1;
        if (i == 6000) vel = 0.02;
        if (i == 8000) vel = -0.1;
        v += vel;
        kalman_predict2(&k, i, vel);
        if (i % 10 == 0) {
            v2 = v + 10 * (((rand() & 0xff) / 256.0) - 0.5);
            kalman_update(&k, v2, i, 1);
        }
        v2 = v + 100 * (((rand() & 0xff) / 256.0) - 0.5);
        kalman_update(&k, v2, i, 0);

        printf("%d, %f %f %f %f %f %f %f %f\n",
               i, k.k[0][0], k.k[1][1], k.p[0][0], k.x[0], v, v2, k.x[1], k.p[1][0]);
        
        //kalman_predict(&k, i);
    }
}

int main(int argc, char *argv[])
{
    test();
    return 0;
}

