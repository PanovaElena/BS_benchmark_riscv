
#include <riscv-vector.h>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <sstream>
#include <string>
#include <iomanip>

vfloat32m4_t my_exp(vfloat32m4_t vx, size_t vl){
    
    float Log2 = (float)0x1.62e43p-1;
    float Log2h = (float)0xb.17200p-4;
    float Log2l = (float)0x1.7f7d1cf8p-20;
    float InvLog2 = (float)0x1.715476p0;
    
    // Here should be the tests for exceptional cases
    
    // rounding x/Log2
    
    //t.f = (x*InvLog2 + 12582912.0);
    vfloat32m4_t vt = vfmul_vf_f32m4(vx, InvLog2, vl);
    vt = vfadd_vf_f32m4(vt, (float)12582912.0, vl);
    
    //kf = (t.f - 12582912.0);
    vfloat32m4_t vkf = vfsub_vf_f32m4(vt, (float)12582912.0, vl);
    
    // k = t.i32 & 0x0000ffff;
    vint32m4_t vti = vreinterpret_v_f32m4_i32m4(vt);      // cast
    vint32m4_t vk = vand_vx_i32m4(vti, 0x0000ffff, vl);   // &
    
    // moving x to the small range
    //y = (x - kf*Log2h) - kf*Log2l;      
    vfloat32m4_t vy = vfmul_vf_f32m4(vkf, Log2h, vl);
    vy = vfsub_vv_f32m4(vx, vy, vl);
    vkf = vfmul_vf_f32m4(vkf, Log2l, vl);
    vy = vfsub_vv_f32m4(vy, vkf, vl);
    
    // computing in small range
    //r.f = (float)0x1p0 + y * ((float)0x1p0 + y * ((float)0x1.000000000000bp-1 + 
    //      y * ((float)0x1.5555555555511p-3 + y * ((float)0x1.55555555502a1p-5 + 
    //      y * ((float)0x1.1111111122322p-7 + y * ((float)0x1.6c16c1852b7afp-10 + 
    //      y * ((float)0x1.a01a014761f65p-13 + y * ((float)0x1.a01997c89eb71p-16 + 
    //      y * ((float)0x1.71dee62401315p-19 + y * ((float)0x1.28af3fca213eap-22 + 
    //      y * (float)0x1.ade1569ce2bdfp-26))))))))));
    vfloat32m4_t vr = vfmul_vf_f32m4(vy, (float)0x1.6850e4p-10, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.123bccp-7, vl);
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.555b98p-5, vl);    
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.55548ep-3, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.fffff8p-2, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1p0, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1p0, vl);    
    
    // Exponent update
    //r.i32 += k << 23;
    vint32m4_t vri = vreinterpret_v_f32m4_i32m4(vr); // cast
    vk = vsll_vx_i32m4(vk, 23, vl);                  // <<
    vri = vadd_vv_i32m4(vri, vk, vl);                // +
    vr = vreinterpret_v_i32m4_f32m4(vri);            // cast
    
    return vr;
}

vfloat32m4_t my_log(vfloat32m4_t x, int vl)
{
   
        //fast_log abs(rel) : avgError = 2.85911e-06(3.32628e-08), MSE = 4.67298e-06(5.31012e-08), maxError = 1.52588e-05(1.7611e-07)
    float s_log_C0 = -19.645704f;
    float s_log_C1 = 0.767002f;
    float s_log_C2 = 0.3717479f;
    float s_log_C3 = 5.2653985f;
    float s_log_2 = 0.6931472f;
    float s_log_C4 = -(1.0f + s_log_C0) * (1.0f + s_log_C1) / ((1.0f + s_log_C2) * (1.0f + s_log_C3)); //ensures that log(1) == 0

    
// int e = static_cast<int>(ux - 0x3f800000) >> 23;
    vint32m4_t intx = vreinterpret_v_f32m4_i32m4(x);
    vint32m4_t exp = vsub_vx_i32m4(intx, 0x3f800000, vl);
    exp  = vsra_vx_i32m4(exp, 23, vl);
   // ux |= 0x3f800000;
   // ux &= 0x3fffffff; // 1 <= x < 2  after replacing the exponent field
    intx = vor_vx_i32m4(intx, 0x3f800000, vl);
    intx = vand_vx_i32m4(intx, 0x3fffffff, vl);

// x = reinterpret_cast<float&>(ux);
    vfloat32m4_t y = vreinterpret_v_i32m4_f32m4(intx);
    
//float a = (x + s_log_C0) * (x + s_log_C1);
    vfloat32m4_t a = vfadd_vf_f32m4(y, s_log_C0, vl);
    vfloat32m4_t tmp = vfadd_vf_f32m4(y, s_log_C1, vl);
    a = vfmul_vv_f32m4(a, tmp, vl);
// float b = (x + s_log_C2) * (x + s_log_C3);

    vfloat32m4_t b = vfadd_vf_f32m4(y, s_log_C2, vl);
    vfloat32m4_t tmp2 = vfadd_vf_f32m4(y, s_log_C3, vl);
    b = vfmul_vv_f32m4(b, tmp2, vl);
//float c = (float(e) + s_log_C4);

    vfloat32m4_t e = vfcvt_f_x_v_f32m4(exp, vl);
    vfloat32m4_t c = vfadd_vf_f32m4(e, s_log_C4, vl);

// float d = a / b;
    vfloat32m4_t d = vfdiv_vv_f32m4(a, b, vl);


//(c + d)* s_log_2;
    vfloat32m4_t c_plus_d = vfadd_vv_f32m4(c, d, vl);
    vfloat32m4_t result = vfmul_vf_f32m4(c_plus_d, s_log_2, vl);
    return result;
}

vfloat32m4_t my_erf(vfloat32m4_t x, int vl)
{
    float a1 = (float)0.254829592;
    float a2 = (float)-0.284496736;
    float a3 = (float)1.421413741;
    float a4 = (float)-1.453152027;
    float a5 = (float)1.061405429;
    float p = (float)0.3275911;
    vint32m4_t sign;
    vint32m4_t floatcast = vreinterpret_v_f32m4_i32m4(x);
    sign = vsra_vx_i32m4(floatcast, 31, vl);
    sign = vor_vx_i32m4(sign, 1, vl);
    vfloat32m4_t float_sign =  vfcvt_f_x_v_f32m4(sign,  vl);

    vfloat32m4_t x_abs= vfmul_vv_f32m4(float_sign, x, vl);

    vfloat32m4_t oper1 = vfmul_vf_f32m4(x_abs, p, vl);
    oper1 = vfadd_vf_f32m4(oper1, 1.0f, vl);
    vfloat32m4_t vect1 = vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t t = vfdiv_vv_f32m4(vect1, oper1, vl);

    vfloat32m4_t oper2 = vfmul_vf_f32m4(t, a5, vl);
    oper2 = vfadd_vf_f32m4(oper2, a4, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a3, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a2, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a1, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);

    vfloat32m4_t minus_x = vfmul_vf_f32m4(x, -1.0f, vl);
    vfloat32m4_t minus_x_mult_x = vfmul_vv_f32m4(x, minus_x, vl);
    vfloat32m4_t exp_x_x = my_exp(minus_x_mult_x, vl);

    oper2 = vfmul_vv_f32m4(oper2, exp_x_x, vl);
    vfloat32m4_t y = vfsub_vv_f32m4(vect1, oper2, vl);
    
    
    vfloat32m4_t y_mult_sign = vfmul_vv_f32m4(y, float_sign, vl);
    return y_mult_sign;

}

void print(char* x) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            std::cout << ((((uint8_t)x[i])>>j)&1);
    std::cout << std::endl;
}

int main() {
    const int N = 256;
    char x[N*4];//-0.15f;
    for (int i = 0; i < N; i++)
        *((float*)(&x[4*i])) = 3.0f;
    print((char*)(&x[0]));
    //__m512 vx = _mm512_load_ps(x);
    size_t vl = vsetvl_e32m4(N);
    vfloat32m4_t vx = vle_v_f32m4((float*)x, vl);
    vfloat32m4_t vres = my_erf(vx, vl);//my_exp(vx, vl);
    float res[N];
    vse_v_f32m4(res, vres, vl);
    print((char*)(&res[0])); 
    std::cout << res[0] << std::endl;
}