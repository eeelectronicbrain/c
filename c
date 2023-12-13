
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 



float Icell(float G, float V, float R_BL) {
    if (G == 0) {
        return 0;
    }
    else {
        return V / (1.0 / G + R_BL);
    }
}

float sneak_cal_a(float* G_arr, float* V_arr, float Isb, float Vsl, float G_offset, float* G_var, float R_SL, float R_BL, int arr) {
    float V_n = Vsl;
    float I_sum = 0;
    float GV_sum = 0;
    float average_V_arr = 0;
    float average_G_arr = 0;

    for (int n = 0; n < arr; n++) {
        float G_target = G_arr[n];
        float G_write = G_target + G_var[n];
        float I_n;
        float V_in = V_arr[n];


        if (G_target > 0) {
            I_n = Icell(G_write + G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }
        else if (G_target < 0) {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(-G_write + G_offset, -V_in - V_n, R_BL);
        }
        else {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }

        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        float GV_n = G_target * V_in;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

        average_V_arr += V_in;
        average_G_arr += G_target;
    }

    // Calculate average values


    average_V_arr /= arr;
    average_G_arr /= arr;

    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return Isb;
}


float* sneak_cal_b(float* G_arr, float* V_arr, float Isb, float Vsl, float G_offset, float* G_var, float R_SL, float R_BL, int arr) {
    float V_n = Vsl;
    float I_sum = 0;
    float GV_sum = 0;
    float average_V_arr = 0;
    float average_G_arr = 0;

    for (int n = 0; n < arr; n++) {
        float G_target = G_arr[n];
        float G_write = G_target + G_var[n];
        float I_n;
        float V_in = V_arr[n];


        if (G_target > 0) {
            I_n = Icell(G_write + G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }
        else if (G_target < 0) {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(-G_write + G_offset, -V_in - V_n, R_BL);
        }
        else {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }

        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        float GV_n = G_target * V_in;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

        average_V_arr += V_in;
        average_G_arr += G_target;
    }

    // Calculate average values


    average_V_arr /= arr;
    average_G_arr /= arr;
    static float result[5];

    result[0] = I_sum;
    result[1] = Isb;
    result[2] = GV_sum;
    result[3] = average_V_arr;
    result[4] = average_G_arr;


    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return result;
}


float* sneak_cal_Rthev(float* G_arr, float Isb, float G_offset, float* G_var, float R_SL, float R_BL, int arr) {
    float V_n = 0;
    float I_sum = 0;
    float GV_sum = 0;

    for (int n = 0; n < arr; n++) {
        float G_target = G_arr[n];
        float G_write = G_target + G_var[n];
        float I_n;




        I_n = Icell(fabs(G_write) + G_offset, 1 - V_n, R_BL) + Icell(G_offset, 1 - V_n, R_BL);


        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        float GV_n = G_target * 1;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

    }

    // Calculate average values


    static float result[2];

    result[0] = I_sum;
    result[1] = Isb;


    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return result;
}


float R_thev(float* G_arr, float G_offset, float* G_var, float R_SL, float R_BL, int arr) {

    float I_a = sneak_cal_Rthev(G_arr, 0, G_offset, G_var, R_SL, R_BL, arr)[1];
    float I_b = sneak_cal_Rthev(G_arr, 10, G_offset, G_var, R_SL, R_BL, arr)[1];
    float c = 10 - I_b * (10) / (I_b - I_a);




    return c;
}

float* thev(float* G_arr, float* V_arr, float G_offset, float* G_var, float R_SL, float R_BL, float R_ext, int arr) {

    float I_sum = 0;
    float GV_sum = 0;


    float I_a = sneak_cal_a(G_arr, V_arr, 0, 0, G_offset, G_var, R_SL, R_BL, arr);
    float* I_b = sneak_cal_b(G_arr, V_arr, 10, 0, G_offset, G_var, R_SL, R_BL, arr);
    float Ithev = 10 - I_b[1] * (10) / (I_b[1] - I_a);

    float Rthev = R_thev(G_arr, G_offset, G_var, R_SL, R_BL, arr);
    float Vthev = Rthev * Ithev;
    float Iout = Vthev / (Rthev + R_ext);


    static float result[7];

    result[0] = Iout;
    result[1] = Vthev;
    result[2] = Rthev;
    result[3] = Ithev;
    result[4] = I_b[2];
    result[5] = I_b[3];
    result[6] = I_b[4];

    return result;
}


void process_array(float(*arr_test)[8][8], int size1) {
    // Your C code to process the array goes here
    // For example, print the values
    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                printf("%f ", arr_test[i][j][k]);
            }
            printf("\n");
        }
    }
}

float sum_of_array(float(*arr_test)[128][8][8]) {
    float sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}
/*
float sum_of_array_a(float***arr_test) {
    float sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}


float conv(float* G_arr, float* V_arr, float G_offset, float* G_var, float R_SL, float R_BL, float R_ext, int arr) {
    float sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}
*/
float* cal_array(int n_image, int n_out, int n_slide, int arr_size, float* x_np, float* w_np) {
    float* result_a = malloc(n_image * n_out * n_slide * sizeof(float));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
            for (int in_n = 0; in_n < n_slide; in_n++) {
                float* arr_in = malloc(arr_size * sizeof(float));

                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    arr_in[arr_n] = x_np[in_n + arr_n * n_slide + i_b * n_image];
                }

                float result = 0.0f;
                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    result += arr_in[arr_n] * w_np[arr_n + out_ch * n_out];
                }
                result_a[n] = result;
                n = n + 1;
                free(arr_in);
            }
        }
    }

    return result_a;
}


int main() {

    return 0;
}

