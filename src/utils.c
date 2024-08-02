#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<stdarg.h>
#include<stdbool.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "utils.h"
#include "cJSON.h"


void print_matrix(Matrix *arr, bool shape_only){
    int shape[7] = {0};

    if(arr == NULL){
        printf("matrix: NULL\n");
        return;
    }

    printf("shape: ");
    for(int i=0;i<arr->dim;i++){
        printf("%d, ", arr->shape[i]);
    }
    printf("\n\n");
    if(!shape_only){
        if(arr->matrix == NULL)
            printf("empty matrix\n");
        else
            print_recursive_matrix(arr, 0, shape);
        printf("\n\n");
    }
}

void print_recursive_matrix(Matrix* arr, int cur_dim, int shape[7]){
    printf("[ ");
    if(cur_dim == (arr->dim)-1){
        for(int i=0;i<arr->shape[cur_dim]; i++){
            printf("%lf  ", get_value_by_array(arr, shape));
            shape[cur_dim] +=1;
        }
        shape[cur_dim] = 0;
        if(shape[cur_dim-1] == (arr->shape[cur_dim-1]))
            printf(" ]");
        else
            printf(" ]\n");
    }
    else{
        for(int i=0;i<arr->shape[cur_dim]; i++){
            print_recursive_matrix(arr, cur_dim+1, shape);
            shape[cur_dim] +=1;
        }
        shape[cur_dim] = 0;
        printf(" ]");
    }
    
}

double sign(double num) {
    if (num > 0.0) {
        return 1.0;
    } else if (num < 0.0) {
        return -1.0;
    } else {
        return 0.0;
    }
}

int get_index(Matrix* arr, ...) {
    int index = 0;

    va_list args;

    va_start(args, arr->dim);

    for (int i = 0; i < (arr->dim) - 1; i++) {
        index += va_arg(args, int);
        index *= arr->shape[i + 1];
    }
    index += va_arg(args, int);

    va_end(args);

    return index;
}

int get_index_by_array(Matrix* arr, int* multiIndex) {
    int index = 0;

    for (int i = 0; i < (arr->dim) - 1; i++) {
        index += multiIndex[i];
        index *= arr->shape[i + 1];
    }
    index += multiIndex[(arr->dim)-1];

    return index;
}

double get_value(Matrix* arr, ...) {
    int index = 0;

    va_list args;

    va_start(args, arr->dim);

    for (int i = 0; i < (arr->dim) - 1; i++) {
        index += va_arg(args, int);
        index *= arr->shape[i + 1];
    }
    index += va_arg(args, int);

    va_end(args);

    return arr->matrix[index];
}

double get_value_by_array(Matrix* arr, int* multiIndex) {
    int index = 0;

    for (int i = 0; i < (arr->dim) - 1; i++) {
        index += multiIndex[i];
        index *= arr->shape[i + 1];
    }
    index += multiIndex[(arr->dim)-1];

    return arr->matrix[index];
}

void make_connection_matrix(Matrix *arr){
    if(!delete_connection_matrix(arr)){
        return;
    }
    if(arr->dim == 1){
        arr->matrix1d = arr->matrix;
        return;
    }

    int i, idx = 0;
    int single_size;

    
    if(arr->dim == 2){
        arr->matrix2d = malloc(sizeof(double**)*(arr->shape[0]));
        single_size = arr->shape[1];
        for(i=0 ;i<arr->shape[0];i++, idx+= single_size){
            arr->matrix2d[i] = &(arr->matrix[idx]);
        }
    }
    else if(arr->dim == 3){
        arr->matrix3d = malloc(sizeof(double***)*(arr->shape[0]));
        single_size = arr->shape[2];
        for(int d2=0;d2<arr->shape[0];d2++){
            arr->matrix3d[d2]  = malloc(sizeof(double**)*(arr->shape[1]));
            for(i=0 ;i<arr->shape[1];i++, idx+= single_size){
                arr->matrix3d[d2][i] = &(arr->matrix[idx]);
            }
        }
    }
    else if(arr->dim == 4){
        arr->matrix4d = malloc(sizeof(double****)*(arr->shape[0]));
        single_size = arr->shape[3];
        for(int d2=0;d2<arr->shape[0];d2++){
            arr->matrix4d[d2]  = malloc(sizeof(double***)*(arr->shape[1]));
            for(int d3 = 0; d3<arr->shape[1];d3++){
                arr->matrix4d[d2][d3] = malloc(sizeof(double**)*(arr->shape[2]));
                for(i=0 ;i<arr->shape[2];i++, idx+= single_size){
                    arr->matrix4d[d2][d3][i] = &(arr->matrix[idx]);
                }
            }
        }
    }
    else if(arr->dim == 5){
        arr->matrix5d = malloc(sizeof(double*****)*(arr->shape[0]));
        single_size = arr->shape[4];
        for(int d2=0;d2<arr->shape[0];d2++){
            arr->matrix5d[d2]  = malloc(sizeof(double****)*(arr->shape[1]));
            for(int d3 = 0; d3<arr->shape[1];d3++){
                arr->matrix5d[d2][d3] = malloc(sizeof(double***)*(arr->shape[2]));
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    arr->matrix5d[d2][d3][d4] = malloc(sizeof(double**)*(arr->shape[3]));
                    for(i=0 ;i<arr->shape[3];i++, idx+= single_size){
                        arr->matrix5d[d2][d3][d4][i] = &(arr->matrix[idx]);
                    }
                }
            }
        }
    }
    else if(arr->dim == 6){
        arr->matrix6d = malloc(sizeof(double******)*(arr->shape[0]));
        single_size = arr->shape[5];
        for(int d2=0;d2<arr->shape[0];d2++){
            arr->matrix6d[d2]  = malloc(sizeof(double*****)*(arr->shape[1]));
            for(int d3 = 0; d3<arr->shape[1];d3++){
                arr->matrix6d[d2][d3] = malloc(sizeof(double****)*(arr->shape[2]));
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    arr->matrix6d[d2][d3][d4] = malloc(sizeof(double***)*(arr->shape[3]));
                    for(int d5 = 0; d5<arr->shape[3];d5++){
                        arr->matrix6d[d2][d3][d4][d5] = malloc(sizeof(double**)*(arr->shape[4]));
                        for(i=0 ;i<arr->shape[4];i++, idx+= single_size){
                            arr->matrix6d[d2][d3][d4][d5][i] = &(arr->matrix[idx]);
                        }
                    }
                }
            }
        }
    }
    else if(arr->dim == 7){
        arr->matrix7d = malloc(sizeof(double*******)*(arr->shape[0]));
        single_size = arr->shape[6];
        for(int d2=0;d2<arr->shape[0];d2++){
            arr->matrix7d[d2]  = malloc(sizeof(double******)*(arr->shape[1]));
            for(int d3 = 0; d3<arr->shape[1];d3++){
                arr->matrix7d[d2][d3] = malloc(sizeof(double*****)*(arr->shape[2]));
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    arr->matrix7d[d2][d3][d4] = malloc(sizeof(double****)*(arr->shape[3]));
                    for(int d5 = 0; d5<arr->shape[3];d5++){
                        arr->matrix7d[d2][d3][d4][d5] = malloc(sizeof(double***)*(arr->shape[4]));
                        for(int d6 = 0; d6<arr->shape[4];d6++){
                            arr->matrix7d[d2][d3][d4][d5][d6] = malloc(sizeof(double**)*(arr->shape[5]));
                            for(i=0 ;i<arr->shape[5];i++, idx+= single_size){
                                arr->matrix7d[d2][d3][d4][d5][d6][i] = &(arr->matrix[idx]);
                            }
                        }
                    }
                }
            }
        }
    }
    
}

bool delete_connection_matrix(Matrix *arr){
    if(arr == NULL||arr->matrix==NULL)
        return false;
    
    if(arr->dim == 1){
        if(arr->matrix1d==NULL)
            return true;
        arr->matrix1d = NULL;
        return true;
    }
    else if(arr->dim == 2){
        if(arr->matrix2d==NULL)
            return true;
        free(arr->matrix2d);
        arr->matrix2d = NULL;
    }
    else if(arr->dim == 3){
        if(arr->matrix3d==NULL)
            return true;
        for(int d2=0;d2<arr->shape[0];d2++){
            free(arr->matrix3d[d2]);
        }
        free(arr->matrix3d);
        arr->matrix3d = NULL;
    }
    else if(arr->dim == 4){
        if(arr->matrix4d==NULL)
            return true;
        for(int d2=0;d2<arr->shape[0];d2++){
            for(int d3 = 0; d3<arr->shape[1];d3++){
                free(arr->matrix4d[d2][d3]);
            }
            free(arr->matrix4d[d2]);
        }
        free(arr->matrix4d);
        arr->matrix4d = NULL;
    }
    else if(arr->dim == 5){
        if(arr->matrix5d==NULL)
            return true;
        for(int d2=0;d2<arr->shape[0];d2++){
            for(int d3 = 0; d3<arr->shape[1];d3++){
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    free(arr->matrix5d[d2][d3][d4]);
                }
                free(arr->matrix5d[d2][d3]);
            }
            free(arr->matrix5d[d2]);
        }
        free(arr->matrix5d);
        arr->matrix5d = NULL;
    }
    else if(arr->dim == 6){
        if(arr->matrix6d==NULL)
            return true;
        for(int d2=0;d2<arr->shape[0];d2++){
            for(int d3 = 0; d3<arr->shape[1];d3++){
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    for(int d5 = 0; d5<arr->shape[3];d5++){
                        free(arr->matrix6d[d2][d3][d4][d5]);
                    }
                    free(arr->matrix6d[d2][d3][d4]);
                }
                free(arr->matrix6d[d2][d3]);
            }
            free(arr->matrix6d[d2]);
        }
        free(arr->matrix6d);
        arr->matrix6d = NULL;
    }
    else if(arr->dim == 7){
        if(arr->matrix7d==NULL)
            return true;
        for(int d2=0;d2<arr->shape[0];d2++){
            for(int d3 = 0; d3<arr->shape[1];d3++){
                for(int d4 = 0; d4<arr->shape[2];d4++){
                    for(int d5 = 0; d5<arr->shape[3];d5++){
                        for(int d6 = 0; d6<arr->shape[4];d6++){
                            free(arr->matrix7d[d2][d3][d4][d5][d6]);
                        }
                         free(arr->matrix7d[d2][d3][d4][d5]);
                    }
                    free(arr->matrix7d[d2][d3][d4]);
                }
                free(arr->matrix7d[d2][d3]);
            }
            free(arr->matrix7d[d2]);
        }
        free(arr->matrix7d);
        arr->matrix7d = NULL;
    }

    return true;
}

double randn() {
    double u1, u2;
    double r, theta;

    u1 = ((double)rand() / RAND_MAX); 
    u2 = ((double)rand() / RAND_MAX); 

    r = sqrt(-2.0 * log(u1));
    theta = 2.0 * PI * u2;

    return r * cos(theta);
}

void copy_matrix(Matrix* arr, Matrix** _dest, bool form_only){
    Matrix* dest = NULL;
    
    delete_matrix(*_dest);
    dest = (Matrix*)malloc(sizeof(Matrix));
    if(dest == NULL){
        printf("malloc failed\n");
        exit(6);
    }
    *_dest = dest;
    
    if(form_only){
        dest->matrix = (double *)calloc(arr->size,sizeof(double));
    }
    else{
        dest->matrix = (double *)malloc(sizeof(double)*(arr->size));
        memcpy(dest->matrix, arr->matrix, sizeof(double)*(arr->size));
    }
    memcpy(dest->shape, arr->shape, sizeof(int)*MAX_MATRIX_DIM);
    dest->size = arr->size;
    dest->dim = arr->dim;
    dest->connection_ptr = NULL;
    make_connection_matrix(dest);
}

void delete_matrix(Matrix* arr){
    // printf("seg_error1\n");
    if(arr == NULL){
        // printf("seg_OK\n");
        return;
    }
    delete_connection_matrix(arr);
    if(arr->matrix !=NULL){
        // printf("matrix deleted\n");
        free(arr->matrix);
        arr->matrix = NULL;
    }
    // printf("seg_error2\n");
    free(arr);
    // printf("seg_error3\n");
}

void init_by_shape_arr_matrix(Matrix** _arr, int* shape, int dim, int initType, double inputNodeNum_or_customValue){
    Matrix *arr = NULL;
    double stdev;

    delete_matrix(*_arr);
    
    arr = (Matrix*)malloc(sizeof(Matrix));
    if(arr == NULL){
        printf("malloc failed\n");
        exit(6);
    }
    *_arr = arr;
    
    arr->size = 1;
    for(int i=0;i<dim;i++){
        arr->shape[i] = shape[i];
        arr->size *= shape[i];
    }
    arr->dim = dim;
    arr->connection_ptr = NULL;
    switch(initType){
        case NONE:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            make_connection_matrix(arr);
            break;
        case ZEROS:
            arr->matrix = (double*)calloc(arr->size,sizeof(double));
            make_connection_matrix(arr);
            break;
        case CUSTOM:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = inputNodeNum_or_customValue;
            }
            make_connection_matrix(arr);
            break;
        case HE:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            stdev = sqrt(2.0 / inputNodeNum_or_customValue);
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = randn()*stdev;
            }
            make_connection_matrix(arr);
            break;
        case XAVIER:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            stdev = sqrt(1.0 / inputNodeNum_or_customValue);
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = randn()*stdev;
            }
            make_connection_matrix(arr);
            break;
        case SHAPE_ONLY:
            arr->matrix = NULL;
            break;
    }
}

void init_matrix(Matrix** _arr, int dim, int initType, double inputNodeNum_or_customValue, ...){
    Matrix *arr = NULL;
    double stdev;
    
    delete_matrix(*_arr);
    // printf("malloc error\n");
    arr = (Matrix*)malloc(sizeof(Matrix));
    if(arr == NULL){
        printf("malloc failed\n");
        exit(6);
    }
    *_arr = arr;

    va_list args;
    va_start(args, dim);
    
    arr->size = 1;
    for(int i=0;i<dim;i++){
        arr->shape[i] = va_arg(args, int);
        arr->size *= arr->shape[i];
    }
    va_end(args);
    arr->dim = dim;
    arr->connection_ptr = NULL;
    switch(initType){
        case NONE:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            make_connection_matrix(arr);
            break;
        case ZEROS:
            arr->matrix = (double*)calloc(arr->size,sizeof(double));
            make_connection_matrix(arr);
            break;
        case CUSTOM:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = inputNodeNum_or_customValue;
            }
            make_connection_matrix(arr);
            break;
        case HE:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            stdev = sqrt(2.0 / inputNodeNum_or_customValue);
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = randn()*stdev;
            }
            make_connection_matrix(arr);
            break;
        case XAVIER:
            arr->matrix = (double*)malloc(sizeof(double)*(arr->size));
            stdev = sqrt(1.0 / inputNodeNum_or_customValue);
            for(int i=0; i < arr->size; i++){
                arr->matrix[i] = randn()*stdev;
            }
            make_connection_matrix(arr);
            break;
        case SHAPE_ONLY:
            arr->matrix = NULL;
            break;
    }
}

void reshape_matrix(Matrix* arr, int dim, ...){
    int check = arr->size;
    int orgin_size = arr->size;
    int shapeTo;

    // printf("seg_error_reshape_disconnection\n");
    delete_connection_matrix(arr);
    va_list args;

    va_start(args, dim);
    
    for(int i=0; i<dim; i++){
        shapeTo = va_arg(args, int);
        if(shapeTo == 0){
            printf("reshape_matrix: cannot divied by 0\n");
            exit(3);
        }
        else if(check % shapeTo != 0){
            printf("reshape error\n");
            exit(1);
        }
        
        
        check /= shapeTo;
        arr->shape[i] = shapeTo;
    }
    va_end(args);
    arr->dim = dim;

    if(check != 1){
        printf("wrong reshape order:\n");
        exit(3);
    }
    // printf("seg_error_reshape\n");
    make_connection_matrix(arr);
    // printf("seg_OK\n");
}

void reshape_by_arr_matrix(Matrix* arr, int dim, int*shape){
    int check = arr->size;
    int shapeTo;
    
    delete_connection_matrix(arr);
    for(int i=0; i<dim; i++){
        shapeTo = shape[i];
        if(shapeTo == 0){
            printf("reshape_matrix: cannot divied by 0\n");
            exit(3);
        }
        else if(check % shapeTo != 0){
            printf("reshape error\n");
            exit(1);
        }
        check /= shapeTo;
        arr->shape[i] = shapeTo;
    }
    arr->dim = dim;

    if(check != 1){
        printf("wrong reshape order:\n");
        exit(3);
    }
    make_connection_matrix(arr);
}

void transpose_matrix(Matrix *arr, Matrix ** _dest, ...){ //perform better than recursive one
    int curIndex, destIndex;
    va_list args;
    int transposeTo[arr->dim];
    int temp[arr->dim];
    int transposeShape[7];
    Matrix * dest = NULL;
    int t1,t2,t3,t4,t5,t6;
    // print_matrix(arr, true);

    va_start(args, arr->dim);
    for(int i=0;i<arr->dim;i++){
        transposeTo[i] = va_arg(args, int);
        transposeShape[i] = arr->shape[transposeTo[i]];
    }
    va_end(args);

    init_by_shape_arr_matrix(_dest,transposeShape, arr->dim, NONE, 0);
    dest = *_dest;

    switch(dest->dim){
        case 2:
            t1 = transposeTo[0];
            t2 = transposeTo[1];
            for(int i1 = 0; i1<(arr->shape[0]);i1++){
                temp[0] = i1;
                for(int i2 = 0; i2<(arr->shape[1]);i2++){
                    temp[1] = i2;
                    dest->matrix2d[temp[t1]][temp[t2]] = arr->matrix2d[i1][i2];
                }
            }
            //arr->shape[transposeTo[i]]
            break;
        case 3:
            t1 = transposeTo[0];
            t2 = transposeTo[1];
            t3 = transposeTo[2];
            for(int i1 = 0; i1<(arr->shape[0]);i1++){
                temp[0] = i1;
                for(int i2 = 0; i2<(arr->shape[1]);i2++){
                    temp[1] = i2;
                    for(int i3 = 0; i3<(arr->shape[2]);i3++){
                        temp[2] = i3;
                        dest->matrix3d[temp[t1]][temp[t2]][temp[t3]] = arr->matrix3d[i1][i2][i3];
                    }
                }
            }
            break;
        case 4:
            t1 = transposeTo[0];
            t2 = transposeTo[1];
            t3 = transposeTo[2];
            t4 = transposeTo[3];
            for(int i1 = 0; i1<(arr->shape[0]);i1++){
                temp[0] = i1;
                for(int i2 = 0; i2<(arr->shape[1]);i2++){
                    temp[1] = i2;
                    for(int i3 = 0; i3<(arr->shape[2]);i3++){
                        temp[2] = i3;
                        for(int i4 = 0; i4<(arr->shape[3]);i4++){
                            temp[3] = i4;
                            dest->matrix4d[temp[t1]][temp[t2]][temp[t3]][temp[t4]] = arr->matrix4d[i1][i2][i3][i4];
                        }
                    }
                }
            }
            break;
        case 5:
            t1 = transposeTo[0];
            t2 = transposeTo[1];
            t3 = transposeTo[2];
            t4 = transposeTo[3];
            t5 = transposeTo[4];
            for(int i1 = 0; i1<(arr->shape[0]);i1++){
                temp[0] = i1;
                for(int i2 = 0; i2<(arr->shape[1]);i2++){
                    temp[1] = i2;
                    for(int i3 = 0; i3<(arr->shape[2]);i3++){
                        temp[2] = i3;
                        for(int i4 = 0; i4<(arr->shape[3]);i4++){
                            temp[3] = i4;
                            for(int i5 = 0; i5<(arr->shape[4]);i5++){
                                temp[4] = i5;
                                dest->matrix5d[temp[t1]][temp[t2]][temp[t3]][temp[t4]][temp[t5]] = arr->matrix5d[i1][i2][i3][i4][i5];
                            }
                        }
                    }
                }
            }
            break;
        case 6:
            t1 = transposeTo[0];
            t2 = transposeTo[1];
            t3 = transposeTo[2];
            t4 = transposeTo[3];
            t5 = transposeTo[4];
            t6 = transposeTo[5];
            for(int i1 = 0; i1<(arr->shape[0]);i1++){
                temp[0] = i1;
                for(int i2 = 0; i2<(arr->shape[1]);i2++){
                    temp[1] = i2;
                    for(int i3 = 0; i3<(arr->shape[2]);i3++){
                        temp[2] = i3;
                        for(int i4 = 0; i4<(arr->shape[3]);i4++){
                            temp[3] = i4;
                            for(int i5 = 0; i5<(arr->shape[4]);i5++){
                                temp[4] = i5;
                                for(int i6 = 0; i6<(arr->shape[5]);i6++){
                                    temp[5] = i6;
                                    dest->matrix6d[temp[t1]][temp[t2]][temp[t3]][temp[t4]][temp[t5]][temp[t6]] = arr->matrix6d[i1][i2][i3][i4][i5][i6];
                                }
                            }
                        }
                    }
                }
            }
            break;
        //case 7:
    }
    
}

void dot2d_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha){
    Matrix * dest = NULL;

    if((A->dim) !=2 && (B->dim) != 2 && (A->shape[1])!=(B->shape[0])){
        printf("dot2d error: dimension matching error\n");
        exit(1);
    }
    int m = A->shape[0];
    int n = A->shape[1];
    int p = B->shape[1];
    int shape[2];

    shape[0] = m;
    shape[1] = p;
    init_by_shape_arr_matrix(_dest,shape, 2, NONE, 0);
    dest = *_dest;

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < p; j++) {
    //         double sum = 0.0;
    //         for (int k = 0; k < n; k++) {
    //             sum += (A->matrix2d[i][k]) * (B->matrix2d[k][j]);
    //         }
    //         dest->matrix2d[i][j] = sum;
    //     }
    // }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,p,n, alpha, A->matrix, n, B->matrix, p, 0.0,dest->matrix, p);
}

void dot2d_TN_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha){//nmp=MKN
    Matrix * dest = NULL;

    if((A->dim) !=2 && (B->dim) != 2 && (A->shape[0])!=(B->shape[0])){
        printf("dot2d error: dimension matching error\n");
        exit(1);
    }
    int m = A->shape[1];
    int n = A->shape[0];
    int p = B->shape[1];
    int shape[2];

    shape[0] = m;
    shape[1] = p;
    init_by_shape_arr_matrix(_dest,shape, 2, NONE, 0);
    dest = *_dest;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,p,n, alpha, A->matrix, m, B->matrix, p, 0.0,dest->matrix, p);
}

void dot2d_NT_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha){//mnp=MKK
    Matrix * dest = NULL;

    if((A->dim) !=2 && (B->dim) != 2 && (A->shape[1])!=(B->shape[1])){
        printf("dot2d error: dimension matching error\n");
        exit(1);
    }
    int m = A->shape[0];
    int n = A->shape[1];
    int p = B->shape[0];
    int shape[2];

    shape[0] = m;
    shape[1] = p;
    init_by_shape_arr_matrix(_dest,shape, 2, NONE, 0);
    dest = *_dest;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m,p,n, alpha, A->matrix, n, B->matrix, n, 0.0,dest->matrix, p);
}

void dot2d_TT_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha){
    Matrix * dest = NULL;

    if((A->dim) !=2 && (B->dim) != 2 && (A->shape[0])!=(B->shape[1])){
        printf("dot2d error: dimension matching error\n");
        exit(1);
    }
    int m = A->shape[1];
    int n = A->shape[0];
    int p = B->shape[0];
    int shape[2];

    shape[0] = m;
    shape[1] = p;
    init_by_shape_arr_matrix(_dest,shape, 2, NONE, 0);
    dest = *_dest;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, m,p,n, alpha, A->matrix, m, B->matrix, n, 0.0,dest->matrix, p);
}

void add_two_matrix(Matrix*A,Matrix*B,Matrix**_dest){
    Matrix * dest = NULL;

    if((A->dim) != (B->dim)){
        printf("add_two_matrix_error: dimension matching error\n");
        exit(1);
    }

    for(int i=0;i<(A->dim);i++){
        if((A->shape[i])!= (B->shape[i])){
            printf("add_two_matrix_error: shape matching error\n");
            exit(2);
        }
    }

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) + (B->matrix[i]);
    }    
}

void sub_two_matrix(Matrix*A,Matrix*B,Matrix**_dest){
    Matrix * dest = NULL;

    if((A->dim) != (B->dim)){
        printf("sub_two_matrix_error: dimension matching error\n");
        exit(1);
    }

    for(int i=0;i<(A->dim);i++){
        if((A->shape[i])!= (B->shape[i])){
            printf("sub_two_matrix_error: shape matching error\n");
            exit(2);
        }
    }

    // printf("seg_sub\n");
    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    // printf("seg_sub\n");
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) - (B->matrix[i]);
    }    
}

void mul_two_matrix(Matrix*A,Matrix*B,Matrix**_dest){
    Matrix * dest = NULL;

    if((A->dim) != (B->dim)){
        printf("mul_two_matrix_error: dimension matching error\n");
        exit(1);
    }

    for(int i=0;i<(A->dim);i++){
        if((A->shape[i])!= (B->shape[i])){
            print_matrix(A, true);
            print_matrix(B, true);
            printf("mul_two_matrix_error: shape matching error\n");
            exit(2);
        }
    }

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) * (B->matrix[i]);
    }    
}

void div_two_matrix(Matrix*A,Matrix*B,Matrix**_dest){
    Matrix * dest = NULL;

    if((A->dim) != (B->dim)){
        printf("div_two_matrix_error: dimension matching error\n");
        exit(1);
    }

    for(int i=0;i<(A->dim);i++){
        if((A->shape[i])!= (B->shape[i])){
            printf("div_two_matrix_error: shape matching error\n");
            exit(2);
        }
    }

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) / (B->matrix[i]);
    }    
}

void add_variable_matrix(Matrix*A,double b,Matrix**_dest){
    Matrix * dest = NULL;

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) + b;
    }
}

void mul_variable_matrix(Matrix*A,double b,Matrix**_dest){
    Matrix * dest = NULL;

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) * b;
    }
}

void div_variable_matrix(Matrix*A,double b,Matrix**_dest){
    Matrix * dest = NULL;

    if(b == 0){
        printf("div_variable_matrix: variable b cannot be 0\n");
        exit(3);
    }

    init_by_shape_arr_matrix(_dest,A->shape, A->dim, NONE, 0);
    dest = *_dest;

    for(int i=0;i<(dest->size);i++){
        dest->matrix[i] = (A->matrix[i]) / b;
    }
}

void im2col_matrix(Matrix *input, Matrix **_col, int filter_h, int filter_w, int stride_h,int stride_w,int pad_up,int pad_down,int pad_left, int pad_right){
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int out_h = (H+pad_up+pad_down-filter_h)/stride_h + 1;
    int out_w = (W+pad_left+pad_right-filter_w)/stride_w + 1;
    Matrix* col = NULL;
    Matrix* img = NULL;
    int x,y;
    
    init_matrix(&img, 4,ZEROS,0, N,C, H+pad_up+pad_down, W+pad_left+pad_right);

    
    //padding
    for(int n = 0; n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H;h++){
                for(int w=0;w<W;w++){
                    img->matrix4d[n][c][h+pad_up][w+pad_left] = input->matrix4d[n][c][h][w];
                }
            }
        }
    }
    
    //col initializing
    init_matrix(_col, 6, ZEROS, 0, N,out_h,out_w,C, filter_h, filter_w);
    col = *_col;
    
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int c = 0; c < C; c++) {
                    for (int fh = 0; fh < filter_h; fh++) {
                        for (int fw = 0; fw < filter_w; fw++) {
                            y = oh*stride_h+fh;
                            x = ow*stride_w+fw;
                            col->matrix6d[n][oh][ow][c][fh][fw] = img->matrix4d[n][c][y][x];
                        }
                    }
                }
            }
        }
    }
    delete_matrix(img);
    // print_matrix(col, true);
    // printf("%d,%d,%d,%d,%d,%d,%d\n", col->size, N, out_h, out_w, C, filter_h, filter_w);
    reshape_matrix(col, 2, N*out_h*out_w,C*filter_h*filter_w);
}

void col2im_matrix(Matrix *input, Matrix *_col, Matrix **_img, int filter_h, int filter_w, int stride_h,int stride_w,int pad_up,int pad_down,int pad_left, int pad_right){
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int out_h = (H+pad_up+pad_down-filter_h)/stride_h + 1;
    int out_w = (W+pad_left+pad_right-filter_w)/stride_w + 1;
    Matrix* img = NULL, *col = NULL;
    int img_h_start, img_w_start;

    copy_matrix(_col, &col, false);
    reshape_matrix(col, 6, N, out_h, out_w, C, filter_h, filter_w);
    init_matrix(&img, 4,ZEROS, 0, N, C, H+pad_up+pad_down+stride_h-1, W+pad_left+pad_right+stride_w-1);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < out_h; h++) {
                img_h_start = h*stride_h;
                for (int w = 0; w < out_w; w++) {
                    img_w_start = w*stride_w;
                    for (int fh = 0; fh < filter_h; fh++) {
                        for (int fw = 0; fw < filter_w; fw++) {
                            img->matrix4d[n][c][img_h_start+fh][img_w_start+fw] += col->matrix6d[n][h][w][c][fh][fw];
                        }
                    }
                }
            }
        }
    }
    delete_matrix(col);

    //img without padding
    init_matrix(_img, 4, NONE, 0, N,C,H,W);

    //delete padding
    for(int n = 0; n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H;h++){
                for(int w=0;w<W;w++){
                    (*_img)->matrix4d[n][c][h][w] = img->matrix4d[n][c][h+pad_up][w+pad_down];
                }
            }
        }
    }
    delete_matrix(img);
}

bool is_shape_matching_matrix(Matrix *A, Matrix *B){
    if((A->dim) != (B->dim))
        return false;
    for(int i=0;i<A->dim;i++){
        if((A->shape[i]) != (B->shape[i]))
            return false;
    }
    return true;
}

double sigmoid_(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax_(Matrix *x, Matrix **_y){
    Matrix * y = NULL, *_x = NULL;
    int batchsize, nodes;
    double max_value, temp, sum;

    copy_matrix(x,&_x,false);

    if((x->dim) == 2){
        batchsize = x->shape[0];
        nodes = x->shape[1];
        init_matrix(_y,2, NONE, 0, batchsize, nodes);
        y = *_y;
        
        for(int b=0;b<batchsize;b++){
            max_value = _x->matrix2d[b][0];
            sum = 0.0;
            for(int n=0;n<nodes;n++){
                temp = _x->matrix2d[b][n];
                if(max_value<temp)
                    max_value = temp;
            }
            for(int n=0;n<nodes;n++){
                _x->matrix2d[b][n] -= max_value;
                sum += exp(_x->matrix2d[b][n]);
            }
            for(int n=0;n<nodes;n++){
                y->matrix2d[b][n] = exp(_x->matrix2d[b][n])/sum;
            }
        }
        // print_matrix(y,false);
        // print_matrix(*_y,false);
    }
    else{
        
        nodes = x->shape[0];
        init_matrix(_y,1, NONE, 0, nodes);
        y = *_y;

        max_value = _x->matrix[0];
        sum = 0;
        for(int n=0;n<nodes;n++){
            temp = _x->matrix[n];
            if(max_value<temp)
                max_value = temp;
        }
        for(int n=0;n<nodes;n++){
            _x->matrix[n] -= max_value;
            sum += exp(_x->matrix[n]);
        }
        for(int n=0;n<nodes;n++){
            y->matrix[n]= exp(_x->matrix[n])/sum;
        }
    }
    
    delete_matrix(_x);
}

void cross_entropy_error(Matrix *y, Matrix *t, double*loss){
    Matrix *_y = NULL,*_t = NULL;
    int batchsize = y->shape[0];
    double temp;
    // print_matrix(y,true);
    // print_matrix(t, true);
    // printf("%lf\n\n",*loss );
    copy_matrix(y,&_y,false);
    copy_matrix(t,&_t,false);
    
    
    if(_y->dim == 1){
        reshape_matrix(_t, 2,1,_t->size);
        reshape_matrix(_y, 2,1,_y->size);
        batchsize = 1;
    }

    //if 't' is one-hot vector
    if(_t->size == _y->size){
        Matrix *t_temp = NULL;
        // init_matrix(&t_temp, 1, NONE, 0, batchsize);
        // for(int n=0;n<batchsize;n++){
        //     int max_index = 0;
        //     double max_value = _t->matrix[0];
        //     for(int idx=0;idx<(_t->shape[1]);idx++){
        //         temp = _t->matrix[idx];
        //         if(temp>max_value){
        //             max_index = idx;
        //             max_value = temp;
        //         }
        //     }
        //     t_temp->matrix[n] = max_index;
        // }
        argmax_2d_matrix(_t, &t_temp);
        delete_matrix(_t);
        _t = t_temp;
    }

    *loss = 0.0;
    for(int n=0;n<batchsize;n++){
        *loss -= log(_y->matrix2d[n][(int)(_t->matrix[n])]+MIN_NUM);
    }
    *loss /= batchsize;
    delete_matrix(_y);
    delete_matrix(_t);
}

void one_hot_encoding(Matrix *arr, Matrix **_dest,int min_index, int max_index){
    Matrix *dest = NULL;
    int index_num = max_index-min_index+1;
    int data_num = arr->shape[0];
    // printf("%d, %d, %d\n", index_num, max_index, min_index);
    init_matrix(_dest, 2, ZEROS, 0, data_num,index_num);
    dest = *_dest;

    for(int i=0;i<data_num;i++){
        dest->matrix2d[i][(int)(arr->matrix[i])-min_index] = 1;
    }
}

void categorical_encoding(Matrix *arr, Matrix **_dest){
    Matrix *dest = NULL;
    int index_num = arr->shape[1];
    int data_num = arr->shape[0];

    init_matrix(_dest, 1, ZEROS, 0, arr->shape[0]);
    dest = *_dest;

    for(int i=0;i<data_num;i++){
        for(int f=0;f<index_num;f++){
            if(arr->matrix2d[i][f]==1){
                dest->matrix[i] = f;
                break;
            }
        }
    }
}

void argmax_2d_matrix(Matrix *arr,Matrix **_dest){
    Matrix *dest = NULL;
    int index_num = arr->shape[1];
    int data_num = arr->shape[0];
    int max_index;
    double max_value, temp;

    init_matrix(_dest, 1, ZEROS, 0, arr->shape[0]);
    dest = *_dest;

    for(int i=0;i<data_num;i++){
        max_value = arr->matrix2d[i][0];
        max_index = 0;
        for(int f=0;f<index_num;f++){
            temp = arr->matrix2d[i][f];
            if(temp>max_value){
                max_value = temp;
                max_index = f;
            } 
        }
        dest->matrix[i] = (double)max_index;
    }
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void train_test_split_matrix(Matrix *data_x, Matrix *data_y,Matrix **train_x, Matrix **train_y, Matrix **test_x, Matrix **test_y, double train_ratio, bool stratify){
    int x_train_shape[7];
    int y_train_shape[7];
    int x_test_shape[7];
    int y_test_shape[7];
    int data_num = (data_x->shape[0]);
    int train_num = data_num*train_ratio;
    int test_num = data_num - train_num;
    int single_x_size = (data_x->size)/data_num;
    int single_y_size = (data_y->size)/data_num;
    int indices[data_num];
    int idx = 0;

    //shuffle index
    for(int i=0;i<data_num;i++){
        indices[i] = i;
    }
    for (int i = data_num - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        swap(&indices[i], &indices[j]); // Keep track of shuffled indices
    }

    //low memory testing -> cut data
    // data_num /=800;
    // train_num = data_num*0.8;
    // test_num = data_num - train_num;

    for(int i=0;i<(data_x->dim);i++){
        x_train_shape[i] = data_x->shape[i];
        y_train_shape[i] = data_y->shape[i];
        x_test_shape[i] = data_x->shape[i];
        y_test_shape[i] = data_y->shape[i];
    }
    x_train_shape[0] = train_num;
    y_train_shape[0] = train_num;
    x_test_shape[0] = test_num;
    y_test_shape[0] = test_num;

    init_by_shape_arr_matrix(train_x, x_train_shape, data_x->dim, NONE, 0);
    init_by_shape_arr_matrix(train_y, y_train_shape, data_y->dim, NONE, 0);
    init_by_shape_arr_matrix(test_x, x_test_shape, data_x->dim, NONE, 0);
    init_by_shape_arr_matrix(test_y, y_test_shape, data_y->dim, NONE, 0);

    if(stratify){
        //need codes
    }
    else{
        for(idx=0;idx<train_num;idx++){
            memcpy(((*train_x)->matrix) + single_x_size*idx, (data_x->matrix) + single_x_size*indices[idx], sizeof(double)*single_x_size);
            memcpy(((*train_y)->matrix) + single_y_size*idx, (data_y->matrix) + single_y_size*indices[idx], sizeof(double)*single_y_size);
        }
        for(;idx<data_num;idx++){
            int dest_idx = idx-train_num;
            memcpy(((*test_x)->matrix) + single_x_size*dest_idx, (data_x->matrix) + single_x_size*indices[idx], sizeof(double)*single_x_size);
            memcpy(((*test_y)->matrix) + single_y_size*dest_idx, (data_y->matrix) + single_y_size*indices[idx], sizeof(double)*single_y_size);
        }
    }
}

void init_batch(Matrix **_batch_x, Matrix **_batch_y,  const Matrix *x_sample, const Matrix *y_sample, int batchsize){
    Matrix* batch_x = NULL;
    Matrix* batch_y = NULL;
    int data_num = x_sample->shape[0];
    int single_x_size = (x_sample->size)/data_num;
    int single_y_size = (y_sample->size)/data_num;
    
    delete_matrix(*_batch_x);
    batch_x = (Matrix*)malloc(sizeof(Matrix));
    if(batch_x == NULL){
        printf("malloc failed\n");
        exit(0);
    }
    *_batch_x = batch_x;

    delete_matrix(*_batch_y);
    batch_y = (Matrix*)malloc(sizeof(Matrix));
    if(batch_y == NULL){
        printf("malloc failed\n");
        exit(0);
    }
    *_batch_y = batch_y;

    batch_x->dim = x_sample->dim;
    batch_y->dim = y_sample->dim;
    for(int i=1; i<x_sample->dim;i++){
        batch_x->shape[i] = x_sample->shape[i];
    }
    for(int i=1; i<y_sample->dim;i++){
        batch_y->shape[i] = y_sample->shape[i];
    }
    batch_x->shape[0] = batchsize;
    batch_y->shape[0] = batchsize;
    batch_x->size = batchsize*single_x_size;
    batch_y->size = batchsize*single_y_size;
    batch_x->connection_ptr = NULL;
    batch_y->connection_ptr = NULL;
}

void get_batch_by_index(int index, Matrix *batch_x, Matrix *batch_y,  const Matrix *x_sample, const Matrix *y_sample, int batchsize){
    int data_num = x_sample->shape[0];
    int single_x_size = (x_sample->size)/data_num;
    int single_y_size = (y_sample->size)/data_num;

    if(batch_x==NULL || batch_y == NULL){
        printf("error: batch is not initialized\n");
        exit(1);
    }
    batch_x->matrix = (x_sample->matrix)+index*batchsize*single_x_size;
    batch_y->matrix = (y_sample->matrix)+index*batchsize*single_y_size;
    make_connection_matrix(batch_x);
    make_connection_matrix(batch_y);

}

void cJSON_AddMatrixToObject(cJSON *object, const Matrix matrix, const char* matrix_name){
    cJSON *matrix_json = cJSON_CreateObject();

    cJSON_AddNumberToObject(matrix_json, "dim", matrix.dim);
    cJSON_AddNumberToObject(matrix_json, "size", matrix.size);



    cJSON *shape_json = cJSON_CreateArray();

    for (int i = 0; i < matrix.dim; i++) {
        // printf("segdetail1\n");
        cJSON *shape_item = cJSON_CreateNumber(matrix.shape[i]);
        cJSON_AddItemToArray(shape_json, shape_item);
    }

    cJSON_AddItemToObject(matrix_json, "shape", shape_json);

    //shape[7] int

    cJSON *matrix_value_json = cJSON_CreateArray();

    for (int i = 0; i < matrix.size; i++) {
        // printf("segdetail2\n");
        cJSON *matrix_item = cJSON_CreateNumber(matrix.matrix[i]);
        cJSON_AddItemToArray(matrix_value_json, matrix_item);
    }

    cJSON_AddItemToObject(matrix_json, "matrix", matrix_value_json);

    //matrix* double

    cJSON_AddItemToObject(object, matrix_name, matrix_json);
}

void removeBrackets(char *str) {
    int len = strlen(str);
    if (len >= 2 && str[0] == '[' && str[len - 1] == ']') {
        str[len - 1] = '\0'; 
        str++; 
    }
}

void cJSON_GetMatrix(cJSON *object, Matrix **matrix){
    int dim = cJSON_GetObjectItem(object,"dim")->valueint;
    int size = cJSON_GetObjectItem(object,"size")->valueint;
    cJSON *shape_json = cJSON_GetObjectItem(object,"shape");
    cJSON *matrix_json = cJSON_GetObjectItem(object,"matrix");
    int shape[7];

    for(int i =0;i<dim;i++){
        shape[i] = cJSON_GetArrayItem(shape_json, i)->valueint;
    }

    init_by_shape_arr_matrix(matrix, shape, dim, NONE, 0);
    
    // printf("getting matrix values.....\n");
    cJSON *matrix_string = cJSON_PrintUnformatted(matrix_json);
    removeBrackets(matrix_string);
    char *token;
    token = strtok(matrix_string,",");

    for(int i =0;i<size;i++){
        (*matrix)->matrix[i] = atof(token);
        // if(i%2000 ==0){
        //     printf("%d...\n", i);
        // }
        token = strtok(NULL,",");
        
    }
    cJSON_free(matrix_string);
    // print_matrix(*matrix, true);
    // printf("complete\n");

}

void saveToCSV(const char* filename, int rows, int cols, double** arr, const char* colNames[]) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    // 첫 번째 행에 열 이름 쓰기
    fprintf(file, "Index,");
    for (int j = 0; j < cols; j++) {
        fprintf(file, "%s", colNames[j]);
        if (j < cols - 1) {
            fprintf(file, ",");
        }
    }
    fprintf(file, "\n");
    
    // 데이터 쓰기
    for (int i = 0; i < rows; i++) {
        // 인덱스 쓰기
        fprintf(file, "%d,", i + 1);

        // 데이터 쓰기
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%lf", arr[i][j]);
            if (j < cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void saveToCSV_int(const char* filename, int rows, int cols, int** arr, const char* colNames[]) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    // 첫 번째 행에 열 이름 쓰기
    fprintf(file, "Index,");
    for (int j = 0; j < cols; j++) {
        fprintf(file, "%s", colNames[j]);
        if (j < cols - 1) {
            fprintf(file, ",");
        }
    }
    fprintf(file, "\n");
    
    // 데이터 쓰기
    for (int i = 0; i < rows; i++) {
        // 인덱스 쓰기
        fprintf(file, "%d,", i + 1);

        // 데이터 쓰기
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d", arr[i][j]);
            if (j < cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}
