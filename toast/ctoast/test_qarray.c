/*
test the implementation of quaternion arrays.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <pytoast.h>


void print_qarray(const int verbose, const int n, const double* array, const char array_name[]) {
  int i, j;
  if (verbose == 1) {
    printf("%s = \n",array_name);
    for (i = 0; i < n; ++i)
    {
      for (j = 0; j < 4; ++j)
      {
        printf("%f ", array[4*i+j]);
      } 
      printf("\n");
    }
    printf("\n");
  }
}


void print_array(const int verbose, const int n, const int m, const double* array, const char array_name[]) {
  int i, j;
  if (verbose == 1) {
    printf("%s = \n",array_name);
    for (i = 0; i < n; ++i)
    {
      for (j = 0; j < m; ++j)
      {
        printf("%f ", array[m*i+j]);
      } 
      printf("\n");
    }
    printf("\n");
  }
}


int unittest_isequal(const int n, const int m, const double* array_1, const double* array_2) {
  int i, j;
  int diff_count = 0;
  double tol = 1e-6;
  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < m; ++j)
    {
      if (fabs(array_1[m*i + j]-array_2[m*i + j])>tol) ++diff_count;
    }
  }
  return diff_count;
}


int main(int argc, char *argv[]) {
    int i;
    int verb = 1; /* display all arrays */
    int n_time = 4;

    double *q = malloc( 2 * 4 * sizeof(double) );
    double *q1 = malloc( 2 * 4 * sizeof(double) );
    double *q_interp = malloc( n_time * 4 * sizeof(double) );

    double *time = malloc(2 * sizeof(double));
    double *targettime = malloc(4 * sizeof(double));

    time[0]=0;  time[1]=9;
    targettime[0]=0;  targettime[1]=3;  targettime[2]=4.5;  targettime[3]=9;

    for (i = 0; i < 8; ++i) q[i]=i+2; /* q = [ [2,3,4,5] ; [6,7,8,9] ] */

    /* Beware, tests ahead! */

    /* Testing the dot product */
    print_qarray(verb,2,q,"q (original)");
    double *dotprod = malloc(2 * sizeof(double));
    pytoast_qarraylist_dot(2,4,q,q,dotprod);
    print_array(verb,1,2,dotprod,"q*q dotprod");

    /* Testing the inverse function */
    pytoast_qinv(2,q);
    print_qarray(verb,2,q,"inv(q)");

    /* Testing the amplitude function */
    pytoast_qamplitude(2,4,q,dotprod);
    print_array(verb,1,2,dotprod,"amplitude of q");

    /* Testing the norm */
    pytoast_qnorm_inplace(2,4,q);
    print_qarray(verb,2,q,"unit q");

    /* Testing the qmult */
    for (i = 0; i < 8; ++i) q[i]=i+2; /* q = [ [2,3,4,5] ; [6,7,8,9] ] */
    print_qarray(verb,2,q,"q");
    
    pytoast_qmult(2,q,q,q1);
    print_qarray(verb,2,q1,"q multiplied by q");

    /* Testing the rotation */
    double *q_rot = malloc( 2 * 4 * sizeof(double) );
    double *v_in = malloc( 3 * sizeof(double) );
    double *v_out = malloc( 2 * 3 * sizeof(double) );
    v_in[0]=0.57734543;v_in[1]=0.30271255;v_in[2]=0.75831218;
    q_rot[4*0+0]=0.50487417;q_rot[4*0+1]=0.61426059;q_rot[4*0+2]=0.60118994;q_rot[4*0+3]=0.07972857;
    q_rot[4*1+0]=0.43561544;q_rot[4*1+1]=0.33647027;q_rot[4*1+2]=0.40417115;q_rot[4*1+3]=0.73052901;
    print_qarray(verb,2,q_rot,"q_rot");
    print_array(verb,1,3,v_in,"v (before rotation by q_rot)");
    
    pytoast_qrotate(1,v_in,q_rot,v_out);
    print_array(verb,1,3,v_out,"v (after rotation by q_rot only 1st row)");
    
    pytoast_qrotate(2,v_in,q_rot,v_out);
    print_array(verb,1,3,v_out,"v (after rotation by q_rot 1st & 2nd row)");

    /* Testing eulerian functions */
    print_qarray(verb,2,q,"q");
    
    pytoast_qexp(2,q,q_rot);
    print_qarray(verb,2,q_rot,"exp(q)");
    
    pytoast_qln(2,q,q_rot);
    print_qarray(verb,2,q_rot,"ln(q)");

    /* Testing slerp function */
    
    print_array(verb,1,2,time,"time");
    print_array(verb,1,4,targettime,"targettime");

    pytoast_qnorm_inplace(2,4,q);
    print_qarray(verb,2,q,"q (unit before interpolation)");

    pytoast_slerp(2, 4, time, targettime, q, q_interp);
    pytoast_slerp(2, 4, time, targettime, q, q_interp);
    pytoast_slerp(2, 4, time, targettime, q, q_interp);
    pytoast_slerp(2, 4, time, targettime, q, q_interp);
    print_qarray(verb,4,q_interp,"q_interp (slerp)");


    /* Benchmark */
    int n_loop_1 = 60000;
    double chrono_1 = (double)clock();
    for (i = 0; i < n_loop_1; i++) {
        pytoast_slerp(2, 4, time, targettime, q, q_interp);
    }
    chrono_1 = ((double)clock() - chrono_1)/CLOCKS_PER_SEC;
    printf("Time 1 (slerp) = %f s\n", chrono_1);


    int n_loop_2 = 30000000;
    double chrono_2 = (double)clock();
    for (i = 0; i < n_loop_2; ++i) {
        pytoast_qrotate(2,v_in,q_rot,v_out);
    }
    chrono_2 = ((double)clock() - chrono_2)/CLOCKS_PER_SEC;
    printf("Time 2 (rotate) = %f s\n", chrono_2);


    free(targettime);
    free(time);
    free(q_interp);
    free(q);

    return 0;
}