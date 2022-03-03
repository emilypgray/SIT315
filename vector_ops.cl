//ToDo: Add Comment (what is the purpose of this function? Where its going to get executed?)
__kernel void square_magnitude(const int size,
                      __global int* v1, __global int* v2, __global int* v3) {
    
    // Thread identifiers
    int i = get_global_id(0);   
 
    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d)\n ", globalIndex);

    if (i < size)

    v3[i] = v1[i] + v2[i] ;
}
