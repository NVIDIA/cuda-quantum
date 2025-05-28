#include <cudaq.h>

int main() {
    cudaq::qvector<3> q(1);
    create(q[0]);
    
    return 0;
}