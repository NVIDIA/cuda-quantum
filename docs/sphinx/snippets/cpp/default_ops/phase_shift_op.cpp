#include <cudaq.h>

int main() {
    cudaq::qvector<4> q(1);
    phase_shift(q[0], 0.17);
    
    return 0;
}