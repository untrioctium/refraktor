using Real = double;

namespace catmull {

//template<typename Real>
struct segment {
    Real a, b, c, d;
};

//template<typename Real>
auto generate_segment(Real p0, Real p1, Real p2, Real p3, Real tension, Real alpha) -> segment {

    Real t01 = pow(hypot(p1 - p0, 1), alpha);
    Real t12 = pow(hypot(p2 - p1, 1), alpha);
    Real t23 = pow(hypot(p3 - p2, 1), alpha);

    Real m1 = (Real(1.0) - tension) *
        (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)));
    Real m2 = (Real(1.0) - tension) *
        (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)));

    return segment{
        Real(2.0)  * (p1 - p2) + m1 + m2,
        Real(-3.0) * (p1 - p2) - m1 - m1 - m2,
        m1,
        p1
    };
}

}

//template<typename Real>
__global__ void generate_sample_coefficients( const Real* samples, const unsigned int sample_size, const unsigned int num_segments, catmull::segment* segments ) {
	
	const unsigned int grid_rank = threadIdx.x + blockIdx.x * blockDim.x;
	if(grid_rank > sample_size * num_segments) return;
	
	const unsigned int my_segment = grid_rank / sample_size;
	const unsigned int my_parameter = grid_rank % sample_size;
	const unsigned int sample_index = my_segment * sample_size + my_parameter;
	
	segments[sample_index] = catmull::generate_segment(samples[sample_index], samples[sample_index + sample_size], samples[sample_index + sample_size * 2], samples[sample_index + sample_size * 3], 0.0, 0.5);
}