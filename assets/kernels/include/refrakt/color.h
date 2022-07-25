__device__ uchar3 hsv_to_rgb( const Real h, const Real s, const Real v ) {
	auto f = [&](Real n) {
		auto k = fmodf(n + h / Real(60.0), Real(6.0));
		return v - v * s * max(Real(0.0), min(k, min(Real(4.0) - k, Real(1.0))));
	};

	return {
		static_cast<unsigned char>(f(Real(5.0)) * Real(255.0)),
		static_cast<unsigned char>(f(Real(3.0)) * Real(255.0)),
		static_cast<unsigned char>(f(Real(1.0)) * Real(255.0)),
	};
}