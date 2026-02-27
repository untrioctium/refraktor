constexpr static uint64 num_xforms = @num_standard_xforms@;
constexpr static bool has_final_xform = @num_standard_xforms < length(xforms)@;
constexpr static uint32 affine_indices[] = {@join(affine_indices, ", ")@};
constexpr static uint32 num_affines = @length(affine_indices)@;

template<typename FloatT>
struct __align__(sizeof(FloatT)) affine {
    FloatT a, d, b, e, c, f;

    __device__ void apply(vec2<FloatT>& p) const {
        auto tmp = fl::fma(a, p.x, fl::fma(b, p.y, c));
        p.y = fl::fma(d, p.x, fl::fma(e, p.y, f));
        p.x = tmp;
    }
};

<# for hash, xform in xform_definitions #>
template<typename FloatT, typename RandCtx>
struct __align__(sizeof(FloatT)) xform_@hash@_t {

    FloatT weight;
    FloatT color;
    FloatT color_speed;
    FloatT opacity;

    <# for vlink in xform.vchain #>
    // vlink @loop.index@
    struct __align__(sizeof(FloatT))  {

        affine<FloatT> aff;

        <# for variation in vlink.variations #>
        FloatT v_@variation.name@;
            <# if length(variation.parameters) > 0 #>
            <# for parameter in variation.parameters #>
            FloatT p_@variation.name@_@parameter@;
            <# endfor #> 
            <# endif #>
            <# if length(variation.precalc) > 0 #>
            <# for parameter in variation.precalc #>
            FloatT p_@variation.name@_@parameter@;
            <# endfor #>
            <# endif #>
        <# endfor #> 

        __device__ void apply(const vec2<FloatT>& inp, vec2<FloatT>& outp, RandCtx* rs) const {
            outp.x = outp.y = 0;

            <# for common in vlink.common #>
            const FloatT common_@common.name@ = @common.source@;
            <# endfor #>
            <# for variation in vlink.variations #>
            
            // @variation.name@
            @-get_variation_source(variation.name, "            ")@
            <# endfor #>
        }

        __device__ void do_precalc(RandCtx* rs) {
            <# for variation in vlink.variations #>
            <# if variation_has_precalc(variation.name) #>
            // @variation.name@
            @-get_precalc_source(variation.name, "            ")@
            <# endif #>
            <# endfor #>
        }     

    } vlink_@loop.index@;

    <# endfor #>
    __device__ void apply(vec2<FloatT>& inp, vec2<FloatT>& outp, RandCtx* rs) const {
        <# for vlink in xform.vchain #>
        vlink_@loop.index@.aff.apply(inp);
        vlink_@loop.index@.apply(inp, outp, rs);
            <# if not loop.is_last #>
        inp = outp;
            <# endif #>
        <# endfor #>
    }

    __device__ void do_precalc(RandCtx* rs) {
        <# for vlink in xform.vchain #>
        vlink_@loop.index@.do_precalc(rs);
        <# endfor #>
    }
};

<# endfor #>
template<typename FloatT, typename RandCtx>
struct __align__(sizeof(FloatT)) flame_t {

    affine<FloatT> screen_space;
    affine<FloatT> plane_space;
    
    FloatT weight_sum;

    FloatT cdf[@num_standard_xforms@];

    <# if use_chaos #>
    struct __align__(sizeof(FloatT)) {

        FloatT weight_sum;
        FloatT weights[@num_standard_xforms@];

    } chaos[@num_standard_xforms@];
    <# endif #>

    <# for xform in xforms #>
    xform_@xform.hash@_t<FloatT, RandCtx> xform_@xform.id@;
    <# endfor #>

    __device__ unsigned short select_xform(FloatT ratio) const {
        unsigned short lo = 0, hi = num_xforms - 1;
        while(lo < hi) {
            unsigned short mid = (lo + hi) >> 1;
            if(cdf[mid] < ratio)
                lo = mid + 1;
            else
                hi = mid;
        }

        return lo;
    }

    <# if use_chaos #>

    __device__ unsigned short select_xform(unsigned char last, FloatT ratio) const {
        if(last == static_cast<unsigned char>(255)) return select_xform(ratio);

        const auto& weights = chaos[last].weights;
        FloatT rsum = FloatT(0.0);
        ratio *= chaos[last].weight_sum;
        unsigned char last_nonzero = 0;
        
        <# for xid in range(num_standard_xforms) #>
            <# if not loop.is_last #>
        if( weights[@xid@] != FloatT(0.0) && (rsum + weights[@xid@]) >= ratio) return @loop.index@; else { rsum += weights[@xid@]; if(weights[@xid@] != FloatT(0.0)) last_nonzero=@loop.index@;}
            <# else #>
        return ( weights[@xid@] != FloatT(0.0))? @loop.index@ : last_nonzero;
            <# endif #>
        <# endfor #>
    }

    <# endif #>

    __device__ FloatT dispatch(unsigned short idx, vec3<Real>& inp, vec3<Real>& outp, RandCtx* rs) const {
        switch(idx) {
            default: __builtin_unreachable();

            <# for xform in xforms #>
            case @loop.index@:
            {
                xform_@xform.id@.apply(inp.as_vec2(), outp.as_vec2(), rs);
                outp.z = INTERP(inp.z, xform_@xform.id@.color, xform_@xform.id@.color_speed);
                return xform_@xform.id@.opacity;
            }
            <# endfor #>
        }
        __builtin_unreachable();
    }

    __device__ FloatT* as_array() {
        return reinterpret_cast<FloatT*>(this);
    }

    __device__ void do_precalc(RandCtx* rs) {
        FloatT acc = FloatT(0.0);
        <# for xid in range(num_standard_xforms) #>
        acc += xform_@xid@.weight; cdf[@xid@] = acc;
        <# endfor #>

        weight_sum = acc;

        for(uint32 i = 0; i < @num_standard_xforms@; i++)
            cdf[i] /= weight_sum;


        <# if use_chaos #>
        for(int i = 0; i < @num_standard_xforms@; i++) {
            <# for xid2 in range(num_standard_xforms) #>
            chaos[i].weights[@xid2@] *= xform_@xid2@.weight;
            chaos[i].weight_sum += chaos[i].weights[@xid2@];
            <# endfor #>
        }
        <# endif #>

        // jitter screen_space a bit for antialiasing
        auto jitter = rs->randgauss(0.5f);
        screen_space.c += jitter.x;
        screen_space.f += jitter.y;

        <# for xform in xforms #>
        xform_@xform.id@.do_precalc(rs);
        <# endfor #>
    }
    
};

//static_assert(sizeof(flame_t<Real>) == flame_size_bytes);