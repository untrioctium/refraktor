constexpr static uint64 num_xforms = @num_standard_xforms@;
constexpr static bool has_final_xform = @num_standard_xforms < length(xforms)@;

template<typename FloatT>
struct affine {
    FloatT a, d, b, e, c, f;

    __device__ void apply(FloatT& px, FloatT& py) const {
        auto tmp = fl::fma(a, px, fl::fma(b, py, c));
        py = fl::fma(d, px, fl::fma(e, py, f));
        px = tmp;
    }

    void depolar() {
        FloatT ang = a;
        FloatT mag = exp(b);

        a = mag * cos(ang);
        b = mag * sin(ang);

        ang = d;
        mag = exp(e);

        d = mag * cos(ang);
        e = mag * sin(ang);
    }
};

<# for hash, xform in xform_definitions #>
template<typename FloatT, typename RandCtx>
struct xform_@hash@_t {

    FloatT weight;
    FloatT color;
    FloatT color_speed;
    FloatT opacity;

    <# for vlink in xform.vchain #>
    // vlink @loop.index@
    struct {

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
            aff.depolar();

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
        vlink_@loop.index@.aff.apply(inp.x, inp.y);
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
struct flame_t {

    affine<FloatT> screen_space;
    affine<FloatT> plane_space;
    
    FloatT weight_sum;

    <# if use_chaos #>
    struct {

        FloatT weight_sum;
        FloatT weights[@num_standard_xforms@];

    } chaos[@num_standard_xforms@];
    <# endif #>

    <# for xform in xforms #>
    xform_@xform.hash@_t<FloatT, RandCtx> xform_@xform.id@;
    <# endfor #>

    __device__ unsigned short select_xform(FloatT ratio) const {
        ratio *= weight_sum;
        FloatT rsum = FloatT(0.0);
        unsigned char last_nonzero = 0;

        <# for xid in range(num_standard_xforms) #>
            <# set xform=at(xforms, xid) #>
            <# if not loop.is_last #>
        if( xform_@xform.id@.weight != FloatT(0.0) && (rsum + xform_@xform.id@.weight) >= ratio) return @loop.index@; else { rsum += xform_@xform.id@.weight; if(xform_@xform.id@.weight != FloatT(0.0)) last_nonzero=@loop.index@;}
            <# else #>
        return ( xform_@xform.id@.weight != FloatT(0.0))? @loop.index@ : last_nonzero;
            <# endif #>
        <# endfor #>
    }

    <# if use_chaos #>

    __device__ unsigned short select_xform(unsigned short last, FloatT ratio) const {
        if(last == 2048) return select_xform(ratio);

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

        screen_space.depolar();
        plane_space.depolar();

        // calculate weight sum
        weight_sum = <# for xid in range(num_standard_xforms) #>xform_@xid@.weight<# if not loop.is_last #> +<# endif #><# endfor #>;

        <# if use_chaos #>
        for(int i = 0; i < @num_standard_xforms@; i++) {
            <# for xid2 in range(num_standard_xforms) #>
            chaos[i].weights[@xid2@] *= xform_@xid2@.weight;
            chaos[i].weight_sum += chaos[i].weights[@xid2@];
            <# endfor #>
        }
        <# endif #>

        // jitter screen_space a bit for antialiasing
        auto jitter = rs->randgauss(1/2.0f);
        screen_space.c += jitter.x;
        screen_space.f += jitter.y;

        <# for xform in xforms #>
        xform_@xform.id@.do_precalc(rs);
        <# endfor #>
    }
    
    void print_debug() {
        /*printf("flame_t\n");
        printf("  screen_space: { a: %f, d: %f, b: %f, e: %f, c: %f, f: %f }\n", screen_space.a, screen_space.d, screen_space.b, screen_space.e, screen_space.c, screen_space.f);
        printf("  plane_space: { a: %f, d: %f, b: %f, e: %f, c: %f, f: %f }\n", plane_space.a, plane_space.d, plane_space.b, plane_space.e, plane_space.c, plane_space.f);
        printf("  weight_sum: %f\n", weight_sum);
        <# if use_chaos #>
        <# for xid in range(num_standard_xforms) #>
        printf("  chaos@xid@:\n");
        printf("    weight_sum: %f\n", chaos[@xid@].weight_sum);
        <# for xid2 in range(num_standard_xforms) #>
        printf("    weights@xid2@: %f\n", chaos[@xid@].weights[@xid2@]);
        <# endfor #>
        <# endfor #>
        <# endif #>
        <# for xform in xforms #>
        <# set hash=xform.hash #>
        printf("  xform@xform.id@:\n");
        printf("    weight: %f\n", xform_@xform.id@.weight);
        printf("    color: %f\n", xform_@xform.id@.color);
        printf("    color_speed: %f\n", xform_@xform.id@.color_speed);
        printf("    opacity: %f\n", xform_@xform.id@.opacity);
        <# for vlink in at(xform_definitions, hash).vchain #>
        <# set vid=loop.index #>
        printf("    vlink@loop.index@:\n");
        printf("      aff: { a: %f, d: %f, b: %f, e: %f, c: %f, f: %f }\n", xform_@xform.id@.vlink_@loop.index@.aff.a, xform_@xform.id@.vlink_@loop.index@.aff.d, xform_@xform.id@.vlink_@loop.index@.aff.b, xform_@xform.id@.vlink_@loop.index@.aff.e, xform_@xform.id@.vlink_@loop.index@.aff.c, xform_@xform.id@.vlink_@loop.index@.aff.f);
        <# for variation in vlink.variations #>
        printf("      v_@variation.name@: %f\n", xform_@xform.id@.vlink_@vid@.v_@variation.name@);
        <# if length(variation.parameters) > 0 #>
        <# for parameter in variation.parameters #>
        printf("      p_@variation.name@_@parameter@: %f\n", xform_@xform.id@.vlink_@vid@.p_@variation.name@_@parameter@);
        <# endfor #> 
        <# endif #>
        <# if length(variation.precalc) > 0 #>
        <# for parameter in variation.precalc #>
        printf("      p_@variation.name@_@parameter@: %f\n", xform_@xform.id@.vlink_@vid@.p_@variation.name@_@parameter@);
        <# endfor #>
        <# endif #>
        <# endfor #> 
        <# endfor #>
        <# endfor #>*/
    }
    
};

//static_assert(sizeof(flame_t<Real>) == flame_size_bytes);