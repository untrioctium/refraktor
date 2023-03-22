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
};

<# for xform in xform_definitions #>
template<typename FloatT, typename RandCtx>
struct xform_@xform.hash@_t {

    FloatT weight;
    FloatT color;
    FloatT color_speed;
    FloatT opacity;

    <# for vlink in xform.vchain #>
    // vlink @loop.index@
    struct {

        affine<FloatT> aff;

        // variations
        <# for variation in vlink.variations #>
        FloatT v_@variation.name@;
        <# endfor #>
        <# if length(vlink.parameters) > 0 #>

        // parameters
        <# for parameter in vlink.parameters #>
        FloatT p_@parameter@;
        <# endfor #> 
        <# endif #>
        <# if length(vlink.precalc) > 0 #>

        // precalc
        <# for parameter in vlink.precalc #>
        FloatT p_@parameter@;
        <# endfor #> 
        <# endif #>

        __device__ void apply(const vec2<FloatT>& inp, vec2<FloatT>& outp, RandCtx* rs) const {
            outp.x = outp.y = 0;

            <# for common in vlink.common #>
            const FloatT common_@common.name@ = @common.source@;
            <# endfor #>
            <# for variation in vlink.variations #>
            
            // @variation.name@
            @-get_variation_source(variation.id, "            ")@
            <# endfor #>
        }

        __device__ void do_precalc() {
            <# for variation in vlink.variations #>
            <# if variation_has_precalc(variation.id) #>
            // @variation.name@
            @-get_precalc_source(variation.id, "            ")@
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

    __device__ void do_precalc() {
        <# for vlink in xform.vchain #>
        vlink_@loop.index@.do_precalc();
        <# endfor #>
    }
};

<# endfor #>
template<typename FloatT, typename RandCtx>
struct flame_t {

    affine<FloatT> screen_space;
    affine<FloatT> plane_space;
    
    FloatT weight_sum;

    <# for xform in xforms #>
    xform_@xform.hash@_t<FloatT, RandCtx> xform_@xform.id@;
    <# endfor #>

    __device__ unsigned int select_xform(FloatT ratio) const {
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

    __device__ FloatT dispatch(int idx, vec3<Real>& inp, vec3<Real>& outp, RandCtx* rs) const {
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
    }

    __device__ FloatT* as_array() {
        return reinterpret_cast<FloatT*>(this);
    }

    __device__ void do_precalc() {
        // calculate weight sum
        weight_sum = <# for xid in range(num_standard_xforms) #>xform_@xid@.weight<# if not loop.is_last #> +<# endif #><# endfor #>;

        <# for xform in xforms #>
        xform_@xform.id@.do_precalc();
        <# endfor #>
    }
};

//static_assert(sizeof(flame_t<Real>) == flame_size_bytes);