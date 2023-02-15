#define xcommon(name) xcommon_##name
#define xaffine_a aff.a
#define xaffine_b aff.b
#define xaffine_c aff.c
#define xaffine_d aff.d
#define xaffine_e aff.e
#define xaffine_f aff.f

constexpr static uint64 num_xforms = @num_standard_xforms@;
constexpr static bool has_final_xform = @num_standard_xforms < length(xforms)@;

template<typename FloatT>
struct affine {
    FloatT a, d, b, e, c, f;

    __device__ void apply(FloatT& px, FloatT& py) const {
        auto tmp = __fma(a, px, __fma(b, py, c));
        py = __fma(d, px, __fma(e, py, f));
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
        FloatT @variation.name@;
        <# endfor #>

        // parameters
        <# for parameter in vlink.parameters #>
        FloatT @parameter@;
        <# endfor #> 

        // precalc
        <# for parameter in vlink.precalc #>
        FloatT @parameter@;
        <# endfor #> 

        __device__ void apply(FloatT& px, FloatT& py, FloatT& nx, FloatT& ny, RandCtx* rs) const {
            nx = 0; ny = 0;
            aff.apply(px, py);

            <# for common in vlink.common #>
            FloatT xcommon_@common.name@ = @common.source@;
            <# endfor #>

            <# for variation in vlink.variations #>
            // @variation.name@
            {
                FloatT weight = @variation.name@;
                @-get_variation_source(variation.id, "                ")@
            }
            <# endfor #>
        }

        __device__ void do_precalc() {
            <# for variation in vlink.variations #>
            <# if variation_has_precalc(variation.id) #>
            @-get_precalc_source(variation.id, "            ")@
            <# endif #>
            <# endfor #>
        }     

    } vlink_@loop.index@;

    <# endfor #>
    __device__ void apply(FloatT& px, FloatT& py, FloatT& nx, FloatT& ny, RandCtx* rs) const {
        <# for vlink in xform.vchain #>
        vlink_@loop.index@.apply(px, py, nx, ny, rs);
            <# if not loop.is_last #>
        px = nx;
        py = ny;
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

    __device__ FloatT dispatch(int idx, Real& px, Real& py, Real& pc, Real& nx, Real& ny, Real& nc, RandCtx* rs) const {
        switch(idx) {
            default: __builtin_unreachable();

            <# for xform in xforms #>
            case @loop.index@:
            {
                xform_@xform.id@.apply(px, py, nx, ny, rs);
                nc = INTERP(pc, xform_@xform.id@.color, xform_@xform.id@.color_speed);
                return xform_@xform.id@.opacity;
            }
            <# endfor #>
        }
    }

    __device__ FloatT* as_array() {
        return reinterpret_cast<FloatT*>(this);
    }

    __device__ FloatT do_precalc() {
        // calculate weight sum
        weight_sum = <# for xid in range(num_standard_xforms) #>xform_@xid@.weight<# if not loop.is_last #> +<# endif #><# endfor #>;

        <# for xform in xforms #>
        xform_@xform.id@.do_precalc();
        <# endfor #>
    }
};

//static_assert(sizeof(flame_t<Real>) == flame_size_bytes);