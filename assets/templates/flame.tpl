#define xcommon(name) xcommon_##name
#define xaffine_a aff.a
#define xaffine_b aff.b
#define xaffine_c aff.c
#define xaffine_d aff.d
#define xaffine_e aff.e
#define xaffine_f aff.f

template<typename FloatT>
struct affine {
    FloatT a, d, b, e, c, f;

    __device__ void apply(FloatT& px, FloatT& py) const {
        auto tmp = a * px + b * py + c;
        py = d * px + e * py + f;
        px = tmp;
    }
};

<# for xform in xform_definitions #>
template<typename FloatT>
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

        __device__ void apply(FloatT& px, FloatT& py, FloatT& nx, FloatT& ny, randctx* rs) const {
            nx = 0; ny = 0;
            aff.apply(px, py);

            <# for common in vlink.common #>
            FloatT xcommon_@common.name@ = @common.source@;
            <# endfor #>

            <# for variation in vlink.variations #>
            // @variation.name@
            if(@variation.name@ != FloatT(0.0)) {
                FloatT weight = @variation.name@;
                @-get_variation_source(variation.id, "                ")@
            }
            <# endfor #>
        }               

    } vlink_@loop.index@;

    <# endfor #>
    __device__ void apply(FloatT& px, FloatT& py, FloatT& nx, FloatT& ny, randctx* rs) const {
        <# for vlink in xform.vchain #>
        vlink_@loop.index@.apply(px, py, nx, ny, rs);
            <# if not loop.is_last #>
        px = nx;
        py = ny;
            <# endif #>
        <# endfor #>
    }
};

<# endfor #>
template<typename FloatT>
struct flame_t {

    affine<FloatT> screen_space;
    FloatT weight_sum;

    <# for xform in xforms #>
    xform_@xform.hash@_t<FloatT> xform_@xform.id@;
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

    __device__ FloatT dispatch(int idx, Real& px, Real& py, Real& pc, Real& nx, Real& ny, Real& nc, randctx* rs) const {
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
};

static_assert(sizeof(flame_t<Real>) == flame_size_bytes);