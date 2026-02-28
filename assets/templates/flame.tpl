constexpr static uint64 num_xforms = @num_standard_xforms@;
constexpr static bool has_final_xform = @final_idx > 0@;
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
    
    FloatT cdf[@num_standard_xforms@];

    <# if use_chaos #>
    FloatT chaos[@num_standard_xforms@][@num_standard_xforms@];
    <# endif #>

    <# for hash, def in xform_definitions #>
    <# if def.count > 0 #>
    xform_@hash@_t<FloatT, RandCtx> xform_group_@hash@[@def.count@];
    <# endif #>
    <# endfor #>

    <# if final_idx > 0 #>
    <# set final_xform = at(xforms, final_idx) #>
    xform_@final_xform.hash@_t<FloatT, RandCtx> xform_final;
    <# endif #>

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

        unsigned short lo = 0, hi = num_xforms - 1;
        while(lo < hi) {
            unsigned short mid = (lo + hi) >> 1;
            if(chaos[last][mid] < ratio)
                lo = mid + 1;
            else
                hi = mid;
        }

        return lo;
    }

    <# endif #>

    __device__ FloatT dispatch(unsigned short idx, vec3<Real>& inp, vec3<Real>& outp, RandCtx* rs) const {
        switch(idx) {
            default: __builtin_unreachable();
            
            <# for hash, def in xform_definitions #>
            <# if def.count > 0 #>
            <# for id in def.ids #>
            case @id.global@:
            <# endfor #>
            {
                const auto& xf = xform_group_@hash@[idx - @def.ids.0.global@];
                xf.apply(inp.as_vec2(), outp.as_vec2(), rs);
                outp.z = INTERP(inp.z, xf.color, xf.color_speed);
                return xf.opacity;
            }
            <# endif #>
            <# endfor #>

            <# if final_idx > 0 #>
            case @final_idx@: {
                xform_final.apply(inp.as_vec2(), outp.as_vec2(), rs);
                outp.z = INTERP(inp.z, xform_final.color, xform_final.color_speed);
                return xform_final.opacity;
            }
            <# endif #>
        }
        __builtin_unreachable();
    }

    __device__ FloatT* as_array() {
        return reinterpret_cast<FloatT*>(this);
    }

    __device__ void do_precalc(RandCtx* rs) {
        FloatT acc = FloatT(0.0);
        <# for hash, def in xform_definitions #>
        <# for id in def.ids #>
        acc += xform_group_@hash@[@id.local@].weight; cdf[@id.global@] = acc;
        <# endfor #>
        <# endfor #>

        for(uint32 i = 0; i < @num_standard_xforms@; i++)
            cdf[i] /= acc;


        <# if use_chaos #>
        for(int r = 0; r < @num_standard_xforms@; r++) {
            FloatT row_acc = FloatT(0.0);
            <# for hash, def in xform_definitions #>
            <# for id in def.ids #>
            row_acc += chaos[r][@id.global@] * xform_group_@hash@[@id.local@].weight;
            chaos[r][@id.global@] = row_acc;
            <# endfor #>
            <# endfor #>

            for(int c = 0; c < @num_standard_xforms@; c++) {
                chaos[r][c] /= row_acc;
            }

        }
        <# endif #>

        // jitter screen_space a bit for antialiasing
        auto jitter = rs->randgauss(0.5f);
        screen_space.c += jitter.x;
        screen_space.f += jitter.y;

        <# for hash, def in xform_definitions #>
        <# for id in def.ids #>
        xform_group_@hash@[@id.local@].do_precalc(rs);
        <# endfor #>
        <# endfor #>

        <# if final_idx > 0 #>
        xform_final.do_precalc(rs);
        <# endif #>

    }
    
};