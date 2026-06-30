using Aqua
using Dewdrop

# Package-quality checks (Aqua), in their OWN file with a clean `using` --- not nested
# in the GPU-readiness testset (which loads JLArrays/GPUArrays and toggles allowscalar,
# needlessly widening the method tables the ambiguity subprocess sees).
@testset "Aqua quality" begin
    Aqua.test_all(
        Dewdrop;
        # persistent_tasks precompiles a wrapper package in a subprocess; the 5 s default
        # is the most common spurious failure on loaded/shared CI, so give it headroom.
        persistent_tasks = (tmax = 20,),
        # All other checks on with defaults (ambiguities, unbound_args, undefined_exports,
        # project_extras, stale_deps, deps_compat with check_weakdeps, piracies).
        #
        # FUTURE NOTE: when ext/ weakdeps are added (CUDA, KernelAbstractions, DimensionalData,
        # Makie, ...), deps_compat(check_weakdeps=true) REQUIRES a [compat] entry for each ---
        # add the bound in the same commit as the weakdep or this check fails.
    )
end
