using Dewdrop
using Test

# Generic per-member ("(N,B)") batched parameters --- the shared-connectome ensemble extended so a
# parameter may carry a per-member batch axis, resolved at the EXISTING `_resolve(m,i,b)` /
# `_bsyn_one(s,i,b)` seams (no new kernels). Built model-agnostically. Cases:
#   (a) a single-group `Heterogeneous` model runs in the (N,B) batch (per-neuron resolve under the batch axis);
#   (b) `BatchedModel` wraps ANY base + per-member (B) or per-(neuron,member) (N×B) overrides;
#   (c) synapse scalar params may be per-member (length B).

_adex() = AdEx(; C = 281.0, gL = 30.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -60.0,
    Vpeak = -40.0, a = 4.0, b = 80.0, τw = 144.0, tref = 2.0)
_net(model, N) = DewdropNetwork(model, N; input = 800.0, tspan = (0.0, 200.0))
_run(model, N; kw...) = solve(_net(model, N), FixedStep(0.1); progress = false, kw...)

@testset "batched per-member params" begin
    @testset "(a) Heterogeneous model runs in the (N,B) batch ≡ standalone" begin
        N = 8
        # E/I adaptation pattern: first half adapt, second half do not.
        hm = Heterogeneous(_adex(); b = [i <= N ÷ 2 ? 80.0 : 0.0 for i in 1:N])
        s = _run(hm, N)
        @test sum(s.spike_count) > 0
        bs = _run(hm, N; batch = 4)
        @test all(bs.spike_count[:, b] == s.spike_count for b in 1:4)     # every column ≡ standalone (deterministic)
    end

    @testset "(b) BatchedModel over a Heterogeneous base + per-member (B) override ≡ standalone" begin
        N = 8; B = 3
        gL = [i <= N ÷ 2 ? 30.0 : 22.0 for i in 1:N]          # per-neuron base structure (E/I leak), shared across members
        hbase = Heterogeneous(_adex(); gL = gL)
        VT_mem = [-52.0, -49.0, -46.0]                        # threshold swept per member (length B)
        bm = Dewdrop.BatchedModel(hbase, (; VT = VT_mem))
        bs = _run(bm, N; batch = B)
        @test bs.spike_count[:, 1] != bs.spike_count[:, 3]    # the members really differ
        for m in 1:B
            s = _run(Heterogeneous(_adex(); gL = gL, VT = fill(VT_mem[m], N)), N)
            @test bs.spike_count[:, m] == s.spike_count       # column m ≡ the member-m standalone (per-neuron gL + per-member VT)
        end
    end

    @testset "(b) BatchedModel per-(neuron,member) N×B override ≡ standalone" begin
        N = 8; B = 3
        isE = [i <= N ÷ 2 for i in 1:N]
        b_mem = [0.0, 150.0, 300.0]                           # spike-adaptation increment swept per member
        bmat = [isE[i] ? b_mem[m] : 0.0 for i in 1:N, m in 1:B]   # N×B: E gets a per-member b, I stays 0 (for free)
        bm = Dewdrop.BatchedModel(_adex(), (; b = bmat))
        bs = _run(bm, N; batch = B)
        @test bs.spike_count[:, 1] != bs.spike_count[:, 3]
        for m in 1:B
            s = _run(Heterogeneous(_adex(); b = bmat[:, m]), N)
            @test bs.spike_count[:, m] == s.spike_count       # column m ≡ the member-m standalone (per-neuron-per-member b)
        end
    end

    @testset "(c) per-member synapse `a` (conductance scale) takes effect ≡ standalone" begin
        N = 8; B = 2
        lif = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        mk(w) = (nb = network(; tspan = (0.0, 200.0));
            population!(nb, :E, lif, N; input = 0.25);
            project!(nb, :E => :E, FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0);
                p = 0.6, weight = w, delay = steps(1), seed = UInt64(1), allow_self = false);
            build(nb))
        s_rec = solve(mk(0.5), FixedStep(0.1); progress = false)       # excitatory recurrence on
        s_off = solve(mk(0.0), FixedStep(0.1); progress = false)       # recurrence off (zero weights, same structure)
        @test sum(s_rec.spike_count) > 0
        @test s_rec.spike_count != s_off.spike_count                    # the recurrent synapse matters
        a_base = Dewdrop._dualexp_a(1.0, 5.0)
        # ONE shared connectome (weight 0.5); member 1 gets the full conductance scale, member 2 gets a = 0.
        # a is linear in the deposited weight, so a = a_base ≡ weight 0.5 and a = 0 ≡ weight 0 (synapse off).
        bs = solve(mk(0.5), FixedStep(0.1); batch = B,
            syn_overrides = Dict(1 => (; a = [a_base, 0.0])), progress = false)
        @test bs.spike_count[:, 1] == s_rec.spike_count                 # a = a_base ≡ full recurrence
        @test bs.spike_count[:, 2] == s_off.spike_count                 # a = 0 ≡ recurrence off
    end

    @testset "(d) PoissonSource streaming drive runs in the (N,B) batch ≡ standalone" begin
        # a net sustained by streaming Poisson drives (a `PoissonSource` synapse); the batched
        # path must generate + scatter them per step into the (N,B) ring, SHARED across columns (same drive
        # realization), so each column reproduces the standalone solve. (Without it the batch is silent.)
        N = 8; B = 3
        lif = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        nb = network(; tspan = (0.0, 200.0))
        population!(nb, :E, lif, N; input = 0.0)
        drive!(nb, :E, DeltaSynapse(); rate = 200.0, n_ext = 12, p = 0.5, weight = 2.0,
            delay = steps(1), seed = UInt64(7))
        net = build(nb)
        s = solve(net, FixedStep(0.1); progress = false)               # standalone (the drive sustains it)
        @test sum(s.spike_count) > 0
        bs = solve(net, FixedStep(0.1); batch = B, progress = false)   # batched, shared drive
        @test all(bs.spike_count[:, b] == s.spike_count for b in 1:B)  # every column ≡ standalone
    end

    @testset "(e) `model_overrides` kwarg wraps the model ≡ per-member standalone" begin
        # the ergonomic entry point: pass the per-(neuron,member) field array directly to
        # `solve`, which wraps the network's model in a `BatchedModel` (no hand-built engine types).
        N = 8; B = 3
        isE = [i <= N ÷ 2 for i in 1:N]
        b_mem = [0.0, 150.0, 300.0]
        bmat = [isE[i] ? b_mem[m] : 0.0 for i in 1:N, m in 1:B]
        net = DewdropNetwork(_adex(), N; input = 800.0, tspan = (0.0, 200.0))
        bs = solve(net, FixedStep(0.1); batch = B, model_overrides = (; b = bmat), progress = false)
        for m in 1:B
            s = _run(Heterogeneous(_adex(); b = bmat[:, m]), N)
            @test bs.spike_count[:, m] == s.spike_count
        end
    end

    @testset "(f) batched :itot/:gtot recording ≡ scalar (materialised in-kernel)" begin
        # recording the E INPUT current (`itot`); the batched kernel must materialise itot/gtot
        # per (neuron, member) --- like the scalar fused path --- so each column's recorded trace equals the
        # standalone fused solve. (Previously rejected as kernel-local.)
        N = 8; B = 3
        lif = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        nb = network(; tspan = (0.0, 150.0))
        population!(nb, :E, lif, N; input = 0.3)
        project!(nb, :E => :E, FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0);
            p = 0.5, weight = 0.4, delay = steps(1), seed = UInt64(2), allow_self = false)
        net = build(nb)
        rec = (it = Trace(:itot), gt = Trace(:gtot))
        s = solve(net, FixedStep(0.1); backend = Fused(), record = rec, progress = false)
        bs = solve(net, FixedStep(0.1); batch = B, record = rec, progress = false)
        @test maximum(abs, s.record.it.data) > 0                  # itot non-trivial (input + recurrence)
        @test all(bs.record.it.data[:, b, :] == s.record.it.data for b in 1:B)   # column ≡ scalar fused itot
        @test all(bs.record.gt.data[:, b, :] == s.record.gt.data for b in 1:B)   # and gtot
    end
end
