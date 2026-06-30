using Dewdrop
using Test

# Generic streaming Poisson drive (src/PoissonSource.jl): `PoissonSource{S}` decorates ANY synapse model `S`
# with a once-per-step Poisson event generator, scattering events through the SAME `scatter!` + delay-buffer
# + `_deliver!` pipeline the network uses for real spikes --- so the postsynaptic kinetics are exactly `S`'s.
# Attachment to a postsynaptic population = the extconn's post indices fall in that population's global range.
# The generic streaming external drive (replacing bespoke per-model drive synapses), plus the targeted
# `drive!` builder verb.

_lif() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

@testset "PoissonSource: core decorator + attachment" begin
    arch = Dewdrop.CPU()
    N = 4
    # one virtual Poisson source wired to ONLY neuron 1; a big delta kick → each delivered event = a spike
    extconn = Dewdrop.SparseCSR(arch, [(1, 1, 25.0, 1)]; npre = 1, npost = N)
    mknet(rate; seed = UInt64(1)) = DewdropNetwork(_lif(), N; input = 0.0, tspan = (0.0, 200.0), arch = arch,
        projections = (Projection(PoissonSource(DeltaSynapse(), extconn; rate = rate, seed = seed),
                                  Dewdrop._empty_csr(arch, N)),))
    sol = solve(mknet(300.0), FixedStep(0.1); progress = false)
    @test sol.spike_count[1] > 0                       # neuron 1 is driven → fires
    @test all(==(0), sol.spike_count[2:4])             # neurons 2-4 unwired → silent (ATTACHMENT)
    @test solve(mknet(300.0), FixedStep(0.1); progress = false).spike_count == sol.spike_count   # reproducible
    @test all(==(0), solve(mknet(0.0), FixedStep(0.1); progress = false).spike_count)            # rate 0 → silent
    @test solve(mknet(300.0; seed = UInt64(2)), FixedStep(0.1); progress = false).spike_count != sol.spike_count
end

@testset "PoissonSource is generic over the inner synapse family" begin
    arch = Dewdrop.CPU()
    extconn = Dewdrop.SparseCSR(arch, [(1, 1, 30.0, 1)]; npre = 1, npost = 1)   # 1 source → 1 neuron, strong
    fires(syn) = solve(DewdropNetwork(_lif(), 1; input = 0.0, tspan = (0.0, 200.0), arch = arch,
        projections = (Projection(PoissonSource(syn, extconn; rate = 200.0, seed = UInt64(1)),
                                  Dewdrop._empty_csr(arch, 1)),)),
        FixedStep(0.1); progress = false).spike_count[1]
    @test fires(DeltaSynapse()) > 0                                     # delivered to V directly
    @test fires(CurrentSynapse(; τ = 5.0)) > 0                          # CUBA current
    @test fires(ConductanceSynapse(; τ = 5.0, Erev = 0.0)) > 0          # COBA conductance (Erev)
    @test fires(DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)) > 0   # dual-exp COBA
end

@testset "drive! verb attaches a Poisson source to a population" begin
    nb = network(; tspan = (0.0, 200.0))
    population!(nb, :E, _lif(), 5; input = 0.0)
    population!(nb, :I, _lif(), 5; input = 0.0)
    drive!(nb, :E, ConductanceSynapse(; τ = 5.0, Erev = 0.0);
        rate = 300.0, n_ext = 20, p = 0.6, weight = 2.0, delay = 1.0, seed = UInt64(1))
    net = build(nb)
    @test length(net.projections) == 1                       # one drive projection appended
    sol = solve(net, FixedStep(0.1); progress = false)
    @test all(>(0), sol.spike_count[net.subpops[:E]])        # :E driven → fires
    @test all(==(0), sol.spike_count[net.subpops[:I]])       # :I untouched → silent (attachment)
    @test solve(build(nb), FixedStep(0.1); progress = false).spike_count == sol.spike_count   # reproducible
end

@testset "PoissonSource renders cleanly (no CSR dump)" begin
    arch = Dewdrop.CPU()
    extconn = Dewdrop.SparseCSR(arch, [(1, 1, 1.0, 1), (2, 1, 1.0, 1)]; npre = 3, npost = 2)
    ps = PoissonSource(ConductanceSynapse(; τ = 5.0, Erev = 0.0), extconn; rate = 12.0)
    s = sprint(show, MIME"text/plain"(), ps)
    @test occursin("PoissonSource", s)
    @test occursin("ConductanceSynapse", s)              # the wrapped synapse is named
    @test occursin("12", s)                              # the rate
    @test occursin("n_ext", s) && occursin("3", s)       # source count from extconn.npre
    @test !occursin("SparseCSR", s)                      # NOT a raw CSR field dump
    @test occursin("PoissonSource(", sprint(show, ps))   # compact form
end
