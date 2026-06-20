using Dewdrop
using Test

# M0 contract (schedule) --- the within-step phase order is an explicit, inspectable,
# pinned object, not implicit control flow. For coupled conductance/adaptation models
# the order changes spike timing, so the default order is regression-pinned here and
# its *semantics* are validated against analytic results in M1/M2.
@testset "within-step schedule" begin
    s = Dewdrop.default_schedule()

    # canonical within-step order is explicit and pinned (regression guard)
    @test Dewdrop.phases(s) == (:deliver, :integrate, :threshold, :reset, :propagate, :record)
    @test length(s) == 6

    # schedules are explicit and inspectable, constructible in any order
    s2 = Dewdrop.Schedule(:integrate, :threshold)
    @test Dewdrop.phases(s2) == (:integrate, :threshold)
    @test s2 != s

    # ordering is meaningful: reordering yields a distinct schedule
    @test Dewdrop.Schedule(:a, :b) != Dewdrop.Schedule(:b, :a)
    @test Dewdrop.Schedule(:a, :b) == Dewdrop.Schedule(:a, :b)
end
