using CUDA
using BenchmarkTools

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")
include("Visualisation.jl")

CUDA.allowscalar(false)

function gpu_bench_3D()
    # Warmup
    events, = read_3D("Triple_line_source.lmdT");
    output = reconstruct3D(events, 128, 128, 128, 1);

    visualise(output)

    device!(4)

    dimensions = [32, 128, 512]
    recon_iter = 20

    for d in dimensions
        bench = @benchmark begin
            CUDA.@sync begin
                events, = read_3D("Triple_line_source.lmdT");
                reconstruct3D(events, $d, $d, $d, $recon_iter);
            end;
        end setup=setup_fn()

        bench_time = BenchmarkTools.mean(bench).time/(1000000*1000); # Gives the time in seconds

        print("\n\nTotal GPU time on ($d)x($d)x($d) with $recon_iter iterations: $bench_time seconds \n\n\n");
    end
end

function setup_fn()
    GC.gc(true)
    CUDA.reclaim()
end
